# -*- coding: utf-8 -*-
import glob
import os
import codecs
import math
#from multiprocessing_generator import ParallelGenerator

from collections import Counter, defaultdict
from itertools import chain, cycle

import torch
import torchtext.data
from torchtext.data import Field
from torchtext.vocab import Vocab
from torchtext.data.utils import RandomShuffler

from onmt.inputters.text_dataset import text_fields, TextMultiField
from onmt.inputters.image_dataset import image_fields
from onmt.inputters.audio_dataset import audio_fields
from onmt.inputters.mel_dataset import mel_fields

from onmt.utils.logging import logger
# backwards compatibility
from onmt.inputters.text_dataset import _feature_tokenize  # noqa: F401
from onmt.inputters.image_dataset import (  # noqa: F401
    batch_img as make_img)

import gc


# monkey-patch to make torchtext Vocab's pickleable
def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


Vocab.__getstate__ = _getstate
Vocab.__setstate__ = _setstate


def make_src(data, vocab):
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(src_size, len(data), src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[j, i, t] = 1
    return alignment


def make_tgt(data, vocab):
    tgt_size = max([t.size(0) for t in data]) + 1
    alignment = torch.zeros(tgt_size, len(data)).long()
    for i, sent in enumerate(data):
        alignment[:sent.size(0), i] = sent
    return alignment

def fields_getters(path):
    if "text" in path:
        return text_fields
    elif "img" in path: 
        return image_fields
    elif "audio" in path:
        return audio_fields
    elif "mel" in path:
        return mel_fields

def get_base_name(idx,total):
    if idx==0:
        return "src"
    elif idx==(total-1):
        return "tgt"
    elif total>3 :
        return "mid_"+str(idx)
    else :
        return "mid"

def get_fields(
    data_list,
    nfeats,
    pad='<blank>',
    bos='<s>',
    eos='</s>',
    dynamic_dict=False,
    truncate=None,
    ):
    """
    Args:
        src_data_type: type of the source input. Options are [text|img|audio].
        n_src_feats (int): the number of source features (not counting tokens)
            to create a :class:`torchtext.data.Field` for. (If
            ``src_data_type=="text"``, these fields are stored together
            as a ``TextMultiField``).
        n_tgt_feats (int): See above.
        pad (str): Special pad symbol. Used on src and tgt side.
        bos (str): Special beginning of sequence symbol. Only relevant
            for tgt.
        eos (str): Special end of sequence symbol. Only relevant
            for tgt.
        dynamic_dict (bool): Whether or not to include source map and
            alignment fields.
        src_truncate: Cut off src sequences beyond this (passed to
            ``src_data_type``'s data reader - see there for more details).
        tgt_truncate: Cut off tgt sequences beyond this (passed to
            :class:`TextDataReader` - see there for more details).

    Returns:
        A dict mapping names to fields. These names need to match
        the dataset example attributes.
    """
    
    fields = {}
    for idx, (data, nfeat) in enumerate(zip(data_list,nfeats)):
        base_name=get_base_name(idx,len(data_list))
        field_kwargs = {"n_feats": nfeat,
                        "include_lengths": True,
                        "pad": pad, "bos": bos, "eos": eos,
                        "truncate": truncate,
                        "base_name": base_name}
        
        fields[base_name] = fields_getters(data)(**field_kwargs)

    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = indices

    if dynamic_dict:
        src_map = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)
        fields["src_map"] = src_map

        align = Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)
        fields["alignment"] = align

    return fields

def filter_example(ex, use_src_len=True, use_tgt_len=True,
                   min_src_len=1, max_src_len=float('inf'),
                   min_tgt_len=1, max_tgt_len=float('inf')):
    """Return whether an example is an acceptable length.

    If used with a dataset as ``filter_pred``, use :func:`partial()`
    for all keyword arguments.

    Args:
        ex (torchtext.data.Example): An object with a ``src`` and ``tgt``
            property.
        use_src_len (bool): Filter based on the length of ``ex.src``.
        use_tgt_len (bool): Similar to above.
        min_src_len (int): A non-negative minimally acceptable length
            (examples of exactly this length will be included).
        min_tgt_len (int): Similar to above.
        max_src_len (int or float): A non-negative (possibly infinite)
            maximally acceptable length (examples of exactly this length
            will be included).
        max_tgt_len (int or float): Similar to above.
    """

    src_len = len(ex.src[0])
    tgt_len = len(ex.tgt[0])
    return (not use_src_len or min_src_len <= src_len <= max_src_len) and \
        (not use_tgt_len or min_tgt_len <= tgt_len <= max_tgt_len)


def _pad_vocab_to_multiple(vocab, multiple):
    vocab_size = len(vocab)
    if vocab_size % multiple == 0:
        return
    target_size = int(math.ceil(vocab_size / multiple)) * multiple
    padding_tokens = [
        "averyunlikelytoken%d" % i for i in range(target_size - vocab_size)]
    vocab.extend(Vocab(Counter(), specials=padding_tokens))
    return vocab


def _build_field_vocab(field, counter, size_multiple=1, **kwargs):
    # this is basically copy-pasted from torchtext.
    all_specials = [
        field.unk_token, field.pad_token, field.init_token, field.eos_token
    ]
    specials = [tok for tok in all_specials if tok is not None]
    #import pdb;pdb.set_trace()
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
    if size_multiple > 1:
        _pad_vocab_to_multiple(field.vocab, size_multiple)


def _load_vocab(vocab_path, name, counters, min_freq):
    # counters changes in place
    vocab = _read_vocab_file(vocab_path, name)
    vocab_size = len(vocab)
    logger.info('Loaded %s vocab has %d tokens.' % (name, vocab_size))
    for i, token in enumerate(vocab):
        # keep the order of tokens specified in the vocab file by
        # adding them to the counter with decreasing counting values
        counters[name][token] = vocab_size - i + min_freq
    return vocab, vocab_size


def _build_fv_from_multifield(multifield, counter, build_fv_args,
                              size_multiple=1):
    #import pdb;pdb.set_trace()

    try:
        for name, field in multifield:
            _build_field_vocab(
                field,
                counter,
                size_multiple=size_multiple,
                **build_fv_args)
            logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))
    except:
        _build_field_vocab(
                multifield,
                counter,
                size_multiple=size_multiple,
                **build_fv_args)
        logger.info(" * %s vocab size: %d." % ('tgt', len(multifield.vocab)))

def _build_fields_vocab(fields, counters, data_type, share_vocab,
                        vocab_size_multiple,
                        vocab_size, words_min_frequency):
    build_fv_args = defaultdict(dict)
    
    for idx ,counter in enumerate(counters):
        name = get_base_name(idx, len(fields)-1)
        build_fv_args[name] = dict(
            max_size=vocab_size, min_freq=words_min_frequency)
        multifield = fields[name]
        #import pdb;pdb.set_trace()
        _build_fv_from_multifield(
            multifield,
            counters[name],
            build_fv_args[name],
            size_multiple=vocab_size_multiple if not share_vocab else 1)
    
    return fields

def _merge_field_vocabs(src_field, tgt_field, vocab_size, min_freq,
                        vocab_size_multiple):
    # in the long run, shouldn't it be possible to do this by calling
    # build_vocab with both the src and tgt data?
    specials = [tgt_field.unk_token, tgt_field.pad_token,
                tgt_field.init_token, tgt_field.eos_token]
    merged = sum(
        [src_field.vocab.freqs, tgt_field.vocab.freqs], Counter()
    )
    merged_vocab = Vocab(
        merged, specials=specials,
        max_size=vocab_size, min_freq=min_freq
    )
    if vocab_size_multiple > 1:
        _pad_vocab_to_multiple(merged_vocab, vocab_size_multiple)
    src_field.vocab = merged_vocab
    tgt_field.vocab = merged_vocab
    assert len(src_field.vocab) == len(tgt_field.vocab)


def _read_vocab_file(vocab_path, tag):
    """Loads a vocabulary from the given path.

    Args:
        vocab_path (str): Path to utf-8 text file containing vocabulary.
            Each token should be on a line by itself. Tokens must not
            contain whitespace (else only before the whitespace
            is considered).
        tag (str): Used for logging which vocab is being read.
    """

    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            return [line.strip().split()[0] for line in f if line.strip()]


def batch_iter(data, batch_size, batch_size_fn=None, batch_size_multiple=1):
    """Yield elements from data in chunks of batch_size, where each chunk size
    is a multiple of batch_size_multiple.

    This is an extended version of torchtext.data.batch.
    """
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far >= batch_size:
            overflowed = 0
            if size_so_far > batch_size:
                overflowed += 1
            if batch_size_multiple > 1:
                overflowed += (
                    (len(minibatch) - overflowed) % batch_size_multiple)
            if overflowed == 0:
                yield minibatch
                minibatch, size_so_far = [], 0
            else:
                if overflowed == len(minibatch):
                    logger.warning(
                        "An example was ignored, more tokens"
                        " than allowed by tokens batch_size")
                else:
                    yield minibatch[:-overflowed]
                    minibatch = minibatch[-overflowed:]
                    size_so_far = 0
                    for i, ex in enumerate(minibatch):
                        size_so_far = batch_size_fn(ex, i + 1, size_so_far)
    if minibatch:
        yield minibatch


def _pool(data, batch_size, batch_size_fn, batch_size_multiple,
          sort_key, random_shuffler, pool_factor):
    for p in torchtext.data.batch(
            data, batch_size * pool_factor,
            batch_size_fn=batch_size_fn):
        p_batch = batch_iter(
            sorted(p, key=sort_key),
            batch_size,
            batch_size_fn=batch_size_fn,
            batch_size_multiple=batch_size_multiple)
        for b in random_shuffler(list(p_batch)):
            yield b


class OrderedIterator(torchtext.data.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 pool_factor=1,
                 batch_size_multiple=1,
                 yield_raw_example=False,
                 **kwargs):
        super(OrderedIterator, self).__init__(dataset, batch_size, **kwargs)
        self.batch_size_multiple = batch_size_multiple
        self.yield_raw_example = yield_raw_example
        self.dataset = dataset
        self.pool_factor = pool_factor
        
    def create_batches(self):
        if self.train:
            if self.yield_raw_example:
                self.batches = batch_iter(
                    self.data(),
                    1,
                    batch_size_fn=None,
                    batch_size_multiple=1)
            else:
                self.batches = _pool(
                    self.data(),
                    self.batch_size,
                    self.batch_size_fn,
                    self.batch_size_multiple,
                    self.sort_key,
                    self.random_shuffler,
                    self.pool_factor)
        else:
            self.batches = []
            for b in batch_iter(
                    self.data(),
                    self.batch_size,
                    batch_size_fn=self.batch_size_fn,
                    batch_size_multiple=self.batch_size_multiple):
                self.batches.append(sorted(b, key=self.sort_key))

    def __iter__(self):
        """
        Extended version of the definition in torchtext.data.Iterator.
        Added yield_raw_example behaviour to yield a torchtext.data.Example
        instead of a torchtext.data.Batch object.
        """
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a
                    # minibatch be sorted by decreasing order, which
                    #  requires reversing relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                if self.yield_raw_example:
                    yield minibatch[0]
                else:
                    yield torchtext.data.Batch(
                        minibatch,
                        self.dataset,
                        self.device)
            if not self.repeat:
                return


class MultipleDatasetIterator(object):
    """
    This takes a list of iterable objects (DatasetLazyIter) and their
    respective weights, and yields a batch in the wanted proportions.
    """
    def __init__(self,
                 iterables,
                 device,
                 opt):
        self.index = -1
        self.iterators = [iter(iterable) for iterable in iterables]
        self.iterables = iterables
        self.weights = opt.data_weights
        self.batch_size = opt.batch_size
        self.batch_size_fn = max_tok_len \
            if opt.batch_type == "tokens" else None
        self.batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
        self.device = "cuda" if device >= 0 else "cpu"
        # Temporarily load one shard to retrieve sort_key for data_type
        temp_dataset = torch.load(self.iterables[0]._paths[0])
        self.sort_key = temp_dataset.sort_key
        self.random_shuffler = RandomShuffler()
        self.pool_factor = opt.pool_factor
        del temp_dataset

    def _iter_datasets(self):
        for weight in self.weights:
            self.index = (self.index + 1) % len(self.iterators)
            for i in range(weight):
                yield self.iterators[self.index]

    def _iter_examples(self):
        for iterator in cycle(self._iter_datasets()):
            yield next(iterator)

    def __iter__(self):
        for minibatch in _pool(
                self._iter_examples(),
                self.batch_size,
                self.batch_size_fn,
                self.batch_size_multiple,
                self.sort_key,
                self.random_shuffler,
                self.pool_factor):
            minibatch = sorted(minibatch, key=self.sort_key, reverse=True)
            yield torchtext.data.Batch(minibatch,
                                       self.iterables[0].dataset,
                                       self.device)


class DatasetLazyIter(object):
    """Yield data from sharded dataset files.

    Args:
        dataset_paths: a list containing the locations of dataset files.
        fields (dict[str, Field]): fields dict for the
            datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: See :class:`OrderedIterator` ``device``.
        is_train (bool): train or valid?
    """

    def __init__(self, dataset_paths, fields, batch_size, batch_size_fn,
                 batch_size_multiple, device, is_train, pool_factor,
                 repeat=True, num_batches_multiple=1, yield_raw_example=False):
        self._paths = dataset_paths
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.batch_size_multiple = batch_size_multiple
        self.device = device
        self.is_train = is_train
        self.repeat = repeat
        self.num_batches_multiple = num_batches_multiple
        self.yield_raw_example = yield_raw_example
        self.pool_factor = pool_factor

    def _iter_dataset(self, path):
        cur_dataset = torch.load(path)
        logger.info('Loading dataset from %s, number of examples: %d' %
                    (path, len(cur_dataset)))
        cur_dataset.fields = self.fields
        cur_iter = OrderedIterator(
            dataset=cur_dataset,
            batch_size=self.batch_size,
            pool_factor=self.pool_factor,
            batch_size_multiple=self.batch_size_multiple,
            batch_size_fn=self.batch_size_fn,
            device=self.device,
            train=self.is_train,
            sort=False,
            sort_within_batch=True,
            repeat=False,
            yield_raw_example=self.yield_raw_example
        )
        for batch in cur_iter:
            self.dataset = cur_iter.dataset
            yield batch

        cur_dataset.examples = None
        gc.collect()
        del cur_dataset
        gc.collect()

    def __iter__(self):
        num_batches = 0
        paths = self._paths
        if self.is_train and self.repeat:
            # Cycle through the shards indefinitely.
            paths = cycle(paths)
    
        for path in paths:
            for batch in self._iter_dataset(path):
                yield batch
                num_batches += 1
        
        if self.is_train and not self.repeat and \
           num_batches % self.num_batches_multiple != 0:
            # When the dataset is not repeated, we might need to ensure that
            # the number of returned batches is the multiple of a given value.
            # This is important for multi GPU training to ensure that all
            # workers have the same number of batches to process.
            for path in paths:
                for batch in self._iter_dataset(path):
                    yield batch
                    num_batches += 1
                    if num_batches % self.num_batches_multiple == 0:
                        return


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def build_dataset_iter(corpus_type, fields, opt, batch_size, is_train=True, multi=False):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    dataset_paths = list(sorted(
                glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt')))
    if not dataset_paths:
        return None
    if multi:
        batch_size = 1
        batch_fn = None
        batch_size_multiple = 1
    else:
        batch_fn = max_tok_len \
            if is_train and opt.batch_type == "tokens" else None
        batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1

    device = "cuda" if opt.gpu_ranks else "cpu"
    _iter=DatasetLazyIter(
        dataset_paths,
        fields,
        batch_size,
        batch_fn,
        batch_size_multiple,
        device,
        is_train,
        opt.pool_factor,
        repeat=not opt.single_pass,
        num_batches_multiple=max(opt.accum_count) * opt.world_size,
        yield_raw_example=multi)
    if corpus_type =="test":
        return _iter, torch.load(dataset_paths[0])
    return _iter