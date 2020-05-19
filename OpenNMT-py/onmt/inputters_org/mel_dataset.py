# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchtext.data import Field
import numpy as np
from onmt.inputters.datareader_base import DataReaderBase
from torch.nn.utils.rnn import pad_sequence
from onmt.utils import get_spectrograms
import shutil

class MelDataReader(DataReaderBase):
    """Read audio data from disk.

    Args:
        sample_rate (int): sample_rate.
        window_size (float) : window size for spectrogram in seconds.
        window_stride (float): window stride for spectrogram in seconds.
        window (str): window type for spectrogram generation. See
            :func:`librosa.stft()` ``window`` for more details.
        normalize_audio (bool): subtract spectrogram by mean and divide
            by std or not.
        truncate (int or NoneType): maximum audio length
            (0 or None for unlimited).

    Raises:
        onmt.inputters.datareader_base.MissingDependencyException: If
            importing any of ``torchaudio``, ``librosa``, or ``numpy`` fail.
    """

    def __init__(self,truncate=None):
        self.truncate = truncate

    @classmethod
    def from_opt(cls, opt):
        return cls()

    
    

    def read(self, data, side):
        #import pdb;pdb.set_trace()
        """Read data into dicts.

        Args:
            data (str or Iterable[str]): Sequence of audio paths or
                path to file containing audio paths.
                In either case, the filenames may be relative to ``src_dir``
                (default behavior) or absolute.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            src_dir (str): Location of source audio files. See ``data``.

        Yields:
            A dictionary containing audio data for each line.
        """

        

        if isinstance(data, str):
            data = DataReaderBase._read_file(data)

        for i, line in enumerate(data):
            line = line.decode("utf-8").strip()
            yield {side: line, side + '_path': line, 'indices': i}


def mel_sort_key(ex):
    """Sort using duration time of the sound spectrogram."""
    return ex.src.size(1)


class MelSeqField(Field):
    """Defines an audio datatype and instructions for converting to Tensor.

    See :class:`Fields` for attribute descriptions.
    """

    def __init__(self, preprocessing=None, postprocessing=None,
                 include_lengths=False, batch_first=False, pad_index=0,
                 is_target=False):
        super(MelSeqField, self).__init__(
            sequential=True, use_vocab=False, init_token=None,
            eos_token=None, fix_length=False, dtype=torch.float,
            preprocessing=preprocessing, postprocessing=postprocessing,
            lower=False, tokenize=None, include_lengths=include_lengths,
            batch_first=batch_first, pad_token=pad_index, unk_token=None,
            pad_first=True, truncate_first=False, stop_words=None,
            is_target=is_target
        )
        self.r=5

    def extract_features(self, mel_path):
        # torchaudio loading options recently changed. It's probably
        # straightforward to rewrite the audio handling to make use of
        # up-to-date torchaudio, but in the meantime there is a legacy
        # method which uses the old defaults
        """
        if not os.path.exists("/tmp/btec/en/mel/"):
            os.makedirs("/tmp/btec/en/mel/")
        try:
            if not os.path.exists("/tmp/btec/en/mel/"+mel_path+".npy"):
                shutil.copyfile(mel_path, mel_path.replace(/project/nakamura-lab08/Work/takatomo-k/Data/btec/)"/tmp/btec/en/mel/"+mel_path+".npy")
        """
        try:
            mel = np.load(mel_path)
            #print("tmp",mel_path)
            return mel
        except:
            pass
        
        try:
            mel = np.load(mel_path)
            #print("load",mel_path)
            return mel
        except:
            try:
                mel, mag =get_spectrograms(mel_path.replace("mel","wav").replace("npy","wav"))
                np.save(mel_path,mel)
                np.save(mel_path.replace("mel","mag"),mag)
            #    print("gen",mel_path)
            except:
                print("error",mel_path)
        return mel

    def pad(self, minibatch):
        """Pad a batch of examples to the length of the longest example.

        Args:
            minibatch (List[torch.FloatTensor]): A list of audio data,
                each having shape 1 x n_feats x len where len is variable.

        Returns:
            torch.FloatTensor or Tuple[torch.FloatTensor, List[int]]: The
                padded tensor of shape ``(batch_size, 1, n_feats, max_len)``.
                and a list of the lengths if `self.include_lengths` is `True`
                else just returns the padded tensor.
        """
        #import pdb;pdb.set_trace()
        assert self.pad_first and not self.truncate_first \
            and not self.fix_length and self.sequential
        minibatch = list(minibatch)
        lengths=[]
        sounds=[]
        #import pdb;pdb.set_trace()
        for i,mel in enumerate(minibatch):
            mel = self.extract_features(mel[0])
            mel = torch.from_numpy(mel)
            mel = F.pad(mel,(0,0,0,self.r-mel.size(0)%self.r)) # len x dim
            mel = mel.view(-1,80*self.r)
            #if self.pad_first:
            #    mel = F.pad(mel,(0,0,1,0)) #pad
            lengths.append(mel.size(0))
            sounds.append(mel)
        #import pdb;pdb.set_trace()
        mel=pad_sequence(sounds)
        
        if self.include_lengths:
            #print("mel",mel.shape)
            return (mel, lengths)

        return mel

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has ``include_lengths=True``, a tensor of lengths will be
        included in the return value.

        Args:
            arr (torch.FloatTensor or Tuple(torch.FloatTensor, List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True. Examples have shape
                ``(batch_size, 1, n_feats, max_len)`` if `self.batch_first`
                else ``(max_len, batch_size, 1, n_feats)``.
            device (str or torch.device): See `Field.numericalize`.
        """

        assert self.use_vocab is False
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=torch.int, device=device)

        if self.postprocessing is not None:
            arr = self.postprocessing(arr, None)

        if self.sequential and not self.batch_first:
            arr = arr.permute(3, 0, 1, 2)
        if self.sequential:
            arr = arr.contiguous()
        arr = arr.to(device)
        if self.include_lengths:
            return arr, lengths
        return arr

    def vocab_cls(self,counter, max_size=None, min_freq=1, specials=['<pad>'], vectors=None, unk_init=None, vectors_cache=None, specials_first=True):
        return [0]*80
        
def mel_fields(**kwargs):
    mel = MelSeqField(pad_index=0, batch_first=True, include_lengths=True)
    return mel
