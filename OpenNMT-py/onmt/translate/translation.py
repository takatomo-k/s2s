""" Translation main class """
from __future__ import unicode_literals, print_function

import torch
from onmt.inputters import TextTransform

class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.
    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`
    Args:
       data (onmt.inputters.Dataset): Data.
       fields (List[Tuple[str, torchtext.data.Field]]): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, fields, n_best=1, replace_unk=False,
                 has_tgt=False, phrase_table=""):
        #self.data = data
        #import pdb;pdb.set_trace()
        self.fields = fields
        self._has_text_src = False #isinstance(fields['src'],TextTransform)
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table = phrase_table
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn, context):
        #import pdb;pdb.set_trace()
        tgt_field = self.fields#["tgt"]
        
        tokens = []
        for tok in pred:
            tokens.append(tgt_field.itos(tok))
            
            if tokens[-1] == "</s>":
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == "<unk>":
                    _, max_index = attn[i].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table != "":
                        with open(self.phrase_table, "r") as f:
                            for line in f:
                                if line.startswith(src_raw[max_index.item()]):
                                    tokens[i] = line.split('|||')[1].strip()
        return tokens

    def from_batch(self, translation_batch):
        
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, context, probs, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        translation_batch["context"],
                        translation_batch["probs"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices)
        if self._has_text_src:
            src = batch.src[0][:, :, 0].index_select(1, perm)
        else:
            src = None
        tgt = batch.tgt[0][:, :, 0].index_select(1, perm)
        #import pdb;pdb.set_trace()
            
        translations = []
        for b in range(batch_size):
            src_vocab = None
            src_raw = None
            pred_sents = [self._build_target_tokens(
                src[:, b] if src is not None else None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n], context[b][n])
                for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[1:, b] if tgt is not None else None, None, None)

            translation = Translation(
                src[:, b] if src is not None else None,
                src_raw, pred_sents, attn[b], context[b], probs[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.
    Attributes:
        src (LongTensor): Source word IDs.
        src_raw (List[str]): Raw source words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        attns (List[FloatTensor]) : Attention distribution for each
            translation.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
    """

    __slots__ = ["src", "src_raw", "pred_sents", "attns","context", "probs", "pred_scores",
                 "gold_sent", "gold_score"]

    def __init__(self, src, src_raw, pred_sents,
                 attn, context, probs, pred_scores, tgt_sent, gold_score):
        
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.context = context
        self.probs=probs
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """
        
        msg = {"src":self.src_raw}

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg.update({'pred': pred_sent})
        msg.update({'pred score': best_score})

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.update({'tgt': tgt_sent})
            msg.update({'gold score': self.gold_score})
        if len(self.pred_sents) > 1:
            tmp = []
            for score, sent in zip(self.pred_scores, self.pred_sents):
                tmp.append([score, sent])
            msg.update({'BEST HYP':tmp})
        return msg