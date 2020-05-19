"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import aeq, sequence_mask
from onmt.modules import PositionalEncoding
from onmt.utils import GaussianNoise

class DummyEncoder(EncoderBase):
    """
        Nothing to do
    """

    def __init__(self,):
        super(DummyEncoder, self).__init__()

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls()

    def forward(self, src, lengths=None):
        return src, None, lengths

    def update_dropout(self, dropout):
        pass