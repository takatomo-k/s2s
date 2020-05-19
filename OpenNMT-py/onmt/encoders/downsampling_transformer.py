"""
Implementation of "Attention is All You Need"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask
from onmt.modules import PositionalEncoding
from onmt.utils import GaussianNoise

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0, downsampling=1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ds_layer = nn.Linear(d_model,int(d_model/downsampling)) if downsampling >1 else None
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        #import pdb;pdb.set_trace()
        b,l,d=inputs.size()
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        out = self.feed_forward(out)
        out = self.ds_layer(out).view(b,-1,d) if self.ds_layer is not None else out
        return out

    def update_dropout(self, dropout):
        self.self_attn.update_dropout(dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions, noise, enc_pooling):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.pe = PositionalEncoding(dropout, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.noise = GaussianNoise(noise)
        enc_pooling = enc_pooling.split(',')
        assert len(enc_pooling) == num_layers or len(enc_pooling) == 1
        if len(enc_pooling) == 1:
            enc_pooling = enc_pooling * num_layers
        enc_pooling = [int(p) for p in enc_pooling]
        self.enc_pooling = enc_pooling
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions,
                downsampling=self.enc_pooling[i]
                )
            for i in range(num_layers)])
        
    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.noisy_enc,
            opt.enc_pooling
            )

    def forward(self, src, lengths=None):
        #import pdb;pdb.set_trace()
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        
        emb = self.embeddings(src) if self.embeddings is not None else src
        emb = self.pe(self.noise(emb))
        out = emb.transpose(0, 1).contiguous()
        
        
          # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        
        for layer, ds in zip(self.transformer,self.enc_pooling):
            lengths=lengths+ds-(out.size(1)%ds)
            out = F.pad(out,(0,0,0,ds-(out.size(1)%ds)))
            mask = ~sequence_mask(lengths).unsqueeze(1)
            out = layer(out, mask)
            lengths =-(- lengths//ds)
            
        out = self.layer_norm(out)
        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout)