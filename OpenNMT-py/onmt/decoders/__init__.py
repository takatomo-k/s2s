"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder


class DummyDecoder(object):
    def __init__(self, opt, embeddings):
       pass 
    
    @classmethod
    def from_opt(cls, opt, embeddings=None):
        pass

def str2dec (key):
    if key == "rnn": 
        return StdRNNDecoder
    elif key =="ifrnn": 
        return InputFeedRNNDecoder
    elif key == "cnn": 
        return CNNDecoder
    elif key== "transformer": 
        return TransformerDecoder
    else:
        return DummyDecoder

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
