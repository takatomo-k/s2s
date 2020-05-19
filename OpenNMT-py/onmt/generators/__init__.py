"""Module defining encoders."""
from onmt.generators.text_generator import TextGenerator,SpeakerGenerator, AdversarialGenerator
from onmt.generators.speech_generator import SpeechGenerator
from torch import nn

class DummyGenerator(nn.Module):
    def __init__(self, model_opt, field, decoder):
        super(DummyGenerator, self).__init__()
        pass
    def forward(self, inputs):
        return inputs

str2gen={'asr':TextGenerator, 'nmt':TextGenerator, 'tts': SpeechGenerator, 'vocoder':DummyGenerator}

__all__ = ["TextGenerator", "WldGenerator","SpeakerGenerator","AdversarialGenerator", 'DummyGenerator', 'str2gen']
