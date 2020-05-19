"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""

from onmt.inputters.inputter import MyDataset, MyDataLoader, Batch
from onmt.inputters.audio_dataset import AudioTransform
from onmt.inputters.text_dataset import TextTransform

__all__ = ['MyDataset','TextTransform','AudioTransform','MyDataLoader',"Batch"]
