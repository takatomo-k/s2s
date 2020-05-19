"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel,CascadeModel, TranscoderModel, SpeechAECModel,GoogleMultitaskModel, NeuralCascadeModel,AttentionPassingModel, ModelPostNet
from torch import nn


def str2model(key):
    if key =='cascade': return CascadeModel
    elif key =='osamura': return NeuralCascadeModel
    elif key =='transcode':return TranscoderModel
    elif key == 'spaec':return SpeechAECModel
    elif key == 'ap':return AttentionPassingModel
    elif key == 'google':return GoogleMultitaskModel
    elif key in {'single',"nmt","tts","asr"}:return NMTModel
    elif key == 'ncascade':return NeuralCascadeModel
    else: return ModelPostNet

__all__ = ["build_model_saver", "ModelSaver",
           "NMTModel","CascadeModel", "check_sru_requirement", "TranscoderModel","TranscoderModel2","SpeechAECModel","str2model"]
