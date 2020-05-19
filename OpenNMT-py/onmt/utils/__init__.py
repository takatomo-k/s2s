"""Module defining various utilities."""
from onmt.utils.misc import split_corpus, aeq, use_gpu, set_random_seed
from onmt.utils.report_manager import ReportMgr, build_report_manager
from onmt.utils.statistics import Statistics
from onmt.utils.optimizers import MultipleOptimizer, \
    Optimizer, AdaFactor
from onmt.utils.earlystopping import EarlyStopping, scorers_from_opts
from tools.extract_features import get_spectrograms
from onmt.utils.gaussian_noise import GaussianNoise
from onmt.utils.evaluation import bleu, wer
from onmt.utils.loss import TextLossCompute


__all__ = ["split_corpus", "aeq", "use_gpu", "set_random_seed", "ReportMgr",
           "build_report_manager", "Statistics",
           "MultipleOptimizer", "Optimizer", "AdaFactor", "EarlyStopping",
           "scorers_from_opts", "get_spectrograms", "GaussianNoise", "bleu",
           "wer","TextLossCompute"]