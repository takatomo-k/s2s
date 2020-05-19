""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
import torch
from onmt.utils.logging import logger
from statistics import mean

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, stats={}):
        #import pdb;pdb.set_trace()
        self.stats=stats
        self.start_time = time.time()


    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from onmt.utils.distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats


    def update(self, stat, update_n_src_words=False,):
        """
        Update statistics by suming values with another `Statistics` object
        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not
        """
        #import pdb;pdb.set_trace()
        for key, value in stat.stats.items():
            if key in self.stats:
                self.stats[key].append(value)
            else:
                self.stats[key]=[value]
        
    
    def get(self,key):
        return self.stats[key]

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.stats['accuracy'] / self.stats['n_words'])

    def p_accuracy(self):
        """ compute accuracy """
        return 100 * (self.p_n_correct / self.n_words)

    def L1(self):
        return self._L1/self.num_steps
        
    def xent(self):
        """ compute cross entropy """
        return self._xent / self.n_words

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self._xent / self.n_words, 100))
    
    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def msg(self, step_fmt, learning_rate, start):
        t = self.elapsed_time()
        msg = "Step "+ step_fmt+"; "
        #import pdb;pdb.set_trace()
        for key,value in self.stats.items():
            msg+=("%s: %6.3f; ")%(key,mean(value))
        msg+=("lr: %7.5f; ")%(learning_rate)
        #msg+=("steps: %7.5f; ")%(self.steps)
        
        msg+=("%6.0f sec")%(time.time() - start)
        return msg

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.
        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(self.msg(step_fmt, learning_rate, start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        for key,value in self.stats.items():
            writer.add_scalar(prefix + "/"+key, mean(value), step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
