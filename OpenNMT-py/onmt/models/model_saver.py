import os
import shutil
import torch
import torch.nn as nn
from datetime import datetime
from collections import deque
from onmt.utils.logging import logger, make_log_path

from copy import deepcopy


def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             opt,
                             fields,
                             optim,
                             opt.keep_checkpoint)
    return model_saver

class ModelSaverBase(object):
    """Base class for model saving operations

    Inherited classes must implement private methods:
    * `_save`
    * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, opt, fields, optim,
                 keep_checkpoint=-1):

        self.base_path = os.path.join("exp/model/", make_log_path(opt))
        #import pdb; pdb.set_trace()
        if not os.path.exists(self.base_path):
            logger.info("Create saving model path %s" % ( self.base_path))
            os.makedirs(self.base_path)
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.last_saved_step = None
        self.keep_checkpoint = keep_checkpoint
        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)
    
    def save(self, step, prefix, valid_loss, test_score, moving_average=None):
        """Main entry point for model saver

        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """

        if self.keep_checkpoint == 0 or step == self.last_saved_step:
            return

        if moving_average:
            save_model = deepcopy(self.model)
            for avg, param in zip(moving_average, save_model.parameters()):
                param.data.copy_(avg.data)
        else:
            save_model = self.model

        chkpt, chkpt_name = self._save(step, save_model, prefix, valid_loss, test_score)
        self.last_saved_step = step

        if moving_average:
            del save_model

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step, prefix, valid_loss, test_score):
        """Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            (object, str):

            * checkpoint: the saved object
            * checkpoint_name: name (or path) of the saved checkpoint
        """

        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """

        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model, prefix, valid_loss, test_score):
        real_model = (model.module
                      if isinstance(model, nn.DataParallel)
                      else model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        #import pdb;pdb.set_trace()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': self.fields,
            'opt': self.model_opt,
            'optim': self.optim.state_dict(),
            'valid_loss': valid_loss,
            'test_score': test_score
        }

        checkpoint_path = os.path.join(self.base_path, prefix+".pt")
        try:
            os.remove(checkpoint_path+"*")
        except:
            pass
        #checkpoint_path += ("_"+str(step)+".pt")
        logger.info("Saving checkpoint %d step %s" % ( step, checkpoint_path))
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)
