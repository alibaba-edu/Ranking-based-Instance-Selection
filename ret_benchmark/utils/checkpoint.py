import logging
import os
import glob
import torch
from ret_benchmark.utils.model_serialization import load_state_dict


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name):
        if not self.save_dir:
            return

        data = {}

        if not hasattr(self.model, "module"):
            data["model"] = self.model.state_dict()
        else:
            data["model"] = self.model.module.state_dict()

        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        
        iteration=int(f[-len('035000.pth'):-len('.pth')])
        self.logger.info("Loading checkpoint from {}, start from iteration={}".format(f,iteration))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        
        # return any further checkpoint data
        return checkpoint,iteration

    def has_checkpoint(self):
        save_file = sorted(glob.glob(os.path.join(self.save_dir, "model_*.pth")))
        return len(save_file)>0

    def get_checkpoint_file(self):
        save_file = sorted(glob.glob(os.path.join(self.save_dir, "model_*.pth")))
        return save_file[-1]

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))
