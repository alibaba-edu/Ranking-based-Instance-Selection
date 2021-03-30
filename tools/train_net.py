# encoding: utf-8

import argparse
import torch
import os,random
import numpy as np
from ret_benchmark.config import cfg
from ret_benchmark.data import build_data,build_trainVal_data
from ret_benchmark.engine.trainer import do_train
from ret_benchmark.losses import build_loss
from ret_benchmark.modeling import build_model
from ret_benchmark.solver import build_lr_scheduler, build_optimizer
from ret_benchmark.utils.logger import setup_logger
from ret_benchmark.utils.checkpoint import Checkpointer
from tensorboardX import SummaryWriter

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(0)

def train(cfg):
    logger = setup_logger(name="Train", level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    train_loader = build_data(cfg, is_train=True)
    num_classes=max(set([int(i) for i in train_loader.dataset.label_list]))+1
    cfg.num_classes=num_classes
    criterion = build_loss(cfg.LOSSES.NAME, num_classes, cfg)
    train_loader = build_data(cfg, is_train=True)
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
   
    if isinstance(criterion,tuple):
        criterion,optimizer_center=criterion
        criterion=criterion.cuda()
        scheduler_center = build_lr_scheduler(cfg, optimizer_center)
    else:
        optimizer_center=None
        scheduler_center=None
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    val_loader = build_data(cfg, is_train=False)

    trainVal_loader=build_trainVal_data(cfg,val_loader[0].dataset)

    if cfg.LOSSES.NAME_XBM_LOSS!='same':
        criterion_xbm = build_loss(cfg.LOSSES.NAME_XBM_LOSS,num_classes,cfg)
    else:
        criterion_xbm = None

    logger.info(train_loader.dataset)
    logger.info(trainVal_loader.dataset)
    for x in val_loader:
        logger.info(x.dataset)


    arguments = dict()
    arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    ckp_save_path = os.path.join(cfg.SAVE_DIR, cfg.NAME)
    os.makedirs(ckp_save_path, exist_ok=True)
    checkpointer = Checkpointer(model, optimizer, scheduler, ckp_save_path)

    tb_save_path = os.path.join(cfg.TB_SAVE_DIR, cfg.NAME)
    os.makedirs(tb_save_path, exist_ok=True)
    writer = SummaryWriter(tb_save_path)

    do_train(
        cfg,
        model,
        train_loader,
        trainVal_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        scheduler_center,
        criterion,
        criterion_xbm,
        checkpointer,
        writer,
        device,
        arguments,
        logger,
    )


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a retrieval network")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="config file", default=None, type=str
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    train(cfg)
