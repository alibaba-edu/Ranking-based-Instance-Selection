
from torch.utils.data import DataLoader

from .collate_batch import collate_fn
from .datasets import BaseDataSet
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def build_data(cfg, is_train=True):
    transforms = build_transforms(cfg, is_train=is_train)
    if is_train:
        dataset = BaseDataSet(
            cfg.DATA.TRAIN_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE,is_train=True
        )
        if cfg.DATA.SAMPLE == "RandomIdentitySampler":
            sampler = RandomIdentitySampler(
                dataset=dataset,
                batch_size=cfg.DATA.TRAIN_BATCHSIZE,
                num_instances=cfg.DATA.NUM_INSTANCES,
                max_iters=cfg.SOLVER.MAX_ITERS,
            )
            data_loader = DataLoader(
                dataset,
                collate_fn=collate_fn,
                batch_sampler=sampler,
                num_workers=cfg.DATA.NUM_WORKERS,
                pin_memory=True,
            )
        elif cfg.DATA.SAMPLE == "Random":
            data_loader = DataLoader(
                dataset,
                collate_fn=collate_fn,
                shuffle=True,
                batch_size=cfg.DATA.TRAIN_BATCHSIZE,
                drop_last=True,
                num_workers=cfg.DATA.NUM_WORKERS,
                pin_memory=True,
            )
        else:
            assert False
        return data_loader
    else:
        all_data_loader = list()
        for x in [
            cfg.DATA.TEST_IMG_SOURCE,
            cfg.DATA.QUERY_IMG_SOURCE,
            cfg.DATA.PKUVID_IMG_SOURCE,
        ]:
            if len(x) != 0:
                dataset = BaseDataSet(x, transforms=transforms, mode=cfg.INPUT.MODE,is_train=False)
                data_loader = DataLoader(
                    dataset,
                    collate_fn=collate_fn,
                    shuffle=False,
                    batch_size=cfg.DATA.TEST_BATCHSIZE,
                    num_workers=cfg.DATA.NUM_WORKERS,
                    pin_memory=False,
                )
                all_data_loader.append(data_loader)
        return all_data_loader
def build_trainVal_data(cfg,val_loader):
    transforms = build_transforms(cfg, is_train=False)
    dataset = BaseDataSet(
        # cfg.DATA.TRAIN_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE,leng=len(val_loader),is_train=False
        cfg.DATA.TRAIN_IMG_SOURCE, transforms=transforms, mode=cfg.INPUT.MODE,leng=-1,is_train=False
    )
    data_loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=cfg.DATA.TEST_BATCHSIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=False,
    )
    return data_loader