import torch

class XBM:
    def __init__(self, cfg):
        self.K = cfg.XBM.SIZE
        self.feats = torch.zeros(self.K, cfg.MODEL.HEAD.DIM).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.targets[:]=-1
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size