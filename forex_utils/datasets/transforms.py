import torch
from torch import nn
from torch._six import container_abcs
import numpy as np
import pandas as pd


__all__ = ['Collate', 'ts_tensor', 'Change', 'ChangeEncode', 'TPSLEncode']


class Collate:
    
    def __init__(self, flatten=False):
        self.flatten = nn.Flatten() if flatten else nn.Identity()

    def __call__(self, batch):
        elem = batch[0]
        
        if isinstance(elem, torch.Tensor):
            return self.flatten(torch.stack(batch))
        
        if isinstance(elem, pd.DataFrame):
            return self([torch.as_tensor(b.values.T) for b in batch])
        
        elif isinstance(elem, container_abcs.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in batch should be of equal size')
            transposed = zip(*batch)
            return [self(samples) for samples in transposed]

        raise TypeError(f'Could not collate batch of type {type(elem)}')


def ts_tensor(arr, index):
    t = torch.as_tensor(arr).float()
    t.index = index
    return t


class Change:

    def __call__(self, x):
        return x.iloc[-1] - x.iloc[[0]]


class ChangeEncode:

    def __init__(self, spread):
        self.spread = spread

    def __call__(self, x):
        change = x.iloc[-1] - x.iloc[[0]]
        return change.apply(self.apply, result_type='broadcast')
    
    def apply(self, change):
        if change[0] > self.spread:
            action = 2
        elif change[0] < -self.spread:
            action = 0
        else:
            action = 1
        
        return action


class TPSLEncode:
    
    def __init__(self, tp, sl, spread):
        self.tp = tp
        self.sl = sl
        self.spread = spread
    
    def __call__(self, x):
        change = x - x.iloc[0]
        return change.apply(self.apply, result_type='broadcast').iloc[[0]].astype(int)
    
    def apply(self, change):
        buy_tp = min(np.where(change - self.spread > self.tp)[0], default=len(change))
        buy_sl = min(np.where(change - self.spread < -self.sl)[0], default=len(change))
        sell_tp = min(np.where(change + self.spread < -self.tp)[0], default=len(change))
        sell_sl = min(np.where(change + self.spread > self.sl)[0], default=len(change))

        if buy_tp == len(change) and sell_tp == len(change):
            action = 1
        elif buy_tp <= sell_tp and buy_tp < buy_sl:
            action = 2
        elif sell_tp < buy_tp and sell_tp < sell_sl:
            action = 0
        else:
            action = 1
        
        return action
