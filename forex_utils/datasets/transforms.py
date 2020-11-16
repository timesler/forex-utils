import torch
from torch import nn
import numpy as np


__all__ = ['collate_fn', 'ts_tensor', 'TSModel', 'Change', 'ChangeEncode', 'TPSLEncode']


def collate_fn(batch):
    x, y, xindex, yindex = [], [], [], []
    for xi, yi in batch:
        x.append(xi.values.T)
        y.append(yi.values.T)
        xindex.append(xi.index)
        yindex.append(yi.index)
    x, y = np.array(x), np.array(y)

    x = ts_tensor(x, xindex)
    y = ts_tensor(y, yindex)

    return x, y


def ts_tensor(arr, index):
    t = torch.as_tensor(arr).float()
    t.index = index
    return t


class TSModel(nn.Module):

    def __init__(self, model):
        self.model = model
    
    def __call__(self, x):
        return self.model(x.data)


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
        return change.apply(self.apply, result_type='broadcast').iloc[[0]]
    
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
