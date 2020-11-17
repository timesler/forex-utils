import torch
from torch import nn
from functools import lru_cache


class Profit(nn.Module):
    def __init__(
        self, spread=0.0001, lot_size=100000, min_lot=0.0, reduce=True, is_loss=False, dim=2
    ):
        super().__init__()
        self.spread = spread
        self.lot_size = lot_size
        self.min_lot = min_lot
        self.reduce = reduce
        self.is_loss = is_loss
        self.dim = dim

    def forward(self, y_pred, change):
        position = y_pred
        profit = position_profit(
            position,
            change,
            min_lot=self.min_lot,
            lot_size=self.lot_size,
            spread=self.spread,
            dim=self.dim,
        )
        if self.reduce:
            profit = profit.mean()
        if self.is_loss:
            profit = -profit
        return profit


class SharpeRatio(nn.Module):
    def __init__(
        self,
        spread=0.0001,
        lot_size=100000,
        min_lot=0.0,
        exp_penalty=0,
        prof_penalty=0,
        reduce=True,
        is_loss=False,
        dim=2,
    ):
        super().__init__()
        self.spread = spread
        self.lot_size = lot_size
        self.min_lot = min_lot
        self.exp_penalty = exp_penalty
        self.prof_penalty = prof_penalty
        self.reduce = reduce
        self.is_loss = is_loss
        self.dim = dim

    def forward(self, y_pred, change):
        position = y_pred
        exposure = position.cumsum(dim=self.dim)
        sharpe, profit = sharpe_ratio(
            position,
            change,
            min_lot=self.min_lot,
            lot_size=self.lot_size,
            spread=self.spread,
            return_profit=True,
            dim=self.dim,
        )
        sharpe = sharpe - self.exp_penalty * exposure.abs() + self.prof_penalty * profit
        if self.reduce:
            sharpe = sharpe.mean()
        if self.is_loss:
            sharpe = -sharpe
        return sharpe


class MeanExposure(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, y_pred, change):
        position = y_pred
        exposure = position.cumsum(dim=self.dim)
        return exposure.abs().mean()


class Volume(nn.Module):
    def __init__(self, min_lot=0.0, reduce=True, dim=2):
        super().__init__()
        self.min_lot = min_lot
        self.reduce = reduce
        self.dim = dim

    def forward(self, y_pred, change):
        position = y_pred
        exposure = position.cumsum(dim=self.dim)
        lagged_exposure = exposure - position

        # Round exposure to nearest allowable lot unit
        exposure, lagged_exposure = round_exposure(exposure, lagged_exposure, self.min_lot)
        position = exposure - lagged_exposure

        # Calculate volume traded in instrument units
        volume = spread_cost(exposure, lagged_exposure, position)

        if self.reduce:
            volume = volume.mean()

        return volume


class Cost(nn.Module):
    def __init__(self, min_lot=0.0, lot_size=100000, spread=0.0001, reduce=True, dim=2):
        super().__init__()
        self.min_lot = min_lot
        self.lot_size = lot_size
        self.spread = spread
        self.reduce = reduce
        self.dim = dim

    def forward(self, y_pred, change):
        position = y_pred
        exposure = position.cumsum(dim=self.dim)
        lagged_exposure = exposure - position
        exposure, lagged_exposure = round_exposure(exposure, lagged_exposure, self.min_lot)
        position = exposure - lagged_exposure
        spread_factor = spread_cost(exposure, lagged_exposure, position)
        cost = self.spread * spread_factor * self.lot_size

        if self.reduce:
            cost = cost.mean()

        return cost


class Wins(nn.Module):
    def __init__(self, spread=0.0001, lot_size=100000, min_lot=0.0, dim=2):
        super().__init__()
        self.dim = dim
        self.profit = Profit(
            spread=spread, lot_size=lot_size, min_lot=min_lot, reduce=False, is_loss=False, dim=dim
        )

    def forward(self, y_pred, change):
        profit = self.profit(y_pred, change).mean(dim=self.dim)
        return (profit >= 0).float().mean()


@lru_cache(8)
def position_profit(position, change, min_lot=0.0, lot_size=100000, spread=0.0001, dim=2):
    exposure = position.cumsum(dim=dim)
    lagged_exposure = exposure - position

    # Round exposure to nearest allowable lot unit
    exposure, lagged_exposure = round_exposure(exposure, lagged_exposure, min_lot)
    position = exposure - lagged_exposure
    spread_factor = spread_cost(exposure, lagged_exposure, position)
    profit = (change * exposure - spread * spread_factor) * lot_size
    return profit


@lru_cache(8)
def sharpe_ratio(
    position, change, min_lot=0.0, lot_size=100000, spread=0.0001, return_profit=False, dim=2
):
    profit = position_profit(position, change, min_lot=min_lot, lot_size=lot_size, spread=spread)
    sharpe_r = profit / profit.std(dim=dim, keepdims=True)
    if return_profit:
        return sharpe_r, profit
    return sharpe_r


def round_inc(x, inc):
    return x.div(inc).round() * inc


def round_exposure(exposure, lagged_exposure, inc):
    if inc > 1e-6:
        exposure = round_inc(exposure, inc)
        lagged_exposure = round_inc(lagged_exposure, inc)
    return exposure, lagged_exposure


def compare_direction(a, b):
    return (a * b).sign().clamp(min=0)


@lru_cache(8)
def spread_cost(exposure, lagged_exposure, position):
    # Determine if exposure has increased (used to charge spread)
    exposure_incr = compare_direction(position, exposure) * position.abs()
    # Determine if net exposure position has changed (used to charge spread)
    zero_cross = compare_direction(-exposure, lagged_exposure) * exposure.abs()
    # Combine spread adjustments
    spread_factor = torch.max(exposure_incr, zero_cross)

    return spread_factor


def hinge(x, penalty):
    penalty = (1 + penalty) ** 0.5
    return x.clamp(max=0) * penalty + x.clamp(min=0) / penalty
