import os

import pandas as pd
import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative as default_colours


__all__ = ['TradingDataset']


class TradingDataset:

    def __init__(
        self, root, instruments, xcols, ycols, window, horizon,
        skip=1440*5, transform=None, label_transform=None, norm_x=True, norm_y=False
    ):
        self.all_cols = np.unique(xcols + ycols)
        self.window = window
        self.horizon = horizon
        self.transform = transform
        self.label_transform = label_transform
        self.norm_x = norm_x
        self.norm_y = norm_y
        
        # Get unique names for all instrument-columns
        self.xcols, self.ycols = [], []
        for inst in instruments:
            self.xcols.extend([f'{c}__{inst}' for c in xcols])
            self.ycols.extend([f'{c}__{inst}' for c in ycols])

        # Load data, skip (possibly null) starting samples, and ensure unique col names
        data = []
        for inst in instruments:
            data_inst = pd.read_parquet(
                os.path.join(root, f'{inst}.parquet'),
                columns=self.all_cols
            )
            data_inst = data_inst.sort_index().iloc[skip:]
            new_names = {c: f'{c}__{inst}' for c in data_inst.columns}
            data_inst = data_inst.rename(columns=new_names)
            data.append(data_inst)
        
        # Combine datasets (inner join on sample_time)
        self.data = data[0]
        for data_inst in data[1:]:
            self.data = self.data.join(data_inst, how='inner')
        
        # Get statistics for normalisation
        self.means, self.stds, self.covs = self.get_stats()

    def __len__(self):
        return len(self.data) - self.window - self.horizon
    
    def __getitem__(self, i):
        x, y = self.get_raw(i)

        if self.norm_x:
            x = x - x.mean()
            x = x / self.stds[self.xcols]
        if self.norm_y:
            y = y - y.iloc[0]
            y = y / self.stds[self.ycols]

        if self.transform is not None:
            x = self.transform(x)
        if self.label_transform is not None:
            y = self.label_transform(y)

        return x, y
    
    def get_raw(self, i):
        win_start, win_end = i, i + self.window
        hor_start, hor_end = win_end, win_end + self.horizon
        x = self.data[self.xcols].iloc[win_start:win_end]
        y = self.data[self.ycols].iloc[hor_start:hor_end]

        return x.copy(), y.copy()
    
    def get_stats(self):
        means = self.data[:-self.window-self.horizon].rolling(self.window).mean().mean()
        stds = self.data[:-self.window-self.horizon].rolling(self.window).std().mean()
        covs = means.abs() / stds
        return means, stds, covs
    
    def show_sample(self, i=None):
        if not i:
            i = np.random.choice(len(self))

        x, y = self[i]

        if len(y) == 1:
            ylabels = y
            _, y = self.get_raw(i)
            if self.norm_x:
                y = y - y.iloc[0]
                y = y / self.stds[self.ycols]
        else:
            ylabels = None

        
        fig = make_subplots(
            1, 2,
            column_widths=[len(x), len(y)],
            subplot_titles=['X', 'Y'],
            shared_yaxes=True,
            horizontal_spacing=0.02
        )
        colours = {c: colour for c, colour in zip(x.columns, default_colours.Plotly)}
        for c in x.columns:
            fig.add_trace(
                go.Scatter(
                    x=x.index, y=x[c], name=c, line=dict(color=colours[c]),
                    legendgroup=c, mode='lines'
                ),
                row=1, col=1
            )
        for c in y.columns:
            yc, colour, showlegend = y[c], None, None
            if c in x.columns:
                yc = yc - yc[0] + x[c][-1]
                colour = colours[c]
                showlegend = False
            
            fig.add_trace(
                go.Scatter(
                    x=y.index, y=yc, name=c,
                    line=dict(color=colour),
                    showlegend=showlegend, legendgroup=c,
                    mode='lines'
                ),
                row=1, col=2
            )

            if ylabels is not None:
                fig.add_annotation(
                    x=y.index[-1], y=yc[-1],
                    yshift=20,
                    text=str(ylabels[c]),
                    showarrow=False,
                    font=dict(size=18, color=colour),
                    row=1, col=2
                )
        
        fig.show()
        


