from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import W
from sklearn.model_selection import GroupKFold

# block cv groups on a coarser hex than the feature hex (res 8 -> parent res 6) so a
# held-out block is a contiguous region, not just individual hexes with leaky neighbours
# still sitting in the training fold
BLOCK_RESOLUTION = 6


def spatial_block_groups(hex_ids: pd.Series) -> pd.Series:
    return hex_ids.apply(lambda h: h3.cell_to_parent(h, BLOCK_RESOLUTION))


def spatial_group_kfold(df: pd.DataFrame, n_splits: int = 5):
    groups = spatial_block_groups(df["hex_id"])
    return GroupKFold(n_splits=n_splits).split(df, groups=groups)


def hex_adjacency_weights(hex_ids: list[str]) -> W:
    present = set(hex_ids)
    neighbors = {h: [n for n in h3.grid_disk(h, 1) if n != h and n in present] for h in present}
    return W(neighbors)


def morans_i_on_residuals(df: pd.DataFrame, residual_col: str, hex_col: str = "hex_id") -> Moran:
    hex_resid = df.groupby(hex_col)[residual_col].mean()
    w = hex_adjacency_weights(hex_resid.index.tolist())
    y = hex_resid.reindex(w.id_order).values
    return Moran(y, w)
