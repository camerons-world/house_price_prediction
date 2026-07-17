import h3
import pandas as pd

from src.spatial_diagnostics import BLOCK_RESOLUTION, spatial_block_groups, spatial_group_kfold


def test_spatial_block_groups_are_coarser_parent_hexes():
    hex_ids = pd.Series(["881f1d4887fffff", "881f1d4883fffff"])
    groups = spatial_block_groups(hex_ids)
    assert (groups == hex_ids.apply(lambda h: h3.cell_to_parent(h, BLOCK_RESOLUTION))).all()


def test_group_kfold_never_splits_a_block_across_train_and_test():
    # points far enough apart to land in different res-6 blocks, each repeated across
    # many rows -- a leaky split would put some of a block's rows in train, others in test
    points = [(51.5, -0.12), (51.7, 0.05), (51.3, -0.35), (51.6, -0.5)]
    fine_hexes = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in points]
    hex_id = (fine_hexes * 10)[:40]
    df = pd.DataFrame({"hex_id": hex_id})
    groups = spatial_block_groups(df["hex_id"])
    assert groups.nunique() > 1

    for train_idx, test_idx in spatial_group_kfold(df, n_splits=2):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)
