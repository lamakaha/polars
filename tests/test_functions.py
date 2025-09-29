import polars as pl
from polars.testing import assert_frame_equal
from polars_snippets.functions import create_hash_rollup_expr
import numpy as np

def test_create_hash_rollup_expr_comprehensive():
    """
    Tests the create_hash_rollup_expr function in both group_by and window
    contexts with a complex DataFrame.
    """
    df = pl.DataFrame({
        'group':     ['A', 'A', 'A', 'B', 'B', 'C', 'A', 'B', 'A', 'D', 'D'],
        'mixed_col': ['x', '3.0', 'y', 'z#A', '3.20', None, 'x', 'w', 'B#C', '5.00', str(np.nan)],
    })

    # Expected rolled-up strings for each group before hashing
    # Group A: unique sorted values are ['3', 'B', 'C', 'x', 'y'] -> "3#B#C#x#y"
    # Group B: unique sorted values are ['3.2', 'A', 'w', 'z'] -> "3.2#A#w#z"
    # Group C: only a null value, so the result is an empty string -> ""
    # Group D: unique sorted values are ['5'] -> "5"
    expected_strings = {
        "A": "3#B#C#x#y",
        "B": "3.2#A#w#z",
        "C": "",
        "D": "5"
    }

    # Calculate the expected hashes
    expected_hashes_map = {
        group: pl.Series([s]).hash(seed=0)[0] for group, s in expected_strings.items()
    }

    # --- Test GroupBy Aggregation ---
    rollup_expr = create_hash_rollup_expr(pl.col('mixed_col'), delimiter='#')

    result_groupby = df.group_by('group').agg(
        rollup_expr.alias('rolled_up_hash')
    ).sort("group")

    expected_groupby = pl.DataFrame({
        "group": list(expected_hashes_map.keys()),
        "rolled_up_hash": list(expected_hashes_map.values())
    }).with_columns(pl.col("rolled_up_hash").cast(pl.UInt64)).sort("group")

    assert_frame_equal(result_groupby, expected_groupby)

    # --- Test Window/Over Operation ---
    result_window = df.with_columns(
        rollup_expr.over('group').alias('window_rolled_up_hash')
    )

    expected_window = df.with_columns(
        pl.col('group').replace(expected_hashes_map).cast(pl.UInt64).alias('window_rolled_up_hash')
    )

    # Sort both frames to ensure the comparison is order-independent
    assert_frame_equal(
        result_window.sort('group', 'mixed_col'),
        expected_window.sort('group', 'mixed_col')
    )