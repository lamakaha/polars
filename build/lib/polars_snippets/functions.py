import polars as pl

def hash_rollup(expr: pl.Expr) -> pl.Expr:
    """
    Rolls up a column by hashing its unique values and concatenating them.

    This function takes a Polars expression, finds the unique values,
    sorts them, concatenates them with a '#' separator, and then
    computes the hash of the resulting string.

    Args:
        expr: A Polars expression representing the column to roll up.

    Returns:
        A Polars expression that computes the hash rollup.
    """
    return (
        expr.unique()
        .sort()
        .cast(pl.Utf8)
        .implode()
        .list.join("#")
        .hash()
    )