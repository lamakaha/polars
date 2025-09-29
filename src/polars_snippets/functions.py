import polars as pl

def create_hash_rollup_expr(
    expr: pl.Expr,
    delimiter: str = "#"
) -> pl.Expr:
    """
    Creates a Polars expression to roll up and hash unique values from a column.

    This function handles mixed data types, nulls, NaNs, and values that already
    contain the delimiter. It is designed to work in both group_by and window
    contexts.

    The process involves:
    1. Casting all values to a string representation.
    2. Formatting numeric strings to remove trailing '.0'.
    3. Splitting values by the delimiter to handle pre-existing delimited strings.
    4. Filtering out null, empty, and 'NaN' strings.
    5. Exploding the lists of strings into individual rows.
    6. Aggregating the unique, sorted strings back into a single delimited string.
    7. Hashing the final result.

    Args:
        expr: A Polars expression for the column to be rolled up.
        delimiter: The delimiter to use for joining values.

    Returns:
        A Polars expression that computes the hash rollup.
    """
    # Cast to string to handle mixed types
    col_str = expr.cast(pl.String, strict=False)

    # Clean up numeric representations (e.g., '3.0' -> '3')
    formatted_col = (
        pl.when(expr.cast(pl.Float64, strict=False).is_not_null())
        .then(
            expr.cast(pl.Float64, strict=False)
            .cast(pl.String)
            .str.replace(r"\.0+$", "")
        )
        .otherwise(col_str)
    )

    # Split by delimiter, filter out unwanted values, and explode
    exploded_list = (
        formatted_col.str.split(delimiter)
        .list.eval(
            pl.element().filter(
                pl.element().is_not_null()
                & (pl.element() != "")
                & (pl.element().str.to_lowercase() != "nan")
            )
        )
        .list.explode()
    )

    # Aggregate, concatenate, and hash
    return (
        exploded_list.unique()
        .sort()
        .str.join(delimiter=delimiter)
        .hash(seed=0)
    )