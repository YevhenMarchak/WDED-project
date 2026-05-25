import time

from testing.conflict_checker import (
    find_conflicts,
    count_conflict_rows
)


def normalize_interval(interval):
    """
    Zamienia różne reprezentacje
    przedziałów na jednolity tuple
    """

    # pandas.Interval
    if hasattr(interval, "left"):

        return (
            float(interval.left),
            float(interval.right)
        )

    # tuple/list
    if isinstance(interval, (tuple, list)):

        return (
            float(interval[0]),
            float(interval[1])
        )

    # fallback
    return interval


def count_intervals(discretized_df):
    """
    Liczy liczbę przedziałów
    osobno dla każdej kolumny
    """

    condition_cols = discretized_df.columns[:-1]

    total = 0

    for col in condition_cols:

        total += len(
            discretized_df[col].unique()
        )

    return total


def evaluate_algorithm(algorithm, df):
    """
    Pełna ocena algorytmu
    """

    # pomiar czasu
    start = time.perf_counter()

    discretized_df = algorithm(df)

    end = time.perf_counter()

    execution_time = end - start

    # liczba przedziałów
    interval_count = count_intervals(
        discretized_df
    )

    # konflikty
    conflicts = find_conflicts(
        discretized_df
    )

    conflict_count = count_conflict_rows(
        conflicts
    )

    return {
        "discretized_df": discretized_df,

        "execution_time": execution_time,

        "interval_count": interval_count,

        "conflict_count": conflict_count,

        "conflicts": conflicts
    }