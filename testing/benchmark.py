import time

from testing.conflict_checker import (
    find_conflicts,
    count_conflict_rows
)


def count_intervals(discretized_df):
    """
    Liczy liczbę różnych przedziałów
    """

    condition_cols = discretized_df.columns[:-1]

    unique_intervals = set()

    for col in condition_cols:

        unique_intervals.update(
            discretized_df[col].unique()
        )

    return len(unique_intervals)


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