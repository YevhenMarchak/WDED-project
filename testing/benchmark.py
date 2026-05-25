import time

from testing.conflict_checker import (
    find_conflicts,
    count_conflict_rows
)


import math


def normalize_interval(interval):

    left, right = interval

    # normalizacja inf
    if math.isinf(left):
        left = float("-inf")

    else:
        left = round(float(left), 10)

    if math.isinf(right):
        right = float("inf")

    else:
        right = round(float(right), 10)

    return (left, right)


def count_intervals(discretized_df):
    """
    Liczy liczbę różnych przedziałów
    osobno dla każdej kolumny
    """

    condition_cols = discretized_df.columns[:-1]

    total = 0

    for col in condition_cols:

        unique_intervals = set()

        for interval in discretized_df[col]:

            normalized = normalize_interval(
                interval
            )

            unique_intervals.add(normalized)

        total += len(unique_intervals)

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