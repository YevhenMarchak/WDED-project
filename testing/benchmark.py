import time

from testing.interval_utils import count_unique_intervals

from testing.conflict_checker import (
    find_conflicts,
    count_conflict_rows
)


def evaluate_algorithm(algorithm, df):
    """
    Pełna ocena algorytmu dyskretyzującego:
    - pomiar czasu,
    - wypisanie przedziałów,
    - liczenie przedziałów,
    - wykrywanie konfliktów.
    """
    start = time.perf_counter()

    discretized_df = algorithm(df)

    for col in discretized_df.columns[:-1]:

        normalized = []

        for interval in discretized_df[col]:

            left, right = interval

            if left != float("-inf"):
                left = round(float(left), 3)

            if right != float("inf"):
                right = round(float(right), 3)

            normalized.append((left, right))

        unique_vals = sorted(
            list(set(normalized)),
            key=str
        )

        print(f"\nKOLUMNA: {col}")
        print(f"LICZBA PRZEDZIAŁÓW: {len(unique_vals)}")

        for v in unique_vals:
            left, right = v
            print(f"({left}, {right})")

    end = time.perf_counter()

    execution_time = end - start

    interval_count = count_unique_intervals(
        discretized_df
    )

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