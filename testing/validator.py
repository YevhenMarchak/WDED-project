import pandas as pd

from testing.interval_utils import (
    belongs,
    is_valid_interval
)


def validate_dataframe_structure(original_df, discretized_df):
    """
    Sprawdza strukturę DataFrame
    """

    # musi być DataFrame
    if not isinstance(discretized_df, pd.DataFrame):
        return False, "Output nie jest DataFrame"

    # liczba wierszy
    if len(original_df) != len(discretized_df):
        return False, "Zmieniono liczbę wierszy"

    # liczba kolumn
    if len(original_df.columns) != len(discretized_df.columns):
        return False, "Zmieniono liczbę kolumn"

    # indeksy
    if not original_df.index.equals(discretized_df.index):
        return False, "Zmieniono indeksy"

    # kolejność kolumn
    if list(original_df.columns) != list(discretized_df.columns):
        return False, "Zmieniono kolejność kolumn"

    return True, "OK"


def validate_intervals(original_df, discretized_df):
    """
    Sprawdza czy wszystkie przedziały są poprawne
    """

    condition_cols = discretized_df.columns[:-1]

    for col in condition_cols:

        for interval in discretized_df[col]:

            if not is_valid_interval(interval):
                return False, f"Niepoprawny przedział: {interval}"

    return True, "OK"


def validate_membership(original_df, discretized_df):
    """
    Sprawdza czy każda wartość należy do swojego przedziału
    """

    condition_cols = original_df.columns[:-1]

    for col in condition_cols:

        for idx in original_df.index:

            original_value = original_df.loc[idx, col]
            interval = discretized_df.loc[idx, col]

            if not belongs(original_value, interval):
                return False, (
                    f"Wartość {original_value} "
                    f"nie należy do przedziału {interval}"
                )

    return True, "OK"


def validate_all(original_df, discretized_df):
    """
    Główna funkcja walidująca
    """

    validators = [
        validate_dataframe_structure,
        validate_intervals,
        validate_membership
    ]

    for validator in validators:

        valid, message = validator(original_df, discretized_df)

        if not valid:
            return False, message

    return True, "Walidacja zakończona sukcesem"