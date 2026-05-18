def belongs(value, interval):
    """
    Sprawdza czy value należy do przedziału [a, b)
    """
    left, right = interval
    return left <= value < right


def is_valid_interval(interval):
    """
    Sprawdza poprawność przedziału
    """

    # musi być tuple
    if not isinstance(interval, tuple):
        return False

    # długość 2
    if len(interval) != 2:
        return False

    left, right = interval

    # liczby
    if not isinstance(left, (int, float)):
        return False

    if not isinstance(right, (int, float)):
        return False

    # poprawna kolejność
    if left >= right:
        return False

    return True


def count_unique_intervals(df):
    """
    Liczy liczbę różnych przedziałów
    """

    unique_intervals = set()

    for col in df.columns[:-1]:  # bez decyzji
        unique_intervals.update(df[col].unique())

    return len(unique_intervals)

