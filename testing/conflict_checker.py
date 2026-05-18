def find_conflicts(discretized_df):
    """
    Znajduje konflikty w zdyskretyzowanym DataFrame
    """

    decision_col = discretized_df.columns[-1]
    condition_cols = discretized_df.columns[:-1]

    groups = {}

    for idx, row in discretized_df.iterrows():

        # opis obiektu
        key = tuple(row[col] for col in condition_cols)

        if key not in groups:
            groups[key] = []

        groups[key].append(
            (idx, row[decision_col])
        )

    conflicts = []

    for key, values in groups.items():

        decisions = set(
            decision
            for _, decision in values
        )

        # konflikt = różne decyzje
        if len(decisions) > 1:

            conflict_indexes = [
                idx
                for idx, _ in values
            ]

            conflicts.append(conflict_indexes)

    return conflicts

def count_conflict_rows(conflicts):
    """
    Liczy liczbę konfliktujących rekordów
    """

    return sum(len(conflict) for conflict in conflicts)

def print_conflicts(conflicts):
    """
    Wypisuje konflikty
    """

    if not conflicts:
        print("Brak konfliktów")
        return

    print("Konflikty:")

    for conflict in conflicts:
        print(", ".join(map(str, conflict)))

    print()
    print(
        "Liczba konfliktujących wierszy:",
        count_conflict_rows(conflicts)
    )