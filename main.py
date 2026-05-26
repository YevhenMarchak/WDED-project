import pandas as pd

# ALGORYTMY
from algorithms.team1 import discretize as team1_discretize
from algorithms.team2 import discretize as team2_discretize
from algorithms.team3 import discretize as team3_discretize
from algorithms.team4 import discretize as team4_discretize

# TESTING
from testing.benchmark import evaluate_algorithm
from testing.validator import validate_all
from testing.report import (
    generate_report,
    save_report
)
from testing.summary import (
    rank_algorithms,
    generate_summary_report,
    save_summary_report
)

# DANE TESTOWE
def load_dataset(path):
    """
    Wczytuje CSV i wymusza sensowny układ:
      - pierwszy wiersz traktowany jako nagłówek tylko jeśli faktycznie
        jest tekstowy (a nie wartościami liczbowymi)
      - wszystkie kolumny cech (poza ostatnią - decyzyjną) konwertowane na float
      - ostatnia kolumna zostaje jak jest (decyzja moze byc int lub str)
    """
    # Sprawdzamy pierwszy wiersz: jesli da sie zamienic kazda wartosc na float,
    # to znaczy ze nie ma naglowka i trzeba wczytac z header=None.
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    first_cells = [c.strip() for c in first_line.split(",")]

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    has_header = not all(is_number(c) for c in first_cells)

    if has_header:
        df = pd.read_csv(path)
    else:
        # Brak naglowka -> generujemy nazwy a1, a2, ..., dec
        n_cols = len(first_cells)
        names = [f"a{i+1}" for i in range(n_cols - 1)] + ["dec"]
        df = pd.read_csv(path, header=None, names=names)

    # Wymuszamy typ numeryczny na cechach (wszystko poza ostatnia kolumna).
    # Jezeli ktoras kolumna mialaby smieci, wyrzucimy blad z jasnym komunikatem.
    feature_cols = df.columns[:-1]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


df = load_dataset("data/data2"
                  ".csv")

print(f"Wczytano dataset: {df.shape[0]} wierszy, "
      f"{df.shape[1]} kolumn (kolumny: {list(df.columns)})")

# LISTA ALGORYTMÓW
algorithms = {
    "team1": team1_discretize,
    "team2": team2_discretize,
    "team3": team3_discretize,
    "team4": team4_discretize
}

# WYNIKI WSZYSTKICH ALGORYTMÓW
all_results = []

# TESTOWANIE ALGORYTMÓW
for team_name, algorithm in algorithms.items():

    print()
    print("=" * 50)
    print(f"TESTOWANIE: {team_name}")
    print("=" * 50)

    try:
        # EVALUATION
        results = evaluate_algorithm(
            algorithm,
            df
        )

        discretized_df = results[
            "discretized_df"
        ]

        # WALIDACJA
        valid, message = validate_all(
            df,
            discretized_df
        )

        print()
        print("WALIDACJA:")
        print(valid)
        print(message)

        # DODANIE TEAM NAME DO RANKINGU
        results["team_name"] = team_name

        results["validation"] = valid

        all_results.append(results)

        # RAPORT INDYWIDUALNY
        report = generate_report(
            team_name,
            results
        )

        save_report(
            team_name,
            report
        )

        print()
        print(report)

    except Exception as e:

        print()
        print(
            f"BŁĄD ALGORYTMU {team_name}:"
        )

        print(e)

# RANKING KOŃCOWY
ranking = rank_algorithms(
    all_results
)

summary_report = (
    generate_summary_report(
        ranking
    )
)

save_summary_report(
    summary_report
)

print()
print("=" * 50)
print("FINALNY RANKING")
print("=" * 50)

print()
print(summary_report)