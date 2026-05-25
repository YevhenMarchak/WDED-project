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
df = pd.read_csv("data/data0.csv")

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