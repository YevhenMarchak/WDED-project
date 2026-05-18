from pathlib import Path


def rank_algorithms(all_results):
    """
    Sortuje algorytmy wg jakości
    """

    ranking = sorted(
        all_results,

        key=lambda x: (
            x["conflict_count"],
            x["interval_count"],
            x["execution_time"]
        )
    )

    return ranking


def generate_summary_report(ranking):
    """
    Generuje raport rankingowy
    """

    lines = []

    lines.append(
        "=== FINALNY RANKING ALGORYTMÓW ===\n"
    )

    for position, result in enumerate(
        ranking,
        start=1
    ):

        lines.append(
            f"{position}. "
            f"{result['team_name']}"
        )

        lines.append(
            f"   Konflikty: "
            f"{result['conflict_count']}"
        )

        lines.append(
            f"   Przedziały: "
            f"{result['interval_count']}"
        )

        lines.append(
            f"   Czas: "
            f"{result['execution_time']:.6f} s"
        )

        lines.append("")

    return "\n".join(lines)


def save_summary_report(content):
    """
    Zapisuje ranking do pliku
    """

    Path("results").mkdir(
        exist_ok=True
    )

    file_path = (
        "results/final_ranking.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(content)

    print()
    print(
        f"Ranking zapisany: {file_path}"
    )