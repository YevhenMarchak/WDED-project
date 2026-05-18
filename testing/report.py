from pathlib import Path


def generate_report(team_name, results):
    """
    Generuje raport tekstowy
    """

    report = []

    report.append(
        f"=== RAPORT: {team_name} ===\n"
    )

    report.append(
        f"Czas wykonania: "
        f"{results['execution_time']:.6f} s"
    )

    report.append(
        f"Liczba przedziałów: "
        f"{results['interval_count']}"
    )

    report.append(
        f"Liczba konfliktów: "
        f"{results['conflict_count']}"
    )

    report.append("\n=== KONFLIKTY ===")

    conflicts = results["conflicts"]

    if not conflicts:

        report.append(
            "Brak konfliktów"
        )

    else:

        for conflict in conflicts:

            line = ", ".join(
                map(str, conflict)
            )

            report.append(line)

    return "\n".join(report)


def save_report(team_name, report_content):
    """
    Zapisuje raport do pliku
    """

    # tworzy folder results jeśli nie istnieje
    Path("results").mkdir(
        exist_ok=True
    )

    file_path = (
        f"results/{team_name}_report.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report_content)

    print(
        f"Raport zapisany: {file_path}"
    )