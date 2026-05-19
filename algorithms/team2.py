import numpy as np
import pandas as pd

#Filipek Krystian - 5

# ==============================================================================
# PROJEKT: ZACHŁANNA DYSKRETYZACJA ZSTĘPUJĄCA (GLOBALNA)
# ZESPÓŁ: Konrad Fugiel, Filip Lisowski, Oliwier Hędrzak, Krystian Filipek
# ==============================================================================

def discretize(df):
    """
    Główna funkcja dyskretyzująca. Przyjmuje DataFrame, zwraca zdyskretyzowany DataFrame.
    Przedziały zapisywane są w krotkach (tuples).
    """
    # ==================================================================
    # SEKCJA 1: PRZYGOTOWANIE DANYCH WEJŚCIOWYCH
    # AUTOR: Konrad Fugiel (Data Engineer)
    # ==================================================================
    # Zgodnie z wymaganiami, ostatnia kolumna to zawsze decyzja
    kolumny_warunkowe = df.columns[:-1]
    kolumna_decyzyjna = df.columns[-1]

    X_arr = df[kolumny_warunkowe].values
    y_arr = df[kolumna_decyzyjna].values
    n_obszarow, n_cech = X_arr.shape

    # Szukamy par obiektów o różnych klasach decyzyjnych do odseparowania
    pary_do_oddzielenia = [(i, j) for i in range(n_obszarow) for j in range(i + 1, n_obszarow) if y_arr[i] != y_arr[j]]
    pary_odseparowane = set()

    # Generujemy wszystkie potencjalne cięcia (środki między unikalnymi wartościami) dla wszystkich kolumn
    wszystkie_potencjalne_ciecia = []
    for col in range(n_cech):
        unikalne = np.sort(np.unique(X_arr[:, col]))
        if len(unikalne) > 1:
            ciecia = (unikalne[:-1] + unikalne[1:]) / 2.0
            for c in ciecia:
                wszystkie_potencjalne_ciecia.append((col, c))

    selectedCuts = []
    restOfCuts = list(wszystkie_potencjalne_ciecia)

    # ==================================================================
    # SEKCJA 2: ALGORYTM ZACHŁANNY GLOBALNY (SZUKANIE NAJLEPSZYCH CIĘĆ)
    # AUTOR: Filip Lisowski (Core ML Engineer)
    # ==================================================================
    # Pętla działa dopóki nie odseparujemy wszystkich możliwych par
    while len(pary_odseparowane) < len(pary_do_oddzielenia) and len(restOfCuts) > 0:
        najlepsze_ciecie = None
        najwiecej_nowych_par = -1
        pary_nowego_ciecia = set()

        # Szukamy cięcia, które oddzieli globalnie najwięcej par o różnych decyzjach
        for (col, ciecie) in restOfCuts:
            nowo_odseparowane = set()
            for (i, j) in pary_do_oddzielenia:
                if (i, j) in pary_odseparowane:
                    continue
                # Sprawdzenie czy cięcie rozdziela obiekty na danej cesze
                if (X_arr[i, col] <= ciecie and X_arr[j, col] > ciecie) or (
                        X_arr[j, col] <= ciecie and X_arr[i, col] > ciecie):
                    nowo_odseparowane.add((i, j))

            if len(nowo_odseparowane) > najwiecej_nowych_par:
                najwiecej_nowych_par = len(nowo_odseparowane)
                najlepsze_ciecie = (col, ciecie)
                pary_nowego_ciecia = nowo_odseparowane

        if najlepsze_ciecie is not None and najwiecej_nowych_par > 0:
            selectedCuts.append(najlepsze_ciecie)
            restOfCuts.remove(najlepsze_ciecie)
            pary_odseparowane.update(pary_nowego_ciecia)
        else:
            break

    # ==================================================================
    # SEKCJA 3: POST-PROCESSING (USUWANIE ZBĘDNYCH CIĘĆ I TUPLE)
    # AUTOR: Oliwier Hędrzak (Optimization Engineer)
    # ==================================================================
    ostateczne_ciecia = []
    max_separacja = len(pary_odseparowane)

    # Opcja 2d: Weryfikujemy każde cięcie z osobna, czy jest "nadmiarowe"
    for idx in range(len(selectedCuts)):
        testowane = selectedCuts[idx]
        pozostale = ostateczne_ciecia + selectedCuts[idx + 1:]

        odseparowane_bez = set()
        for (col, c) in pozostale:
            for (u, v) in pary_do_oddzielenia:
                if (X_arr[u, col] <= c and X_arr[v, col] > c) or (X_arr[v, col] <= c and X_arr[u, col] > c):
                    odseparowane_bez.add((u, v))

        if len(odseparowane_bez) < max_separacja:
            ostateczne_ciecia.append(testowane)

    # Budowanie docelowego systemu decyzyjnego w oparciu o Tuple (Krotki)
    df_wynikowy = pd.DataFrame(index=df.index)

    # Funkcja pomocnicza zamieniająca wartość liczbową na krotkę przedziału [a, b)
    def mapuj_na_tuple(wartosc, ciecia_kolumny):
        if not ciecia_kolumny:
            return (float('-inf'), float('inf'))
        if wartosc < ciecia_kolumny[0]:
            return (float('-inf'), float(ciecia_kolumny[0]))
        for idx_c in range(len(ciecia_kolumny) - 1):
            if ciecia_kolumny[idx_c] <= wartosc < ciecia_kolumny[idx_c + 1]:
                return (float(ciecia_kolumny[idx_c]), float(ciecia_kolumny[idx_c + 1]))
        return (float(ciecia_kolumny[-1]), float('inf'))

    ciecia_zgrupowane = {col: [] for col in range(n_cech)}
    for (col, c) in ostateczne_ciecia:
        ciecia_zgrupowane[col].append(c)

    # Aplikowanie mapowania na tuplę bez zmieniania indeksu i rzędów
    for col_idx, nazwa in enumerate(kolumny_warunkowe):
        ciecia_tej_kolumny = sorted(ciecia_zgrupowane[col_idx])
        df_wynikowy[nazwa] = df[nazwa].apply(lambda x: mapuj_na_tuple(x, ciecia_tej_kolumny))

    df_wynikowy[kolumna_decyzyjna] = df[kolumna_decyzyjna]
    return df_wynikowy


# ==================================================================
# SEKCJA 4: RAPORTOWANIE KONFLIKTÓW I TESTOWANIE
# AUTOR: Krystian Filipek (QA & Integration Lead)
# ==================================================================
def raportuj_konflikty(df_wynikowy):
    """
    Znajduje i wypisuje niespójności deterministyczne (konflikty)
    według sztywnych zasad ze specyfikacji.
    """
    kolumny_warunkowe = list(df_wynikowy.columns[:-1])
    kolumna_decyzyjna = df_wynikowy.columns[-1]

    # Grupujemy po wszystkich zdyskretyzowanych cechach wiersze, które wyglądają identycznie
    grupy = df_wynikowy.groupby(kolumny_warunkowe)

    liczba_konfliktowych_wierszy = 0
    print("\nWynik (Raport Konfliktów):")

    for _, grupa in grupy:
        # Jeśli w grupie identycznych obiektów jest więcej niż 1 klasa decyzyjna - mamy konflikt
        if grupa[kolumna_decyzyjna].nunique() > 1:
            indeksy_konfliktu = list(grupa.index)
            # Wypisujemy po przecinku, np: 1, 2, 3
            print(", ".join(map(str, indeksy_konfliktu)))
            liczba_konfliktowych_wierszy += len(indeksy_konfliktu)

    print(f"Liczba konfliktujących wierszy: {liczba_konfliktowych_wierszy}")