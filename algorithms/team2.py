import numpy as np
import pandas as pd

# ==============================================================================
# PROJEKT: ZACHŁANNA DYSKRETYZACJA ZSTĘPUJĄCA (GLOBALNA)
# ZESPÓŁ: Konrad Fugiel, Filip Lisowski, Oliwier Hędrzak, Krystian Filipek
# ==============================================================================

def discretize(df):
    # ==================================================================
    # SEKCJA 1: PRZYGOTOWANIE DANYCH WEJŚCIOWYCH
    # AUTOR: Konrad Fugiel
    # ==================================================================
    kolumny_warunkowe = df.columns[:-1]
    kolumna_decyzyjna = df.columns[-1]

    X_arr = df[kolumny_warunkowe].to_numpy(dtype=float)
    y_arr = df[kolumna_decyzyjna].values
    n_obszarow, n_cech = X_arr.shape

    # Szukamy par obiektów o różnych klasach decyzyjnych do odseparowania
    indeksy_i, indeksy_j = np.triu_indices(n_obszarow, k=1)
    maska_roznych_decyzji = y_arr[indeksy_i] != y_arr[indeksy_j]

    pary_i = indeksy_i[maska_roznych_decyzji].astype(np.int32)
    pary_j = indeksy_j[maska_roznych_decyzji].astype(np.int32)

    liczba_par_do_oddzielenia = len(pary_i)
    pary_odseparowane = np.zeros(liczba_par_do_oddzielenia, dtype=bool)

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
    # SEKCJA 2: SZUKANIE NAJLEPSZYCH CIĘĆ
    # AUTOR: Filip Lisowski
    # ==================================================================
    # Pętla działa, dopóki nie odseparujemy wszystkich możliwych par
    while np.count_nonzero(pary_odseparowane) < liczba_par_do_oddzielenia and len(restOfCuts) > 0:
        najlepsze_ciecie = None
        najlepszy_indeks = None
        najwiecej_nowych_par = -1

        aktywne_pary = ~pary_odseparowane
        aktywne_i = pary_i[aktywne_pary]
        aktywne_j = pary_j[aktywne_pary]

        if len(aktywne_i) == 0:
            break

        # Szukamy cięcia, które oddzieli globalnie najwięcej par o różnych decyzjach
        for col in range(n_cech):
            indeksy_ciec = [idx for idx, (kol, _) in enumerate(restOfCuts) if kol == col]

            if not indeksy_ciec:
                continue

            ciecia_kolumny = np.array([restOfCuts[idx][1] for idx in indeksy_ciec], dtype=float)

            wartosci_i = X_arr[aktywne_i, col]
            wartosci_j = X_arr[aktywne_j, col]

            dolne = np.minimum(wartosci_i, wartosci_j)
            gorne = np.maximum(wartosci_i, wartosci_j)

            dolne_posortowane = np.sort(dolne)
            gorne_posortowane = np.sort(gorne)

            # Liczba par rozdzielanych przez cięcie c:
            # para jest rozdzielona wtedy, gdy dolna wartość < c < górna wartość.
            liczby_nowych_par = (
                    np.searchsorted(dolne_posortowane, ciecia_kolumny, side="left") -
                    np.searchsorted(gorne_posortowane, ciecia_kolumny, side="right")
            )

            lokalny_indeks = int(np.argmax(liczby_nowych_par))
            lokalny_wynik = int(liczby_nowych_par[lokalny_indeks])

            if lokalny_wynik > najwiecej_nowych_par:
                najwiecej_nowych_par = lokalny_wynik
                najlepszy_indeks = indeksy_ciec[lokalny_indeks]
                najlepsze_ciecie = restOfCuts[najlepszy_indeks]

        if najlepsze_ciecie is not None and najwiecej_nowych_par > 0:
            selectedCuts.append(najlepsze_ciecie)

            col, ciecie = najlepsze_ciecie

            wartosci_i = X_arr[pary_i, col]
            wartosci_j = X_arr[pary_j, col]

            dolne = np.minimum(wartosci_i, wartosci_j)
            gorne = np.maximum(wartosci_i, wartosci_j)

            pary_nowego_ciecia = (~pary_odseparowane) & (dolne < ciecie) & (ciecie < gorne)

            pary_odseparowane |= pary_nowego_ciecia
            restOfCuts.pop(najlepszy_indeks)
        else:
            break

    # ==================================================================
    # SEKCJA 3: POST-PROCESSING
    # AUTOR: Oliwier Hędrzak
    # ==================================================================
    ostateczne_ciecia = []
    ostateczne_maski = []

    # Zapamiętujemy docelową separację par po wszystkich cięciach wybranych zachłannie.
    # Dzięki temu usuwamy tylko takie cięcia, bez których nadal odseparowane są dokładnie te same pary.
    docelowa_separacja = pary_odseparowane.copy()

    # Przygotowujemy maski separacji dla każdego wybranego cięcia.
    # Każda maska mówi, które pary obiektów rozdziela dane cięcie.
    maski_selectedCuts = []

    for (col, c) in selectedCuts:
        wartosci_i = X_arr[pary_i, col]
        wartosci_j = X_arr[pary_j, col]

        dolne = np.minimum(wartosci_i, wartosci_j)
        gorne = np.maximum(wartosci_i, wartosci_j)

        maska_ciecia = (dolne < c) & (c < gorne)
        maski_selectedCuts.append(maska_ciecia)

    # Weryfikujemy każde cięcie z osobna, czy jest zbędne
    for idx in range(len(selectedCuts)):
        testowane = selectedCuts[idx]

        # Tak jak wcześniej: bierzemy już zaakceptowane cięcia oraz wszystkie późniejsze.
        pozostale_maski = ostateczne_maski + maski_selectedCuts[idx + 1:]

        if pozostale_maski:
            odseparowane_bez = np.logical_or.reduce(pozostale_maski)
        else:
            odseparowane_bez = np.zeros(liczba_par_do_oddzielenia, dtype=bool)

        # Cięcie jest potrzebne, jeśli bez niego nie uzyskujemy dokładnie tej samej separacji par.
        if not np.array_equal(odseparowane_bez, docelowa_separacja):
            ostateczne_ciecia.append(testowane)
            ostateczne_maski.append(maski_selectedCuts[idx])

    # Budowanie systemu decyzyjnego w oparciu o tuple
    df_wynikowy = pd.DataFrame(index=df.index)

    # Funkcja pomocnicza zamieniająca wartość liczbową na tuple przedziału [a, b)
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

    # Aplikowanie mapowania na tuple bez zmieniania indeksu i rzędów
    for col_idx, nazwa in enumerate(kolumny_warunkowe):
        ciecia_tej_kolumny = sorted(ciecia_zgrupowane[col_idx])
        df_wynikowy[nazwa] = df[nazwa].apply(lambda x: mapuj_na_tuple(x, ciecia_tej_kolumny))

    df_wynikowy[kolumna_decyzyjna] = df[kolumna_decyzyjna]
    return df_wynikowy

# ==================================================================
# SEKCJA 4: RAPORTOWANIE KONFLIKTÓW
# AUTOR: Krystian Filipek
# ==================================================================
def raportuj_konflikty(df_wynikowy):
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
            print(", ".join(map(str, indeksy_konfliktu)))
            liczba_konfliktowych_wierszy += len(indeksy_konfliktu)

    print(f"Liczba konfliktujących wierszy: {liczba_konfliktowych_wierszy}")