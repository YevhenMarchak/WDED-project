import pandas as pd
import numpy as np


# =====================================================================
# [Artur Jurkowski]
# ROLA: Walidacja, integralność danych i śledzenie konfliktów.
# =====================================================================
def policz_globalne_konflikty(X, y, aktywne_ciecia):
    sygnatury = np.zeros(X.shape, dtype=int)

    for j in range(X.shape[1]):
        if not aktywne_ciecia[j]:
            sygnatury[:, j] = 0
        else:
            sygnatury[:, j] = np.searchsorted(aktywne_ciecia[j], X[:, j])

    slownik_sygnatur = {}
    for i, rzad in enumerate(sygnatury):
        klucz = tuple(rzad)
        if klucz not in slownik_sygnatur:
            slownik_sygnatur[klucz] = []
        slownik_sygnatur[klucz].append(y[i])

    ilosc_konfliktow = 0
    for decyzje in slownik_sygnatur.values():
        if len(set(decyzje)) > 1:
            ilosc_konfliktow += len(decyzje)
    return ilosc_konfliktow


# =====================================================================
# [Kamil Krawczyk]
# ROLA: Metryki zanieczyszczenia i preprocessing danych.
# =====================================================================
def szybki_gini(podzbior_y):
    if len(podzbior_y) == 0:
        return 0.0
    _, zliczenia = np.unique(podzbior_y, return_counts=True)
    return 1.0 - np.sum((zliczenia / len(podzbior_y)) ** 2)


def discretize(df):
    df_wyjscie = df.copy()
    kolumna_decyzyjna = df.columns[-1]
    kolumny_warunkowe = df.columns[:-1]

    y = df[kolumna_decyzyjna].values
    X = df[kolumny_warunkowe].values

    # -----------------------------------------------------------------
    # [Kamil]: Wyciąganie unikalnych wartości i generowanie puli cięć
    # -----------------------------------------------------------------
    aktywne_ciecia = []
    for j in range(X.shape[1]):
        wartosci_kolumny = pd.to_numeric(df[kolumny_warunkowe[j]], errors='coerce').values
        X[:, j] = wartosci_kolumny
        unikalne_wartosci = np.sort(np.unique(wartosci_kolumny[~np.isnan(wartosci_kolumny)]))

        if len(unikalne_wartosci) <= 1:
            aktywne_ciecia.append([])
        else:
            ciecia = ((unikalne_wartosci[:-1] + unikalne_wartosci[1:]) / 2.0).tolist()
            aktywne_ciecia.append(ciecia)

    # [Kamil]: Ustalenie sztywnej bazy konfliktów, których nie da się pominąć
    bazowe_konflikty = policz_globalne_konflikty(X, y, aktywne_ciecia)

    # =====================================================================
    # [Patryk Długosz]
    # ROLA: Architektura głównej pętli algorytmu Bottom-Up i formatowanie wyjścia.
    # =====================================================================

    # [Patryk]: Globalna ocena użyteczności wszystkich cięć z użyciem metryki od Kamila
    globalna_pula = []
    for j in range(X.shape[1]):
        for idx, ciecie in enumerate(aktywne_ciecia[j]):
            lewa_granica = aktywne_ciecia[j][idx - 1] if idx > 0 else -np.inf
            prawa_granica = aktywne_ciecia[j][idx + 1] if idx < len(aktywne_ciecia[j]) - 1 else np.inf
            maska = (X[:, j] >= lewa_granica) & (X[:, j] < prawa_granica)

            wynik = szybki_gini(y[maska])
            globalna_pula.append((wynik, j, ciecie))

    globalna_pula.sort(key=lambda element: element[0])

    # [Patryk]: Zachłanne, globalne usuwanie najgorszych cięć
    for wynik, j, ciecie in globalna_pula:
        if ciecie in aktywne_ciecia[j]:
            aktywne_ciecia[j].remove(ciecie)

            # [Artur]: Walidacja bezpieczeństwa bazy po usunięciu cięcia
            aktualne_konflikty = policz_globalne_konflikty(X, y, aktywne_ciecia)

            if aktualne_konflikty > bazowe_konflikty:
                aktywne_ciecia[j].append(ciecie)
                aktywne_ciecia[j].sort()

    # -----------------------------------------------------------------
    # [Patryk]: Konwersja na tuple, dodanie nieskończoności na brzegach
    #            i mapowanie do DataFrame (zgodność ze specyfikacją).
    # -----------------------------------------------------------------
    for j, kolumna in enumerate(kolumny_warunkowe):
        ciecia = aktywne_ciecia[j]
        granice = [-np.inf] + ciecia + [np.inf]
        przedzialy = [(granice[i], granice[i + 1]) for i in range(len(granice) - 1)]

        def mapuj_na_przedzial(wartosc, p=przedzialy):
            for przedzial in p:
                if przedzial[0] <= wartosc < przedzial[1]:
                    return przedzial
            return p[-1]

        df_wyjscie[kolumna] = df_wyjscie[kolumna].apply(mapuj_na_przedzial)

    return df_wyjscie