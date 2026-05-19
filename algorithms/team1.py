import numpy as np
import pandas as pd

#Buczulinski - 1

def discretize(df):
    """
    Zachłanny algorytm dyskretyzacji danych.
    Zwraca nowy DataFrame, gdzie wartości liczbowe zostały zamienione na przedziały (tuple).
    Zakłada:
      - Index DataFrame: Odpowiada za ID/Indeksy
      - Kolumny: Cechy do dyskretyzacji
      - Ostatnia kolumna: Klasa decyzyjna
    """
    try:
        # Zabezpieczenie typu wejściowego i utworzenie kopii
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        wynik_df = df.copy()

        # Skoro użyto index_col=0, indeksy są natywnie obsługiwane przez Pandas.
        # Wszystkie kolumny jako cechy (:-1) i ostatnią jako klasę (-1).
        surowe_klasy = df.iloc[:, -1].values
        macierz_cech = df.iloc[:, :-1].values

        liczba_obiektow, liczba_cech = macierz_cech.shape

        if liczba_cech == 0:
            return wynik_df

        # Pozostawienie oryginalnych wartości klas (bez konwersji na liczby)
        klasy = surowe_klasy

        # Znalezienie par wrogów (obiekty o różnych klasach)
        pary_konfliktowe = []
        for i in range(liczba_obiektow):
            for k in range(i + 1, liczba_obiektow):
                if klasy[i] != klasy[k]:
                    pary_konfliktowe.append((i, k))

        # Utworzenie zbioru wszystkich możliwych cięć
        aktywne_ciecia = []  # Format: (indeks_cechy, wartość_cięcia)

        for f in range(liczba_cech):
            kolumna_x = macierz_cech[:, f]
            # Złączenie z klasami i sortowanie
            dane = sorted(zip(kolumna_x, klasy), key=lambda item: item[0])
            posortowane_x = [item[0] for item in dane]
            posortowane_y = [item[1] for item in dane]

            # Sprawdzanie, czy kolejne wiersze o tej samej wartości cechy należą do różnych klas
            unikalne_x = []
            klasy_dla_x = []
            aktualne_x = posortowane_x[0]
            aktualne_klasy = set([posortowane_y[0]])

            for i in range(1, len(posortowane_x)):
                if posortowane_x[i] == aktualne_x:
                    aktualne_klasy.add(posortowane_y[i])
                else:
                    unikalne_x.append(aktualne_x)
                    klasy_dla_x.append(aktualne_klasy)
                    aktualne_x = posortowane_x[i]
                    aktualne_klasy = set([posortowane_y[i]])
            unikalne_x.append(aktualne_x)
            klasy_dla_x.append(aktualne_klasy)

            # Cięcia stawiane są idealnie pośrodku unikalnych wartości różnych klas
            for i in range(len(unikalne_x) - 1):
                polaczone_klasy = klasy_dla_x[i].union(klasy_dla_x[i + 1])
                if len(polaczone_klasy) > 1:
                    miejsce_ciecia = (unikalne_x[i] + unikalne_x[i + 1]) / 2.0
                    aktywne_ciecia.append((f, miejsce_ciecia))

        # Słowniki powiązań do testu nadmiarowości
        ciecie_rozdziela = {ciecie: [] for ciecie in aktywne_ciecia}
        para_przecieta_przez = {p: [] for p in pary_konfliktowe}

        for ciecie in aktywne_ciecia:
            f, wartosc = ciecie
            for p in pary_konfliktowe:
                i, k = p
                xi = macierz_cech[i, f]
                xk = macierz_cech[k, f]
                if (xi <= wartosc < xk) or (xk <= wartosc < xi):
                    ciecie_rozdziela[ciecie].append(p)
                    para_przecieta_przez[p].append(ciecie)

        # Sprawdzenie ile par rozdziela każde cięcie
        sily_ciec = {ciecie: len(ciecie_rozdziela[ciecie]) for ciecie in aktywne_ciecia}

        nietykalne_ciecia = set()

        # Pętla Eliminacyjna
        while True:
            kandydaci = [c for c in aktywne_ciecia if c not in nietykalne_ciecia]
            if not kandydaci:
                break

            najgorsze_ciecie = min(kandydaci, key=lambda c: (sily_ciec[c], c[0], c[1]))

            # Sprawdzenie, czy cięcie jest zbędne
            czy_zbedne = True
            for p in ciecie_rozdziela[najgorsze_ciecie]:
                if len(para_przecieta_przez[p]) <= 1:
                    czy_zbedne = False
                    break

            # Usunięcie cięcia tylko jeśli jest bezużyteczne
            if czy_zbedne:
                aktywne_ciecia.remove(najgorsze_ciecie)
                for p in ciecie_rozdziela[najgorsze_ciecie]:
                    para_przecieta_przez[p].remove(najgorsze_ciecie)
            else:
                nietykalne_ciecia.add(najgorsze_ciecie)

        # Aplikacja cięć na DataFrame
        # Bierzemy nazwy kolumn bez ostatniej (bez decyzji)
        kolumny_cech = df.columns[:-1]

        for f in range(liczba_cech):
            ciecia_cechy = sorted([float(c[1]) for c in aktywne_ciecia if c[0] == f])

            # Krawędzie punktowe (-inf, cięcia, +inf)
            krawedzie = [-np.inf] + ciecia_cechy + [np.inf]

            # Budowa gotowych krotek z góry
            etykiety = [(krawedzie[i], krawedzie[i + 1]) for i in range(len(krawedzie) - 1)]

            # Aplikujemy podział. Wskazujemy f bez przesunięcia, bo cechy zaczynają się od indeksu 0
            wynik_df[kolumny_cech[f]] = pd.cut(df.iloc[:, f], bins=krawedzie, right=False, labels=etykiety)

        return wynik_df

    except Exception as e:
        print(f"Błąd dyskretyzacji: {e}")
        return df