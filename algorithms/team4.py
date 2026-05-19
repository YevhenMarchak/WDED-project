import pandas as pd
import numpy as np

#Dlugosz - 2
def szybkie_sprawdzanie_konfliktow(df, slownik_ciec, kolumny_warunkowe, kolumna_decyzyjna):
    """
    Funkcja pomocnicza do błyskawicznego sprawdzania konfliktów.
    """
    df_temp = pd.DataFrame()
    for col in kolumny_warunkowe:
        df_temp[col] = np.digitize(df[col], slownik_ciec[col])

    df_temp['decyzja'] = df[kolumna_decyzyjna].values

    liczba_konfliktow = 0
    for _, grupa in df_temp.groupby(list(kolumny_warunkowe)):
        if grupa['decyzja'].nunique() > 1:
            liczba_konfliktow += len(grupa)

    return liczba_konfliktow


def discretize(df):
    """
    Metoda zachłannego usuwania najgorszych cięć w dół (True Backward Elimination).
    W każdym kroku ocenia wszystkie cięcia i usuwa to, które wnosi najmniej informacji.
    """
    df_disc = df.copy()
    kolumny_warunkowe = df.columns[:-1]
    kolumna_decyzyjna = df.columns[-1]

    # 1. KROK: Zebranie wszystkich potencjalnych cięć (tylko na zmianach klas)
    slownik_ciec = {}

    for col in kolumny_warunkowe:
        temp = df[[col, kolumna_decyzyjna]].sort_values(by=col)
        wartosci = temp[col].values
        decyzje = temp[kolumna_decyzyjna].values

        wszystkie_ciecia = []
        for i in range(len(wartosci) - 1):
            if wartosci[i] != wartosci[i + 1] and decyzje[i] != decyzje[i + 1]:
                ciecie = (wartosci[i] + wartosci[i + 1]) / 2.0
                wszystkie_ciecia.append(ciecie)

        slownik_ciec[col] = sorted(list(set(wszystkie_ciecia)))

    bazowe_konflikty = szybkie_sprawdzanie_konfliktow(df, slownik_ciec, kolumny_warunkowe, kolumna_decyzyjna)

    # 2. KROK: PRAWDZIWIE ZACHŁANNE USUWANIE W DÓŁ
    while True:
        najlepsze_ciecie_do_usuniecia = None
        najlepsza_kolumna = None
        min_konflikty = float('inf')

        # Testujemy usunięcie każdego cięcia po kolei we wszystkich kolumnach
        for col in kolumny_warunkowe:
            for i in range(len(slownik_ciec[col])):
                testowane_ciecie = slownik_ciec[col].pop(i)
                aktualne_konflikty = szybkie_sprawdzanie_konfliktow(df, slownik_ciec, kolumny_warunkowe,
                                                                    kolumna_decyzyjna)

                # Szukamy usunięcia, które pozostawia jak najmniej konfliktów
                if aktualne_konflikty < min_konflikty:
                    min_konflikty = aktualne_konflikty
                    najlepsze_ciecie_do_usuniecia = testowane_ciecie
                    najlepsza_kolumna = col

                # Przywracamy cięcie na swoje miejsce do dalszych testów
                slownik_ciec[col].insert(i, testowane_ciecie)

        # Jeśli znalezliśmy "najgorsze" cięcie i jego usunięcie NIE zwiększa konfliktów, usuwamy je na stałe
        if najlepsza_kolumna is not None and min_konflikty <= bazowe_konflikty:
            slownik_ciec[najlepsza_kolumna].remove(najlepsze_ciecie_do_usuniecia)
        else:
            # Żadne cięcie nie może już zostać usunięte bez wygenerowania nowych konfliktów. Koniec optymalizacji.
            break

    # 3. KROK: Tworzenie przedziałów (tupli) -inf i inf
    for col in kolumny_warunkowe:
        ostateczne_ciecia = slownik_ciec[col]

        def przypisz_do_przedzialu(val):
            if not ostateczne_ciecia:
                return (float('-inf'), float('inf'))
            if val < ostateczne_ciecia[0]:
                return (float('-inf'), ostateczne_ciecia[0])
            for k in range(len(ostateczne_ciecia) - 1):
                if ostateczne_ciecia[k] <= val < ostateczne_ciecia[k + 1]:
                    return (ostateczne_ciecia[k], ostateczne_ciecia[k + 1])
            return (ostateczne_ciecia[-1], float('inf'))

        df_disc[col] = df_disc[col].apply(przypisz_do_przedzialu)

    return df_disc

