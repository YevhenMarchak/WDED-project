import pandas as pd
import numpy as np

#Knapik - 6

def discretize(df):
    """
    Implementacja zachłannego algorytmu dyskretyzacji wstępującej (Bottom-Up).

    Algorytm redukuje pełen zbiór potencjalnych cięć, usuwając w każdej iteracji
    cięcie separujące najmniejszą liczbę par obiektów z różnych klas decyzyjnych (WorstCut),
    pod warunkiem zachowania spójności (każda para pozostaje rozróżnialna).

    Parametry:
    df (pd.DataFrame): Ramka danych, gdzie ostatnia kolumna to atrybut decyzyjny.

    Zwraca:
    pd.DataFrame: Kopia ramki danych z dyskretyzowanymi atrybutami warunkowymi.
                  Wartości zamieniane są na krotki reprezentujące przedziały [a, b).
    """

    # Inicjalizacja: podział na atrybuty warunkowe i klasę decyzyjną
    conditional_cols = df.columns[:-1]
    decision_col = df.columns[-1]

    X = df[conditional_cols].values
    y = df[decision_col].values
    n_samples, n_features = X.shape

    # Etap 1: Wyznaczenie potencjalnych punktów cięć
    # Cięcia definiowane jako średnie wartości między posortowanymi unikalnymi wartościami w kolumnach
    all_cuts = []
    for j in range(n_features):
        unique_vals = np.unique(X[:, j])
        sorted_vals = np.sort(unique_vals)
        for i in range(len(sorted_vals) - 1):
            cut_val = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0
            all_cuts.append((j, cut_val))

    # Zdefiniowanie zbioru par obiektów należących do różnych klas decyzyjnych.
    # Pominięcie konfliktów deterministycznych (identyczne wektory cech, różna decyzja).
    pairs_to_separate = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if y[i] != y[j]:
                if not np.array_equal(X[i], X[j]):
                    pairs_to_separate.append((i, j))

    # Utworzenie struktur mapujących relacje: cięcie <-> separowane pary
    cut_to_pairs = {c_idx: set() for c_idx in range(len(all_cuts))}
    pair_to_cuts = {p_idx: set() for p_idx in range(len(pairs_to_separate))}

    for p_idx, (u, v) in enumerate(pairs_to_separate):
        for c_idx, (feat_idx, cut_val) in enumerate(all_cuts):
            val_u = X[u, feat_idx]
            val_v = X[v, feat_idx]

            if (val_u <= cut_val < val_v) or (val_v <= cut_val < val_u):
                cut_to_pairs[c_idx].add(p_idx)
                pair_to_cuts[p_idx].add(c_idx)

    # Etap 2: Zachłanna redukcja cięć (Bottom-Up)
    active_cuts = set(range(len(all_cuts)))
    pair_cover_counts = {p_idx: len(cuts) for p_idx, cuts in pair_to_cuts.items()}

    while True:
        removable_cuts = []

        # Identyfikacja cięć nadmiarowych, których usunięcie nie spowoduje
        # utraty rozróżnialności żadnej z badanych par.
        for c_idx in active_cuts:
            can_remove = True
            for p_idx in cut_to_pairs[c_idx]:
                if pair_cover_counts[p_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                removable_cuts.append(c_idx)

        # Warunek stopu: brak cięć do usunięcia
        if not removable_cuts:
            break

        # Heurystyka WorstCut: wybór i usunięcie cięcia separującego najmniejszą liczbę par
        worst_cut = min(removable_cuts, key=lambda c: len(cut_to_pairs[c]))

        active_cuts.remove(worst_cut)
        for p_idx in cut_to_pairs[worst_cut]:
            pair_cover_counts[p_idx] -= 1

    # Etap 3: Transformacja ramki danych z wykorzystaniem wyselekcjonowanych cięć
    final_cuts_per_feature = {j: [] for j in range(n_features)}
    for c_idx in active_cuts:
        feat_idx, cut_val = all_cuts[c_idx]
        final_cuts_per_feature[feat_idx].append(cut_val)

    for j in range(n_features):
        final_cuts_per_feature[j].sort()

    discretized_df = df.copy()

    for j, col in enumerate(conditional_cols):
        cuts = final_cuts_per_feature[j]

        # Definicja przedziałów ze skrajnymi wartościami nieskończonymi
        bins = [-np.inf] + cuts + [np.inf]

        # Generowanie indeksów przedziałów (lewostronnie domkniętych)
        bin_indices = pd.cut(df[col], bins=bins, right=False, labels=False)

        # Konwersja do oczekiwanego formatu krotek (tupli)
        tuple_labels = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

        discretized_df[col] = [tuple_labels[idx] for idx in bin_indices]

    return discretized_df
