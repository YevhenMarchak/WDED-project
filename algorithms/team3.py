import pandas as pd
import numpy as np
import bisect

# Wojciech Knapik
def discretize(df):
    """
    implementacja zachłannego algorytmu dyskretyzacji wstępującej.
    """
    conditional_cols = df.columns[:-1]
    decision_col = df.columns[-1]

    X = df[conditional_cols].values
    y = df[decision_col].values
    n_samples, n_features = X.shape

    # Etap 1: Wyznaczenie potencjalnych punktów cięć
    cuts_per_feature = []
    all_cuts = []
    feature_cut_offsets = []

    current_offset = 0
    for j in range(n_features):
        unique_vals = np.unique(X[:, j])
        sorted_vals = np.sort(unique_vals)
        # wyliczenie średnich używając wektoryzacji NumPy
        cuts = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
        cuts_per_feature.append(cuts.tolist())
        feature_cut_offsets.append(current_offset)
        for cut_val in cuts:
            all_cuts.append((j, cut_val))
        current_offset += len(cuts)

    n_cuts = len(all_cuts)

    # Etap 2: identyfikacja par do odseparowania
    pairs_to_separate = []
    for i in range(n_samples):
        # inna klasa decyzyjna
        diff_mask = y[i + 1:] != y[i]
        # różne wektory cech (przynajmniej jedna cecha inna)
        feat_diff_mask = np.any(X[i + 1:] != X[i], axis=1)

        # Pobranie indeksów spełniających oba warunki
        valid_j_indices = np.where(diff_mask & feat_diff_mask)[0] + i + 1
        for j in valid_j_indices:
            pairs_to_separate.append((i, j))

    n_pairs = len(pairs_to_separate)

    # Etap 3: Mapowanie z użyciem wyszukiwania binarnego
    cut_to_pairs = [set() for _ in range(n_cuts)]
    pair_to_cuts = [list() for _ in range(n_pairs)]

    for p_idx, (u, v) in enumerate(pairs_to_separate):
        val_u_all = X[u]
        val_v_all = X[v]

        for f in range(n_features):
            val_u = val_u_all[f]
            val_v = val_v_all[f]
            if val_u == val_v:
                continue

            if val_u > val_v:
                val_u, val_v = val_v, val_u

            cuts_f = cuts_per_feature[f]
            if not cuts_f:
                continue

            # Binary search znajduje zakres indeksów cięć separujących parę
            idx_start = bisect.bisect_left(cuts_f, val_u)
            idx_end = bisect.bisect_left(cuts_f, val_v)

            offset = feature_cut_offsets[f]
            for local_c_idx in range(idx_start, idx_end):
                global_c_idx = offset + local_c_idx
                cut_to_pairs[global_c_idx].add(p_idx)
                pair_to_cuts[p_idx].append(global_c_idx)

    # Etap 4: Zachłanna redukcja Bottom-Up
    active_cuts = set(range(n_cuts))
    pair_cover_counts = np.array([len(cuts) for cuts in pair_to_cuts], dtype=np.int32)

    # ilość par które dane cięcie początkowo pokrywa
    cut_scores = [len(pairs) for pairs in cut_to_pairs]

    removable_cuts = set()
    for c_idx in active_cuts:
        can_remove = True
        for p_idx in cut_to_pairs[c_idx]:
            if pair_cover_counts[p_idx] <= 1:
                can_remove = False
                break
        if can_remove:
            removable_cuts.add(c_idx)

    while removable_cuts:
        # Heurystyka WorstCut
        worst_cut = min(removable_cuts, key=lambda c: cut_scores[c])

        removable_cuts.remove(worst_cut)
        active_cuts.remove(worst_cut)

        for p_idx in cut_to_pairs[worst_cut]:
            pair_cover_counts[p_idx] -= 1
            # Jeśli parę oddziela już tylko jedno cięcie to cięcie staje się nietykalne
            if pair_cover_counts[p_idx] == 1:
                for remaining_c_idx in pair_to_cuts[p_idx]:
                    if remaining_c_idx in active_cuts:
                        if remaining_c_idx in removable_cuts:
                            removable_cuts.remove(remaining_c_idx)
                        break

    # Etap 5: Transformacja finalna
    final_cuts_per_feature = {j: [] for j in range(n_features)}
    for c_idx in active_cuts:
        feat_idx, cut_val = all_cuts[c_idx]
        final_cuts_per_feature[feat_idx].append(cut_val)

    for j in range(n_features):
        final_cuts_per_feature[j].sort()

    discretized_df = df.copy()

    for j, col in enumerate(conditional_cols):
        cuts = final_cuts_per_feature[j]
        bins = [-np.inf] + cuts + [np.inf]
        bin_indices = pd.cut(df[col], bins=bins, right=False, labels=False)
        tuple_labels = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        discretized_df[col] = [tuple_labels[idx] for idx in bin_indices]

    return discretized_df
