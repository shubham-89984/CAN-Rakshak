"""
SynCANExtractor — Feature extractor for the SynCAN dataset
===========================================================
Paper: "CANShield: Deep Learning-Based Intrusion Detection Framework
        for Controller Area Networks at the Signal-Level"
        Shahriar et al., IEEE IoT Journal 2023 (arXiv:2205.01306v4)

Implements Module 1 of the CANShield paper as a pipeline FeatureExtractor:
    1. Forward-fill signal matrix from raw SynCAN rows
    2. Min-max normalization (fit on training data only)
    3. Correlation-based hierarchical clustering (reorder signals)
    4. Multi-view window creation (sampling periods T = 1, 5, 10)

Outputs saved to  datasets/SynCAN/features/SynCAN/:
    train_views.npz            {T1, T5, T10} train view arrays
    train_labels.npy           windowed train labels (all 0 for SynCAN train)
    test_<name>_views.npz      test view arrays per attack type
    test_<name>_labels.npy     windowed test labels per attack type
    syncan_config.pkl          normalization params, reorder indices, etc.
    dendrogram.png             signal dendrogram
    correlation.png            correlation matrices before/after reordering
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from features.feature_extractors.base import FeatureExtractor


# ---------------------------------------------------------------------------
# SynCAN dataset constants (from the paper)
# ---------------------------------------------------------------------------
SIGNALS_PER_ID = {
    "id1": 2, "id2": 3, "id3": 2, "id4": 1, "id5": 2,
    "id6": 2, "id7": 2, "id8": 1, "id9": 1, "id10": 4,
}
M = sum(SIGNALS_PER_ID.values())   # 20 signals total
WINDOW_SIZE = 50                    # w
SAMPLING_PERIODS = [1, 5, 10]      # T values for multi-view
STRIDE_TRAIN = 50
STRIDE_TEST  = 50
TEST_NAMES = ["normal", "plateau", "continuous", "playback", "suppress", "flooding"]


class SynCANExtractor(FeatureExtractor):
    """
    Full Module 1 of CANShield, adapted as a pipeline FeatureExtractor.

    The extractor processes ALL SynCAN train and test CSV files in one
    pass and saves pre-computed multi-view tensors.  The CANShield IDS
    model then loads these tensors directly in its train() / test()
    methods, skipping redundant signal-level processing.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Output directory inside the dataset's features/ folder
        self.output_dir = os.path.join(self.features_path, "SynCAN")
        os.makedirs(self.output_dir, exist_ok=True)

        # Build signal name list and ID → column mapping
        self.m = M
        self.w = WINDOW_SIZE
        self.signal_names = []
        self.signal_map = {}          # {id_name: (start_col, n_signals)}
        col = 0
        for id_name, count in SIGNALS_PER_ID.items():
            self.signal_map[id_name] = (col, count)
            for s in range(1, count + 1):
                self.signal_names.append(f"S{s}_{id_name}")
            col += count

        self.extract_features()

    # ------------------------------------------------------------------
    # Required by FeatureExtractor base class
    # ------------------------------------------------------------------

    def extract_features(self):
        """Entry point: run the full SynCAN preprocessing pipeline."""
        print(f"\n{'='*60}")
        print(f"  SynCANExtractor — Feature Extraction")
        print(f"{'='*60}")

        dataset_dir = self.dataset_path   # …/modified_dataset/

        # ── Step 1: Load & forward-fill all training files ──────────
        print("\n[1/6] Loading and forward-filling training data...")
        train_files = sorted(
            f for f in os.listdir(dataset_dir)
            if f.startswith("train_") and f.endswith(".csv")
        )
        if not train_files:
            print("  No train_*.csv files found — skipping extraction.")
            return

        parts, label_parts = [], []
        for fname in train_files:
            print(f"  {fname}...", end=" ", flush=True)
            df = self._load_csv(os.path.join(dataset_dir, fname))
            mat, lab = self._forward_fill(df)
            print(f"rows={len(mat)}")
            parts.append(mat)
            label_parts.append(lab)

        train_matrix = np.concatenate(parts, axis=0)
        train_labels_raw = np.concatenate(label_parts, axis=0)
        print(f"  Combined shape: {train_matrix.shape}")
        del parts, label_parts

        # ── Step 2: Min-max normalization ────────────────────────────
        print("\n[2/6] Min-max normalization...")
        t_min = train_matrix.min(axis=0)
        t_max = train_matrix.max(axis=0)
        t_range = np.where(t_max - t_min == 0, 1.0, t_max - t_min)
        train_norm = np.clip(
            (train_matrix - t_min) / t_range, 0, 1
        ).astype(np.float32)
        del train_matrix

        # ── Step 3: Correlation clustering ───────────────────────────
        print("\n[3/6] Correlation clustering...")
        reorder_idx = self._correlation_clustering(train_norm)
        train_norm = train_norm[:, reorder_idx]
        reordered_names = [self.signal_names[i] for i in reorder_idx]
        print(f"  Signal order: {reordered_names}")

        # ── Step 4: Multi-view training tensors ──────────────────────
        print("\n[4/6] Creating multi-view training tensors...")
        train_views, train_vlabels = self._create_views(
            train_norm, train_labels_raw, stride=STRIDE_TRAIN
        )
        for T in SAMPLING_PERIODS:
            print(f"  T={T}: {train_views[T].shape}")
        del train_norm, train_labels_raw

        # ── Step 5: Save training data & config ──────────────────────
        print("\n[5/6] Saving training views and config...")
        np.savez_compressed(
            os.path.join(self.output_dir, "train_views.npz"),
            **{f"T{T}": train_views[T] for T in SAMPLING_PERIODS},
        )
        np.save(
            os.path.join(self.output_dir, "train_labels.npy"), train_vlabels
        )

        config = {
            "m": self.m,
            "w": self.w,
            "periods": SAMPLING_PERIODS,
            "signal_names": self.signal_names,
            "reordered_names": reordered_names,
            "reorder_indices": reorder_idx,
            "train_min": t_min,
            "train_max": t_max,
            "signals_per_id": SIGNALS_PER_ID,
        }
        with open(os.path.join(self.output_dir, "syncan_config.pkl"), "wb") as fh:
            pickle.dump(config, fh)

        print(f"  Saved: train_views.npz, train_labels.npy, syncan_config.pkl")

        # ── Step 6: Process test files ────────────────────────────────
        print("\n[6/6] Processing test files...")
        for name in TEST_NAMES:
            test_path = os.path.join(dataset_dir, f"test_{name}.csv")
            if not os.path.exists(test_path):
                print(f"  test_{name}.csv not found — skipping")
                continue

            print(f"  test_{name}...", end=" ", flush=True)
            df = self._load_csv(test_path)
            mat, lab = self._forward_fill(df)

            # Normalize with training stats
            test_norm = np.clip(
                (mat - t_min) / t_range, 0, 1
            ).astype(np.float32)
            del mat

            # Reorder columns
            test_norm = test_norm[:, reorder_idx]

            # Create views
            test_views, test_vlabels = self._create_views(
                test_norm, lab, stride=STRIDE_TEST
            )
            del test_norm, lab

            np.savez_compressed(
                os.path.join(self.output_dir, f"test_{name}_views.npz"),
                **{f"T{T}": test_views[T] for T in SAMPLING_PERIODS},
            )
            np.save(
                os.path.join(self.output_dir, f"test_{name}_labels.npy"),
                test_vlabels,
            )
            print(
                f"samples={len(test_vlabels)}, attacks={test_vlabels.sum()}"
            )

        print(f"\n  Extraction complete → {self.output_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_csv(self, path):
        """Load a SynCAN CSV: Label, Time, ID, Signal1–Signal4."""
        df = pd.read_csv(
            path,
            header=None,
            skiprows=1,
            names=["Label", "Time", "ID", "Signal1", "Signal2",
                   "Signal3", "Signal4"],
            engine="python",
            on_bad_lines="warn",
        )
        df["ID"] = df["ID"].astype(str).str.strip()
        return df

    def _forward_fill(self, df):
        """
        Vectorized forward-fill of SynCAN signals into a dense matrix.

        Strategy:
          1. Pre-allocate a (n_rows, m) matrix filled with NaN.
          2. For each CAN ID, scatter its signal values into the
             appropriate columns at the rows where that ID appears.
          3. Use pandas ffill() column-wise to propagate values forward.
          4. Fill any remaining leading NaN with 0.

        Returns:
            matrix : np.float32 array of shape (n, m)
            labels : np.int32 array of shape (n,)
        """
        n = len(df)
        labels = df["Label"].values.astype(np.int32)
        can_ids = df["ID"].values

        # NaN sentinel so ffill can distinguish "not yet seen" from "zero"
        matrix = np.full((n, self.m), np.nan, dtype=np.float32)

        sig_cols = ["Signal1", "Signal2", "Signal3", "Signal4"]
        col_arrays = {c: pd.to_numeric(df[c], errors="coerce").values
                      for c in sig_cols}

        for id_name, (start, count) in self.signal_map.items():
            row_idx = np.where(can_ids == id_name)[0]
            if len(row_idx) == 0:
                continue
            for s in range(count):
                vals = col_arrays[sig_cols[s]][row_idx]
                matrix[row_idx, start + s] = vals

        # Forward-fill then back-fill leading NaNs with 0
        df_mat = pd.DataFrame(matrix)
        df_mat.ffill(inplace=True)
        df_mat.fillna(0.0, inplace=True)

        return df_mat.values.astype(np.float32), labels

    def _correlation_clustering(self, data):
        """
        Compute absolute Pearson correlation, apply hierarchical
        agglomerative clustering (complete linkage) and return the
        leaf order as a numpy index array.  Also saves diagnostic plots.
        """
        corr = np.abs(np.corrcoef(data.T))
        corr = np.nan_to_num(corr, nan=0.0)
        dist = np.clip((1 - corr + (1 - corr).T) / 2, 0, None)
        np.fill_diagonal(dist, 0)

        Z = linkage(squareform(dist), method="complete")

        # Dendrogram (needed for leaf order)
        fig, ax = plt.subplots(figsize=(12, 5))
        dendro = dendrogram(
            Z, labels=self.signal_names,
            leaf_rotation=90, leaf_font_size=8, ax=ax,
        )
        ax.set_title("SynCAN Signal Dendrogram")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "dendrogram.png"), dpi=150)
        plt.close(fig)

        order = np.array(dendro["leaves"])

        # Correlation before / after reordering
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.imshow(corr, cmap="hot", vmin=0, vmax=1)
        ax1.set_title("Correlation — Before")
        corr_after = np.abs(np.nan_to_num(np.corrcoef(data[:, order].T)))
        ax2.imshow(corr_after, cmap="hot", vmin=0, vmax=1)
        ax2.set_title("Correlation — After")
        fig2.tight_layout()
        fig2.savefig(os.path.join(self.output_dir, "correlation.png"), dpi=150)
        plt.close(fig2)

        return order

    def _create_views(self, matrix, labels, stride):
        """
        Slide a window over the signal matrix to create multi-view tensors.

        For each sampling period T and each window position i:
            view[i] = matrix[pos - (w-1)*T : pos+1 : T].T   shape (m, w)

        A window is labelled attack (1) if any row in the widest window
        [pos - max_T*w, pos] contains an attack label.

        Returns:
            views   : {T: np.float32 array (n_samples, m, w)}
            vlabels : np.int32 array (n_samples,)
        """
        total = len(matrix)
        max_T = max(SAMPLING_PERIODS)
        start = max_T * (self.w - 1)

        if start >= total:
            raise ValueError(
                f"Not enough rows for window creation: "
                f"need >{start}, have {total}"
            )

        n_samples = (total - start) // stride
        views = {
            T: np.zeros((n_samples, self.m, self.w), dtype=np.float32)
            for T in SAMPLING_PERIODS
        }
        vlabels = np.zeros(n_samples, dtype=np.int32)

        for i in range(n_samples):
            pos = start + i * stride
            for T in SAMPLING_PERIODS:
                idxs = np.arange(pos - (self.w - 1) * T, pos + 1, T)
                views[T][i] = matrix[idxs].T
            win_start = max(0, pos - max_T * self.w)
            vlabels[i] = 1 if labels[win_start: pos + 1].max() > 0 else 0

            if i % 50_000 == 0 and i > 0:
                print(f"    {i:,}/{n_samples:,}...", flush=True)

        return views, vlabels
