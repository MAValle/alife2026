"""
reproduce_sensing.py
====================
Reproduction script for Figure 6 (System Dynamics and Bi-GRU Predictions) from:

    "Latent representation of the system's dynamics governed by the Majority Rule
     utilizing Deep Belief Networks and Gated Recurrent Units"
    Valle, M. A. & Ruz, G. A. — ALIFE 2026

This script:
  1. Loads the pre-computed latent trajectory dataset (500-step MV3 sequence).
  2. Loads the three pre-trained DBN layer weights (.npz files) and encodes
     each snapshot into an 81-dimensional latent vector using a manual
     feedforward pass (no training required).
  3. Appends the system magnetization as an 82nd physics-aware feature.
  4. Constructs overlapping sliding windows of length T=50 and runs batch
     inference through the pre-trained Bi-GRU classifier.
  5. Produces and saves the publication-quality Figure 6 (sensing_results.pdf).

Repository layout expected
--------------------------
  reproduce_sensing.py         <- this file
  data/
      long_trajectory.csv      <- 500-step MV3 latent trajectory dataset
  models/
      layer1_gbrbm.npz         <- DBN Layer 1 weights (GB-RBM, 784->4096)
      layer2_bbrbm.npz         <- DBN Layer 2 weights (BB-RBM, 4096->225)
      layer3_bbrbm.npz         <- DBN Layer 3 weights (BB-RBM, 225->81)
      bigru_modelT50.keras     <- Pre-trained Bi-GRU classifier (T=50)

Dataset columns
---------------
  step    : Monte Carlo step index (int)
  p       : control parameter value at that step (float)
  mag     : system magnetization |M| at that step (float)
  lattice : 784-dimensional flattened lattice state, scaled to [0, 0.5, 1.0]
            stored as a string representation of a numpy array

Dependencies
------------
  numpy >= 1.23
  pandas >= 1.5
  tensorflow >= 2.10
  matplotlib >= 3.6
  seaborn >= 0.12

Usage
-----
  python reproduce_sensing.py

The output file sensing_results.pdf will be saved in the current directory.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — edit these paths if your layout differs
# =============================================================================

DATA_PATH   = os.path.join("data", "long_trajectory.csv")

MODEL_PATHS = {
    "layer1": os.path.join("models", "layer1_gbrbm.npz"),
    "layer2": os.path.join("models", "layer2_bbrbm.npz"),
    "layer3": os.path.join("models", "layer3_bbrbm.npz"),
}

GRU_MODEL_PATH = os.path.join("models", "bigru_modelT50.keras")

# DBN hyperparameters
SIGMA      = 0.01   # GB-RBM Gaussian noise parameter (must match training)
LATENT_DIM = 81     # Dimensionality of Layer 3 (last DBN layer used)

# Sliding window parameters
T          = 50     # Window length (time stamps per GRU input sequence)
BATCH_SIZE = 64     # Batch size for Bi-GRU inference

# Output
OUTPUT_PDF = "sensing_results.pdf"

# =============================================================================
# STEP 1 — LOAD TRAJECTORY DATASET
# =============================================================================

def load_trajectory(path: str) -> tuple:
    """
    Load the 500-step MV3 long trajectory dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    lattice_full : np.ndarray, shape (500, 784)
        Normalized lattice snapshots, scaled to {0.0, 0.5, 1.0}.
    mag_features : np.ndarray, shape (500, 1)
        Magnetization |M| at each step.
    m_history : list of float
        Magnetization values for plotting.
    p_schedule : list of float
        Control parameter p values for plotting.
    """
    print("=" * 60)
    print("STEP 1 — Loading trajectory dataset")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"    Dataset shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    # Parse the 'lattice' column (stored as string repr of numpy array)
    def parse_lattice(s):
        if isinstance(s, str):
            return np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)
        return np.asarray(s, dtype=np.float32)

    lattice_full = np.stack(df["lattice"].apply(parse_lattice).values)
    print(f"    Lattice array shape: {lattice_full.shape}")  # (500, 784)

    mag_features = df["mag"].to_numpy().reshape(-1, 1)
    m_history    = df["mag"].tolist()
    p_schedule   = df["p"].tolist()

    print(f"    ✓ Dataset loaded successfully.\n")
    return lattice_full, mag_features, m_history, p_schedule


# =============================================================================
# STEP 2 — LOAD DBN WEIGHTS
# =============================================================================

def load_dbn_weights(model_paths: dict) -> dict:
    """
    Load pre-trained DBN weights from .npz files.

    The DBN consists of:
      - Layer 1: Gaussian-Bernoulli RBM (GB-RBM), 784 -> 4096
      - Layer 2: Bernoulli-Bernoulli RBM (BB-RBM), 4096 -> 225
      - Layer 3: Bernoulli-Bernoulli RBM (BB-RBM), 225 -> 81

    Each .npz file is expected to contain the keys:
      'W'      : weight matrix
      'v_bias' : visible bias vector
      'h_bias' : hidden bias vector

    Parameters
    ----------
    model_paths : dict
        Dictionary with keys 'layer1', 'layer2', 'layer3' mapping to
        the corresponding .npz file paths.

    Returns
    -------
    weights : dict
        Dictionary containing W1, W2, W3 and their bias vectors.
    """
    print("=" * 60)
    print("STEP 2 — Loading DBN layer weights")
    print("=" * 60)

    weights = {}
    layer_info = [
        ("layer1", "W1", "v_bias1", "h_bias1", "784 -> 4096  (GB-RBM)"),
        ("layer2", "W2", "v_bias2", "h_bias2", "4096 -> 225  (BB-RBM)"),
        ("layer3", "W3", "v_bias3", "h_bias3", "225 -> 81    (BB-RBM)"),
    ]

    for key, wk, vk, hk, desc in layer_info:
        path = model_paths[key]
        data = np.load(path)
        weights[wk] = tf.constant(data["W"],      dtype=tf.float32)
        weights[vk] = tf.constant(data["v_bias"], dtype=tf.float32)
        weights[hk] = tf.constant(data["h_bias"], dtype=tf.float32)
        print(f"    Layer {key[-1]} ({desc})")
        print(f"        W shape     : {weights[wk].shape}")
        print(f"        v_bias shape: {weights[vk].shape}")
        print(f"        h_bias shape: {weights[hk].shape}")

    print(f"\n    ✓ All 3 DBN layers loaded successfully.\n")
    return weights


# =============================================================================
# STEP 3 — DBN FORWARD PASS (ENCODING)
# =============================================================================

def dbn_encode(lattice_batch: np.ndarray, weights: dict,
               sigma: float = SIGMA) -> np.ndarray:
    """
    Encode a batch of lattice snapshots through the pre-trained DBN.

    The forward pass computes mean-field activations (sigmoid of pre-activation)
    at each layer, consistent with the training procedure used in Valle & Ruz (2026).

    Layer 1 (GB-RBM) activation:
        h1 = sigmoid( v @ (W1 / sigma^2) + h_bias1 )

    Layers 2-3 (BB-RBM) activation:
        h_l = sigmoid( h_{l-1} @ W_l + h_bias_l )

    Parameters
    ----------
    lattice_batch : np.ndarray, shape (N, 784)
        Normalized lattice snapshots (values in {0.0, 0.5, 1.0}).
    weights : dict
        DBN weights loaded by load_dbn_weights().
    sigma : float
        Gaussian noise parameter for the GB-RBM visible layer.

    Returns
    -------
    np.ndarray, shape (N, 81)
        81-dimensional latent representations.
    """
    v = tf.constant(lattice_batch, dtype=tf.float32)

    # Layer 1: GB-RBM  (784 -> 4096)
    h1 = tf.nn.sigmoid(
        tf.matmul(v, weights["W1"] / tf.square(sigma)) + weights["h_bias1"]
    )

    # Layer 2: BB-RBM  (4096 -> 225)
    h2 = tf.nn.sigmoid(
        tf.matmul(h1, weights["W2"]) + weights["h_bias2"]
    )

    # Layer 3: BB-RBM  (225 -> 81)
    h3 = tf.nn.sigmoid(
        tf.matmul(h2, weights["W3"]) + weights["h_bias3"]
    )

    return h3.numpy()


# =============================================================================
# STEP 4 — ASSEMBLE GRU INPUT AND RUN INFERENCE
# =============================================================================

def build_sliding_windows(latent: np.ndarray, mag: np.ndarray,
                           T: int = 50) -> np.ndarray:
    """
    Concatenate DBN latent vectors with magnetization and build sliding windows.

    Each window has shape (T, 82): the 81-dimensional DBN latent vector
    augmented with the scalar magnetization |M| as a physics-aware 82nd feature,
    making the Bi-GRU "physics-informed" at inference time.

    Parameters
    ----------
    latent : np.ndarray, shape (N, 81)
        DBN-encoded latent representations.
    mag : np.ndarray, shape (N, 1)
        System magnetization at each step.
    T : int
        Sliding window length (number of time stamps per GRU sequence).

    Returns
    -------
    gru_input : np.ndarray, shape (N - T + 1, T, 82)
        3D tensor ready for Bi-GRU batch inference.
    """
    # Append magnetization as 82nd feature
    features = np.hstack([latent, mag])               # (N, 82)
    n_steps  = len(features)

    windows = np.stack(
        [features[i : i + T] for i in range(n_steps - T + 1)],
        axis=0
    ).astype(np.float32)                               # (N-T+1, T, 82)

    print(f"    Sliding windows shape: {windows.shape}")
    print(f"    (each window: T={T} steps × 82 features)\n")
    return windows


def run_gru_inference(model_path: str, gru_input: np.ndarray,
                      batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Load the pre-trained Bi-GRU and run batch inference.

    Parameters
    ----------
    model_path : str
        Path to the saved .keras Bi-GRU model.
    gru_input : np.ndarray, shape (n_windows, T, 82)
        Sliding window sequences.
    batch_size : int
        Mini-batch size for GPU inference.

    Returns
    -------
    predictions : np.ndarray, shape (n_windows, 4)
        Softmax probability vectors over the four trajectory modes:
          0 — Disorder → Criticality  (|M| increases)
          1 — Order    → Criticality  (|M| decreases)
          2 — Criticality → Disorder  (|M| decreases)
          3 — Criticality → Order     (|M| increases)
    """
    print("=" * 60)
    print("STEP 4 — Bi-GRU inference")
    print("=" * 60)

    model = tf.keras.models.load_model(model_path)
    model.summary()

    print(f"\n    Running inference on {gru_input.shape[0]} windows ...")
    predictions = model.predict(gru_input, batch_size=batch_size, verbose=1)
    print(f"    Predictions shape: {predictions.shape}")  # (n_windows, 4)
    print(f"    ✓ Inference complete.\n")
    return predictions


# =============================================================================
# STEP 5 — PUBLICATION-QUALITY FIGURE (Figure 6)
# =============================================================================

def plot_sensing_results(p_schedule: list, m_history: list,
                         predictions: np.ndarray, T: int = 50,
                         output_path: str = OUTPUT_PDF) -> None:
    """
    Reproduce Figure 6: System Dynamics and Bi-GRU Predictions.

    Produces a two-panel figure:
      Panel (a): Evolution of control parameter p (orange dashed) and
                 magnetization |M| (blue), dual y-axes.
      Panel (b): Softmax probability time series for each of the four
                 trajectory modes, with a vertical marker at the quench onset.

    The figure is saved as a PDF suitable for LaTeX inclusion in a
    two-column IEEE/ALIFE conference paper (width ~3.5 inches).

    Parameters
    ----------
    p_schedule : list of float, length N
        Values of the control parameter p at each Monte Carlo step.
    m_history : list of float, length N
        Magnetization |M| at each step.
    predictions : np.ndarray, shape (N - T + 1, 4)
        Bi-GRU softmax outputs from run_gru_inference().
    T : int
        Sliding window length; determines x-axis offset for predictions.
    output_path : str
        File path for the saved figure.
    """
    # ------------------------------------------------------------------ setup
    plt.style.use("seaborn-v0_8-white")
    rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman"],
        "font.size":         9,
        "axes.linewidth":    0.8,
        "axes.labelsize":    10,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   7,
        "legend.frameon":    False,
        "figure.dpi":        300,
        "savefig.dpi":       300,
    })

    n_windows = predictions.shape[0]
    # Time axis: prediction window i covers steps [i, i+T), labelled at i+T-1
    time_axis = np.arange(T - 1, T - 1 + n_windows)

    # Colorblind-safe palette (Wong 2011)
    COLOR_P      = "#D55E00"   # orange-brown  → parameter p
    COLOR_M      = "#0072B2"   # blue          → magnetization
    TRAJ_COLORS  = ["#009E73", "#D55E00", "#0072B2", "#CC79A7"]
    # green, orange, blue, pink  →  modes 0, 1, 2, 3

    TRAJ_LABELS = [
        "Disorder → Critical (M↑)",
        "Order → Critical (M↓)",
        "Critical → Disorder (M↓)",
        "Critical → Order (M↑)",
    ]

    # ------------------------------------------------------------------ figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # ---- Panel (a): system physics ----------------------------------------
    ax1.plot(p_schedule, color=COLOR_P, linestyle="--",
             linewidth=1.5, alpha=0.8, label="Parameter $p$")
    ax1.set_ylabel("$p$", color=COLOR_P, fontsize=10, labelpad=4)
    ax1.tick_params(axis="y", labelcolor=COLOR_P)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(m_history, color=COLOR_M,
                  linewidth=1.8, label="Magnetization $|M|$")
    ax1_twin.set_ylabel("$|M|$", color=COLOR_M, fontsize=10, labelpad=4)
    ax1_twin.tick_params(axis="y", labelcolor=COLOR_M)
    ax1_twin.set_ylim(-0.05, 1.10)

    ax1.set_title("(a) System Dynamics", fontsize=8, pad=10)
    ax1.grid(False)
    ax1_twin.grid(False)

    for ax in (ax1, ax1_twin):
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

    # ---- Panel (b): Bi-GRU probability streams ----------------------------
    for i in range(4):
        ax2.plot(time_axis, predictions[:, i],
                 label=TRAJ_LABELS[i],
                 color=TRAJ_COLORS[i],
                 linewidth=1.5, alpha=0.85)

    # Quench onset marker (step 300 in the experiment described in the paper)
    QUENCH_STEP = 300
    ax2.axvline(x=QUENCH_STEP, color="black", linestyle=":", linewidth=1, alpha=0.7)
    ax2.text(QUENCH_STEP + 2, 0.95, "Quench start",
             rotation=90, verticalalignment="top", fontsize=8, style="italic")

    ax2.set_ylabel("Predicted probability", fontsize=10, labelpad=4)
    ax2.set_xlabel("Time step (MC sweep)",  fontsize=10, labelpad=4)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(time_axis[0], time_axis[-1])
    ax2.set_title("(b) Bi-GRU Predictions", fontsize=8, pad=10)
    ax2.grid(False)

    ax2.tick_params(axis="both", which="major",
                    direction="in", length=4, width=0.7)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.7)

    # Legend below panel (b)
    handles, leg_labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, leg_labels,
               loc="upper center",
               bbox_to_anchor=(0.5, -0.18),
               ncol=4,
               frameon=False,
               fontsize=6.5,
               handlelength=1.2,
               handletextpad=0.5)

    # ---- Save ---------------------------------------------------------------
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"    ✓ Figure saved to '{output_path}'")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  MV3 Sensing Reproduction Script")
    print("  Valle & Ruz — ALIFE 2026")
    print("=" * 60 + "\n")

    # Step 1 — Load trajectory data
    lattice_full, mag_features, m_history, p_schedule = load_trajectory(DATA_PATH)

    # Step 2 — Load DBN weights
    weights = load_dbn_weights(MODEL_PATHS)

    # Step 3 — Encode all snapshots through the DBN
    print("=" * 60)
    print("STEP 3 — DBN encoding (manual feedforward pass)")
    print("=" * 60)
    latent_representations = dbn_encode(lattice_full, weights, sigma=SIGMA)
    print(f"    Latent representations shape: {latent_representations.shape}")
    print(f"    ✓ Encoding complete.\n")

    # Step 4 — Build sliding windows and run Bi-GRU inference
    print("=" * 60)
    print("STEP 4a — Building sliding windows (T={})".format(T))
    print("=" * 60)
    gru_input = build_sliding_windows(latent_representations, mag_features, T=T)
    predictions = run_gru_inference(GRU_MODEL_PATH, gru_input, batch_size=BATCH_SIZE)

    # Step 5 — Plot and save Figure 6
    print("=" * 60)
    print("STEP 5 — Generating Figure 6")
    print("=" * 60)
    plot_sensing_results(p_schedule, m_history, predictions,
                         T=T, output_path=OUTPUT_PDF)

    print("\n✓ All steps completed successfully.")
    print(f"  Output saved to: {OUTPUT_PDF}\n")


if __name__ == "__main__":
    main()
