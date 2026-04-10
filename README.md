# MV3 Sensing — DBN + Bi-GRU Reproduction Code

Reproduction code for **Figure 6** (System Dynamics and Bi-GRU Predictions) from:

> **Latent representation of the system's dynamics governed by the Majority Rule
> utilizing Deep Belief Networks and Gated Recurrent Units**  
> Mauricio A. Valle & Gonzalo A. Ruz  
> *ALIFE 2026 — MIT Press*

---

## What this repository contains

| Path | Description |
|---|---|
| `reproduce_sensing.py` | Main reproduction script (fully documented) |
| `data/long_trajectory.csv` | Pre-computed 500-step MV3 latent trajectory |
| `models/layer1_gbrbm.npz` | DBN Layer 1 weights — GB-RBM (784 → 4096) |
| `models/layer2_bbrbm.npz` | DBN Layer 2 weights — BB-RBM (4096 → 225) |
| `models/layer3_bbrbm.npz` | DBN Layer 3 weights — BB-RBM (225 → 81) |
| `models/bigru_modelT50.keras` | Pre-trained Bi-GRU classifier (T = 50) |

---

## What the script does

The script reproduces Figure 6 of the paper in five sequential steps:

1. **Load trajectory dataset** — reads the 500-step MV3 long-trajectory CSV
   containing lattice snapshots, magnetization `|M|`, and control parameter `p`.

2. **Load DBN weights** — loads the three pre-trained RBM layers from `.npz` files.
   No training is performed; weights are fixed as in the published model.

3. **DBN encoding** — propagates each of the 500 lattice snapshots through the
   three DBN layers using a manual feedforward pass, producing a sequence of
   81-dimensional latent vectors `z_t ∈ R^81`.

4. **Bi-GRU inference** — appends the scalar magnetization `|M|` as a
   physics-aware 82nd feature, constructs overlapping sliding windows of length
   `T = 50`, and runs batch inference through the pre-trained Bi-GRU classifier.
   The output is a `(451, 4)` array of softmax probabilities over the four
   trajectory modes.

5. **Figure generation** — produces and saves `sensing_results.pdf`, a
   publication-quality two-panel figure matching Figure 6 of the paper.

---

## Trajectory modes

The Bi-GRU classifies each sliding window into one of four modes:

| Mode | Label | Direction |
|---|---|---|
| 0 | Disorder → Criticality | \|M\| increases |
| 1 | Order → Criticality | \|M\| decreases |
| 2 | Criticality → Disorder | \|M\| decreases |
| 3 | Criticality → Order | \|M\| increases |

---

## Model architecture summary

**DBN encoder** (pre-trained unsupervised on static equilibrium MV3 samples):
```
Input:   784  (28×28 lattice, normalized to {0.0, 0.5, 1.0})
Layer 1: GB-RBM  →  4096  hidden units  (Gaussian-Bernoulli)
Layer 2: BB-RBM  →   225  hidden units  (Bernoulli-Bernoulli)
Layer 3: BB-RBM  →    81  hidden units  (Bernoulli-Bernoulli) ← latent output
```

**Bi-GRU classifier** (trained supervised on trajectory ensembles):
```
Input:   (T=50, 82)  — 81 DBN features + 1 magnetization feature
Layer 1: Bidirectional GRU, 64 units/direction → full sequence output
Layer 2: Bidirectional GRU, 32 units/direction → final hidden state
         GlobalAveragePooling1D
Dense:   32 units, ReLU activation
Output:  4 units, Softmax → trajectory mode probabilities
```

---

## Installation

```bash
git clone https://github.com/<your-username>/mv3-sensing
cd mv3-sensing
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy>=1.23
pandas>=1.5
tensorflow>=2.10
matplotlib>=3.6
seaborn>=0.12
```

> **GPU note**: the script runs on CPU but is significantly faster on a CUDA-enabled
> GPU. TensorFlow will automatically use the GPU if available and correctly configured.

---

## Usage

```bash
python reproduce_sensing.py
```

The script will print step-by-step progress to stdout and save
`sensing_results.pdf` in the current directory.

---

## Expected output

Running the script should produce a figure identical to Figure 6 of the paper:

- **Panel (a)**: Evolution of `p` (orange dashed) and `|M|` (blue) over 500
  Monte Carlo sweeps. The system starts at criticality (`p ≈ 0.85`), ramps to
  supercriticality, then undergoes an abrupt quench to subcriticality at step 300.

- **Panel (b)**: Softmax probability time series for each of the four trajectory
  modes. The Bi-GRU correctly tracks the system's dynamical regime in real time,
  shifting sharply at phase boundaries and responding within `T = 50` steps of
  the quench onset.

---

## Reproducing the t-SNE results (Figure 5)

Figure 5 (t-SNE of the Bi-GRU latent space) requires the full trajectory
ensemble used for training and testing. The ensemble generation code and
t-SNE analysis are provided in the companion notebook
`tsne_latent_analysis.ipynb` (see the `notebooks/` folder).

---

## Citation

If you use this code or the pre-trained models, please cite:

```bibtex
@inproceedings{valle2026mv3,
  author    = {Valle, Mauricio A. and Ruz, Gonzalo A.},
  title     = {Latent representation of the system's dynamics governed by
               the Majority Rule utilizing Deep Belief Networks and
               Gated Recurrent Units},
  booktitle = {Proceedings of the 2026 Conference on Artificial Life (ALIFE)},
  publisher = {MIT Press},
  year      = {2026}
}
```

---

## License

This code is released under the MIT License. The pre-trained model weights and
dataset are released under CC BY 4.0, consistent with the ALIFE 2026 proceedings
license.
