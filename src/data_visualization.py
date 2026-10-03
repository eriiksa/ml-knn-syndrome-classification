"""Dia 2: projecao t-SNE 2D dos embeddings, colorida por syndrome_id."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from data_processing import flatten_dataset, load_dataset

RANDOM_STATE = 42
PERPLEXITY = 30.0


def plot_tsne(X: np.ndarray, meta: pd.DataFrame, out_png: Path) -> None:
    """Ajusta o t-SNE em 2D e salva a dispersao com uma cor por syndrome_id."""
    emb = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    ).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    rotulos = meta["syndrome_id"].astype(str).to_numpy()
    for i, sind in enumerate(sorted(np.unique(rotulos))):
        mask = rotulos == sind
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.7, color=cmap(i % 10), label=sind)

    ax.set_title("t-SNE 2D dos embeddings (320d) por syndrome_id")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(title="syndrome_id", markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="t-SNE 2D dos embeddings.")
    parser.add_argument("--data", type=Path, default=here.parent / "data" / "mini_gm_public_v0.1.p")
    parser.add_argument("--out", type=Path, default=here.parent / "outputs")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERRO: arquivo nao encontrado: {args.data}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    X, meta = flatten_dataset(load_dataset(args.data))
    print(f"X={X.shape}, {meta['syndrome_id'].nunique()} sindromes. Rodando t-SNE ...")

    out_png = args.out / "tsne.png"
    plot_tsne(X, meta, out_png)
    print(f"Salvo: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
