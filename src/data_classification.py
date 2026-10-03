"""Dia 3: KNN (cosine/euclidiana), 10-fold agrupado por sujeito (sem vazamento), k=1..15.

StratifiedGroupKFold(groups=subject_id) mantem todas as imagens de um sujeito no mesmo
lado do split. Grava as probabilidades out-of-fold em outputs/oof_scores.npz.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier

from data_processing import flatten_dataset, load_dataset

N_SPLITS = 10
K_RANGE = tuple(range(1, 16))
METRICS = ("cosine", "euclidean")


def cross_validate_metric(
    X: np.ndarray,
    y_codes: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_classes: int,
    metric: str,
    ks: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """KNN out-of-fold para uma metrica e todos os k (probabilidade e predicao)."""
    proba = {k: np.zeros((len(y_codes), n_classes)) for k in ks}
    pred = {k: np.empty(len(y_codes), dtype=np.int64) for k in ks}

    for i_tr, i_te in folds:
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric).fit(X[i_tr], y_codes[i_tr])
            # realoca pelas colunas de knn.classes_: um fold pode nao ter todas as classes
            proba[k][np.ix_(i_te, knn.classes_.astype(np.int64))] = knn.predict_proba(X[i_te])
            pred[k][i_te] = knn.predict(X[i_te])

    out: dict[str, np.ndarray] = {}
    for k in ks:
        out[f"proba_{metric}_{k}"] = proba[k]
        out[f"y_pred_{metric}_{k}"] = pred[k]
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="KNN + validacao cruzada agrupada por sujeito.")
    parser.add_argument("--data", type=Path, default=here.parent / "data" / "mini_gm_public_v0.1.p")
    parser.add_argument("--out", type=Path, default=here.parent / "outputs")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERRO: arquivo nao encontrado: {args.data}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    X, meta = flatten_dataset(load_dataset(args.data))
    rotulos = meta["syndrome_id"].to_numpy()
    classes = np.unique(rotulos).astype(str)
    cod = {c: i for i, c in enumerate(classes)}
    y_codes = np.array([cod[r] for r in rotulos], dtype=np.int64)
    groups = meta["subject_id"].to_numpy()
    print(f"X={X.shape}, {len(classes)} sindromes, {len(np.unique(groups))} sujeitos.")

    folds = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=False).split(X, y_codes, groups))
    fold_id = np.full(len(y_codes), -1, dtype=np.int64)
    for i, (_tr, te) in enumerate(folds):
        fold_id[te] = i

    arrays: dict[str, np.ndarray] = {"y_true": y_codes, "classes": classes, "fold_id": fold_id}
    for metric in METRICS:
        arrays.update(cross_validate_metric(X, y_codes, folds, len(classes), metric, K_RANGE))

    out_npz = args.out / "oof_scores.npz"
    np.savez(out_npz, ** arrays) #type: ignore
    print(f"Salvo: {out_npz}  ({N_SPLITS} folds, k {K_RANGE[0]}..{K_RANGE[-1]}, metricas {METRICS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
