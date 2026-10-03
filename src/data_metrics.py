"""Dia 4: metricas a mao (AUC/F1/top-k) a partir de outputs/oof_scores.npz.

Implementacao manual em numpy; o scikit-learn entra so como conferencia
(tolerancia 1e-6). Gera a tabela de resultados e as figuras.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, top_k_accuracy_score

METRICS = ("cosine", "euclidean")
K_RANGE = tuple(range(1, 16))
TOP_K_VALUES = (1, 3, 5)
TOLERANCIA = 1e-6
FPR_GRID = np.linspace(0.0, 1.0, 101)


def matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Matriz de confusao n_classes x n_classes a partir de rotulos inteiros."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precisao_recall_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precisao, recall e F1 por classe (0 onde nao ha tp+fp ou tp+fn)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precisao = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    soma = precisao + recall
    f1 = np.divide(2.0 * precisao * recall, soma, out=np.zeros_like(tp), where=soma > 0)
    return precisao, recall, f1


def f1_macro_manual(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, np.ndarray]:
    """F1 macro (media simples do F1 por classe) e a matriz de confusao."""
    cm = matriz_confusao(y_true, y_pred, n_classes)
    _, _, f1 = precisao_recall_f1(cm)
    return float(f1.mean()), cm


def trapezio(y: np.ndarray, x: np.ndarray) -> float:
    """Integral pela regra do trapezio, sem depender de np.trapz."""
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


def curva_roc_manual(y_bin: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC binaria; empates de score caem num unico limiar (convencao do sklearn)."""
    ordem = np.argsort(-scores, kind="mergesort")
    scores_ord, y_ord = scores[ordem], y_bin[ordem]
    idx = np.r_[np.where(np.diff(scores_ord))[0], len(scores_ord) - 1]
    tps = np.cumsum(y_ord)[idx]
    fps = idx + 1 - tps
    fpr = np.r_[0, fps] / fps[-1]
    tpr = np.r_[0, tps] / tps[-1]
    return fpr, tpr, np.r_[np.inf, scores_ord[idx]]


def auc_manual(y_bin: np.ndarray, scores: np.ndarray) -> float:
    """AUC binaria: area sob a curva ROC."""
    fpr, tpr, _ = curva_roc_manual(y_bin, scores)
    return trapezio(tpr, fpr)


def auc_macro_manual(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> tuple[float, np.ndarray]:
    """AUC one-vs-rest com media macro entre as classes."""
    aucs = np.array([auc_manual((y_true == c).astype(np.int64), proba[:, c]) for c in range(n_classes)])
    return float(aucs.mean()), aucs


def roc_media_manual(
    y_true: np.ndarray, proba: np.ndarray, fold_id: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """ROC media macro: curvas one-vs-rest por (fold, classe) no FPR comum."""
    curvas = []
    for fold in np.unique(fold_id):
        no_fold = fold_id == fold
        for c in range(n_classes):
            y_bin = (y_true[no_fold] == c).astype(np.int64)
            if y_bin.sum() == 0:
                continue  # classe ausente no fold: ROC indefinida
            fpr, tpr, _ = curva_roc_manual(y_bin, proba[no_fold, c])
            fpr_u, inicio = np.unique(fpr, return_index=True)
            curvas.append(np.interp(FPR_GRID, fpr_u, np.maximum.reduceat(tpr, inicio)))
    tpr_media = np.mean(curvas, axis=0)
    return FPR_GRID, tpr_media, trapezio(tpr_media, FPR_GRID)


def top_k_manual(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    """Top-k accuracy com o desempate do sklearn (argsort estavel + reversao)."""
    ordem = np.argsort(proba, axis=1, kind="mergesort")[:, ::-1]
    return float(np.mean((ordem[:, :k] == y_true[:, None]).any(axis=1)))


def metricas_sklearn(y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Mesmas metricas pelo scikit-learn, apenas para conferencia."""
    rotulos = np.arange(proba.shape[1])
    out = {
        "auc": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
        "f1": float(f1_score(y_true, y_pred, average="macro", labels=rotulos)),
    }
    for tk in TOP_K_VALUES:
        out[f"top{tk}"] = float(top_k_accuracy_score(y_true, proba, k=tk, labels=rotulos))
    return out


def figura_roc_comparativa(
    roc_por_metrica: dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]],
    melhor_k: dict[str, int],
    out_png: Path,
) -> None:
    """Curvas ROC medias de cosine e euclidean no mesmo grafico (item 4)."""
    cores = {"cosine": "tab:blue", "euclidean": "tab:red"}
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for metrica in METRICS:
        k = melhor_k[metrica]
        fpr, tpr, auc_media = roc_por_metrica[metrica][k]
        ax.plot(fpr, tpr, color=cores[metrica], linewidth=1.8,
                label=f"{metrica} (k={k}, AUC media={auc_media:.3f})")
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="aleatorio")
    ax.set_xlabel("Taxa de falsos positivos (FPR)")
    ax.set_ylabel("Taxa de verdadeiros positivos (TPR)")
    ax.set_title("Curvas ROC medias (macro, one-vs-rest) — cosine vs euclidean")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def figura_confusao(cm: np.ndarray, classes: list[str], metrica: str, k: int, out_png: Path) -> None:
    """Matriz de confusao do melhor k de uma metrica."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="nº de imagens")
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(classes)), labels=classes, fontsize=8)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(f"Matriz de confusão — KNN {metrica}, k={k} (melhor F1 macro)")
    limiar = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > limiar else "black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Metricas manuais de AUC/F1/top-k a partir do oof_scores.npz.")
    parser.add_argument("--npz", type=Path, default=here.parent / "outputs" / "oof_scores.npz")
    parser.add_argument("--out", type=Path, default=here.parent / "outputs")
    args = parser.parse_args()

    if not args.npz.exists():
        print(f"ERRO: {args.npz} nao encontrado. Rode data_classification.py antes.", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    with np.load(args.npz) as d:
        arrays = {chave: d[chave] for chave in d.files}
    y_true = arrays["y_true"].astype(np.int64)
    fold_id = arrays["fold_id"].astype(np.int64)
    classes = [str(c) for c in arrays["classes"]]
    n_classes = len(classes)

    linhas: list[dict] = []
    roc_por_metrica: dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]] = {}
    cm_por_k: dict[tuple[str, int], np.ndarray] = {}

    for metrica in METRICS:
        roc_por_metrica[metrica] = {}
        for k in K_RANGE:
            proba = arrays[f"proba_{metrica}_{k}"]
            y_pred = arrays[f"y_pred_{metrica}_{k}"]

            auc_man, _ = auc_macro_manual(y_true, proba, n_classes)
            f1_man, cm = f1_macro_manual(y_true, y_pred, n_classes)
            sk = metricas_sklearn(y_true, proba, y_pred)

            linha = {
                "metrica": metrica, "k": k,
                "auc_manual": auc_man, "auc_sklearn": sk["auc"],
                "f1_manual": f1_man, "f1_sklearn": sk["f1"],
            }
            diffs = [abs(auc_man - sk["auc"]), abs(f1_man - sk["f1"])]
            for tk in TOP_K_VALUES:
                linha[f"top{tk}_manual"] = top_k_manual(y_true, proba, tk)
                linha[f"top{tk}_sklearn"] = sk[f"top{tk}"]
                diffs.append(abs(linha[f"top{tk}_manual"] - sk[f"top{tk}"]))
            linha["max_diff"] = float(max(diffs))
            linhas.append(linha)

            roc_por_metrica[metrica][k] = roc_media_manual(y_true, proba, fold_id, n_classes)
            cm_por_k[(metrica, k)] = cm

    tabela = pd.DataFrame(linhas)
    max_diff = float(tabela["max_diff"].max())
    print(f"Conferencia manual x scikit-learn: max diff = {max_diff:.3e} "
          f"(tolerancia {TOLERANCIA:.0e}) -> {'OK' if max_diff <= TOLERANCIA else 'DIVERGENCIA'}")
    print("\nResumo (manual):")
    print(tabela[["metrica", "k", "auc_manual", "f1_manual", "top1_manual", "top3_manual", "top5_manual"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    melhor_k: dict[str, int] = {}
    for metrica in METRICS:
        sub = tabela[tabela["metrica"] == metrica]
        melhor_k[metrica] = int(sub.sort_values(["f1_manual", "k"], ascending=[False, True]).iloc[0]["k"])
    print("\nMelhor k por F1 macro: " + ", ".join(f"{m}={melhor_k[m]}" for m in METRICS))

    tabela.to_csv(args.out / "metrics_summary.csv", index=False, float_format="%.10g")
    figura_roc_comparativa(roc_por_metrica, melhor_k, args.out / "roc_mean_comparison.png")
    for metrica in METRICS:
        k = melhor_k[metrica]
        figura_confusao(cm_por_k[(metrica, k)], classes, metrica, k,
                        args.out / f"confusion_{metrica}_k{k}.png")
    print(f"\nSalvo em {args.out}: metrics_summary.csv, roc_mean_comparison.png, confusion_*.png")
    return 0 if max_diff <= TOLERANCIA else 1


if __name__ == "__main__":
    raise SystemExit(main())
