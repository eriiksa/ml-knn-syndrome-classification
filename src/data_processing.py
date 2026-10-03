"""Dia 1: carga do pickle, checagens de sanidade e EDA basica do dataset."""

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_DIM = 320


def load_dataset(path: Path) -> dict:
    """Le o pickle; alguns arquivos antigos foram gravados em latin1."""
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except UnicodeDecodeError:
        with open(path, "rb") as fh:
            return pickle.load(fh, encoding="latin1")


def flatten_dataset(data: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Achata syndrome -> subject -> image -> vetor(320) em (X, meta)."""
    vectors: list[np.ndarray] = []
    rows: list[tuple[str, str, str]] = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                vectors.append(np.asarray(embedding, dtype=np.float32).ravel())
                rows.append((str(syndrome_id), str(subject_id), str(image_id)))
    if not vectors:
        raise ValueError("Nenhum vetor encontrado: estrutura inesperada.")
    meta = pd.DataFrame(rows, columns=["syndrome_id", "subject_id", "image_id"])
    return np.vstack(vectors), meta


def sanity_checks(X: np.ndarray, meta: pd.DataFrame) -> list[str]:
    """Checa forma e conteudo de X; devolve a lista de problemas encontrados."""
    norms = np.linalg.norm(X, axis=1)
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    n_zero = int((norms == 0).sum())
    n_dup = int(meta.duplicated(["subject_id", "image_id"]).sum())
    n_multi = int((meta.groupby("subject_id")["syndrome_id"].nunique() > 1).sum())

    problems: list[str] = []
    if X.shape[1] != EXPECTED_DIM:
        problems.append(f"Dimensao inesperada: {X.shape[1]} (esperado {EXPECTED_DIM}).")
    if X.shape[0] != len(meta):
        problems.append(f"X tem {X.shape[0]} linhas e meta tem {len(meta)}.")
    if n_nan:
        problems.append(f"{n_nan} valores NaN em X.")
    if n_inf:
        problems.append(f"{n_inf} valores infinitos em X.")
    if n_zero:
        problems.append(f"{n_zero} vetores com norma zero.")
    if n_dup:
        problems.append(f"{n_dup} pares (subject_id, image_id) duplicados.")
    if n_multi:
        problems.append(f"{n_multi} subject_id aparecem em mais de uma sindrome.")

    print(f"X.shape / dtype    : {X.shape} / {X.dtype}")
    print(f"NaN / inf / norma0 : {n_nan} / {n_inf} / {n_zero}")
    print(f"norma min/med/max  : {norms.min():.4f} / {norms.mean():.4f} / {norms.max():.4f}")
    return problems


def eda(X: np.ndarray, meta: pd.DataFrame) -> str:
    """Monta o texto da EDA (impresso e salvo em outputs/eda_resumo.txt)."""
    imgs = meta["syndrome_id"].value_counts().sort_values(ascending=False)
    suj = meta.groupby("syndrome_id")["subject_id"].nunique()
    por_suj = meta.groupby("subject_id").size()
    pct = imgs / imgs.sum() * 100

    linhas = [
        "EDA basica -- mini_gm_public_v0.1.p",
        "=" * 50,
        f"imagens            : {len(meta)}",
        f"sindromes          : {meta['syndrome_id'].nunique()}",
        f"sujeitos           : {meta['subject_id'].nunique()}",
        f"dimensao embedding : {X.shape[1]}",
        f"imagens/sujeito    : min {por_suj.min()}  max {por_suj.max()}  media {por_suj.mean():.2f}",
        f"sujeitos/sindrome  : min {suj.min()}  max {suj.max()}",
        "",
        "imagens por sindrome:",
    ]
    for sind, n in imgs.items():
        linhas.append(f"  {sind:<12} {n:>5} imagens  {pct[sind]:>5.1f}%  ({suj[sind]} sujeitos)") #type: ignore
    linhas += [
        f"  razao maior/menor: {imgs.max() / imgs.min():.2f}x",
        "",
        "imagens por sujeito:",
    ]
    for n_img, qtd in por_suj.value_counts().sort_index().items():
        linhas.append(f"  {qtd:>5} sujeitos com {n_img} imagem(ns)")
    return "\n".join(linhas)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Carga e EDA do dataset de embeddings.")
    parser.add_argument("--data", type=Path, default=here.parent / "data" / "mini_gm_public_v0.1.p")
    parser.add_argument("--out", type=Path, default=here.parent / "outputs")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERRO: arquivo nao encontrado: {args.data}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Lendo {args.data} ...")
    X, meta = flatten_dataset(load_dataset(args.data))
    print(f"OK. X={X.shape}\n")

    problemas = sanity_checks(X, meta)
    print()
    if problemas:
        for p in problemas:
            print(f"  [ATENCAO] {p}")
    else:
        print("  Sanidade OK: sem NaN, inf, norma zero ou duplicatas.")

    texto = eda(X, meta)
    print("\n" + texto)

    meta.to_csv(args.out / "meta.csv", index=False)
    (args.out / "eda_resumo.txt").write_text(texto, encoding="utf-8")
    print(f"\nSalvo: {args.out / 'meta.csv'}")
    print(f"Salvo: {args.out / 'eda_resumo.txt'}")

    if problemas:
        return 1

    # encadeia o resto do pipeline: t-SNE -> KNN -> metricas
    passos = [
        ("data_visualization.py", ["--data", str(args.data), "--out", str(args.out)]),
        ("data_classification.py", ["--data", str(args.data), "--out", str(args.out)]),
        ("data_metrics.py", ["--npz", str(args.out / "oof_scores.npz"), "--out", str(args.out)]),
    ]
    for script, extra in passos:
        print(f"\n=== {script} ===")
        r = subprocess.run([sys.executable, str(here / script), *extra])
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
