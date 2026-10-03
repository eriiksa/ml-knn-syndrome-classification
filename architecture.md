# Decisões técnicas

## Estrutura

```
ml_test/
├── data/mini_gm_public_v0.1.p   # dataset (não versionado)
├── src/
│   ├── data_processing.py       # carga + sanidade + EDA (encadeia o resto)
│   ├── data_visualization.py    # t-SNE 2D
│   ├── data_classification.py   # KNN + validação cruzada
│   └── data_metrics.py          # AUC/F1/top-k na mão + espelho sklearn
├── outputs/                     # CSVs, figuras e npz gerados
├── architecture.md
├── requirements.txt
└── README.md
```

## Dados
- `X` (1116, 320) `float32`; `meta` com `syndrome_id`, `subject_id`, `image_id`.
- 10 síndromes, 941 sujeitos, 1 a 11 imagens por sujeito (média 1,19).
- Sem NaN, sem inf, sem vetor de norma zero, sem duplicata.
- Desbalanceamento 3,28x entre a maior e a menor síndrome.
- Norma dos vetores: min 11,5614 / média 21,8960 / max 32,6741.

## Pipeline
`data_processing.py` roda e no fim encadeia `data_visualization.py` →
`data_classification.py` → `data_metrics.py`, repassando os caminhos. O
`data_classification.py` grava as probabilidades out-of-fold em
`outputs/oof_scores.npz`; o `data_metrics.py` lê esse arquivo e não re-treina
nada. Cada script também roda sozinho.

## Validação cruzada
- `StratifiedGroupKFold(n_splits=10, groups=meta["subject_id"])`, com folds fixos
  reaproveitados pelas duas métricas.
- O mesmo sujeito tem várias imagens; um split por imagem vazaria treino↔teste e
  inflaria as métricas. Agrupar por sujeito evita isso.
- Resultados reportados como média entre folds.

## Modelo e scores
- `KNeighborsClassifier` com `metric="cosine"` e `"euclidean"`, k de 1 a 15.
- Scores sempre de `predict_proba` (AUC e top-k precisam de probabilidade);
  `predict` só para F1 e matriz de confusão.
- k=1 entra em todas as métricas. Com um vizinho só o `predict_proba` é one-hot,
  a curva ROC colapsa num ponto e a AUC vira a acurácia balanceada — não é
  ranking. Deixei assim e anoto a limitação no relatório.

## Métricas
Implementadas em `src/data_metrics.py` (lê o npz, não re-treina):
- Na mão em numpy: AUC one-vs-rest com média macro, F1 macro pela matriz de
  confusão 10x10, top-k por `argsort`.
- scikit-learn só como espelho de conferência. Diferença máxima 2,22e-16
  (≤ 1e-6), registrada no `metrics_summary.csv`.
- Top-k segue a convenção de desempate do sklearn (`argsort` estável + reversão).
- Melhor k por F1 macro: cosine=8 (0,7596), euclidean=15 (0,7188).
- Saídas: `metrics_summary.csv`, `roc_mean_comparison.png` (as duas curvas no
  mesmo gráfico), `confusion_cosine_k8.png`, `confusion_euclidean_k15.png`.

## Reprodutibilidade
- Seeds fixas no t-SNE e nos splits.
- Scripts `.py` standalone; Jupyter é proibido pelo enunciado.
- Dependências em `requirements.txt`, tiradas dos imports reais de `src/`.
