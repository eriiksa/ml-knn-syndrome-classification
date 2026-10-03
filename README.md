# Teste ML — Apollo Solutions

> You are being hired by a fictional biotech company specializing in genetic research. The task involves analyzing embeddings derived from images to classify genetic syndromes. These embeddings are outputs from a pre-trained classification model. The company wants to improve its understanding of the data distribution and enhance the classification accuracy of genetic syndromes based on these embeddings.

Classificação de `syndrome_id` a partir de embeddings de imagem (320 dimensões).

Dataset no formato `syndrome_id -> subject_id -> image_id -> vetor(320)`. Achatando, cada imagem vira uma linha de `X` (1116 x 320) e os rótulos ficam no `meta`. São 10 síndromes, 941 sujeitos e desbalanceamento de 3,28x entre a maior e a menor.

## Como executar

```powershell
pip install -r requirements.txt
python src\data_processing.py
```

O dataset já vem no repo, em `data\mini_gm_public_v0.1.p`.

Rodar o `data_processing.py` já encadeia o resto na ordem certa (t-SNE → KNN → métricas), então esse comando é o suficiente. Cada script também roda sozinho se precisar (`python src\data_classification.py`), mas a ordem importa: o `data_classification.py` gera o `outputs/oof_scores.npz` que o `data_metrics.py` lê.

Tudo cai em `outputs/`: `meta.csv` e `eda_resumo.txt` (dia 1), `tsne.png` (dia 2), `oof_scores.npz` (dia 3) e a tabela + figuras (dia 4).

## Os scripts

**data_processing.py** — lê o pickle, desmonta a hierarquia em `X` e `meta`, roda as checagens (320 dimensões, sem NaN, sem inf, sem vetor de norma zero, sem duplicata) e faz a EDA.

**data_visualization.py** — t-SNE 2D colorido por síndrome, `perplexity=30`, `random_state=42`, `init="pca"`. Rodei sobre todos os dados porque o item do teste só pede pra visualizar os grupos; não é entrada de modelo.

**data_classification.py** — KNN com cosine e euclidean, k de 1 a 15, com 10-fold. Usei `StratifiedGroupKFold(groups=subject_id)`, não KFold comum: o mesmo sujeito tem várias imagens, e splitando por imagem elas caem em treino e teste juntas, inflando as métricas. Agrupando por sujeito isso não acontece. Os folds são fixos e as duas métricas usam os mesmos, então a comparação é justa. Salvo as probabilidades out-of-fold porque AUC e top-k precisam de `predict_proba`, não do rótulo do `predict`.

**data_metrics.py** — AUC, F1 macro e top-k (1/3/5) calculados na mão em numpy, conferidos contra o scikit-learn (diferença máxima 2e-16, tolerância 1e-6). Gera a tabela, a ROC média comparando cosine x euclidean no melhor k de cada uma e as duas matrizes de confusão.

No fim: cosine ganha da euclidean em AUC, F1 e top-k em todos os k. Melhor k por F1 macro: cosine=8 (F1 0,7596 / AUC 0,9467) e euclidean=15 (F1 0,7188 / AUC 0,9426).

## De onde tirei as aulas

A ordem bate com o que fui implementando:

- **Training a machine learning model with scikit-learn** (Data School) — <https://www.youtube.com/watch?v=RlQuVL6-qe8>
  Daqui saiu o fluxo básico do sklearn (`fit`/`predict`, `KNeighborsClassifier`) que usei no `data_classification.py`.

- **K nearest neighbors classification with python code** (codebasics) — <https://www.youtube.com/watch?v=CQveSaMyEwM>
  Foi onde vi o KNN na prática e a ideia de varrer o k pra achar o melhor.

- **How to find the best model parameters in scikit-learn** (Data School) — <https://www.youtube.com/watch?v=Gol_qOgRqfA>
  Usei pra montar a busca do melhor k (1 a 15) em vez de chutar um valor.

- **Selecting the best model in scikit-learn using cross-validation** (Data School) — <https://www.youtube.com/watch?v=6dbrR-WymjI>
  Base do 10-fold do `data_classification.py`. (fallback: <https://www.youtube.com/watch?v=gJo0uNL-5Qw>)

- **ROC Curves and Area Under the Curve (AUC) Explained** (Data School) — <https://www.youtube.com/watch?v=OAl6eAyP-yo>
  Base pra entender e implementar ROC/AUC no `data_metrics.py`.

- **Precision, Recall, F1 score, True Positive** (codebasics) — <https://www.youtube.com/watch?v=2osIZ-dSPGE>
  Base pro F1 macro e pra ler a matriz de confusão.

t-SNE: não achei aula específica que desse pra confirmar, então essa parte saiu da documentação do scikit-learn.
