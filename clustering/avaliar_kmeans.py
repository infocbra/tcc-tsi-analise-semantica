from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parent

ARQ_EMBEDDINGS = BASE_DIR / "dados" / "embeddings_bert_conjuntos.npy"

PASTA_RESULTADOS = BASE_DIR / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)

embeddings = np.load(ARQ_EMBEDDINGS)

print("Shape dos embeddings:", embeddings.shape)
print()

resultados = []

valores_k = range(5, 151, 5)

for k in valores_k:

    print(f"Testando k={k}...")

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = modelo.fit_predict(embeddings)

    silhouette = silhouette_score(
        embeddings,
        labels,
        metric="cosine"
    )

    inercia = modelo.inertia_

    resultados.append({
        "k": k,
        "silhouette": silhouette,
        "inercia": inercia
    })

    print(
        f"k={k:2d} | "
        f"silhouette={silhouette:.4f} | "
        f"inércia={inercia:.2f}"
    )

    print()


df_resultados = pd.DataFrame(resultados)

arquivo_saida = (
    PASTA_RESULTADOS /
    "avaliacao_kmeans.csv"
)

df_resultados.to_csv(
    arquivo_saida,
    index=False,
    encoding="utf-8"
)



melhor = df_resultados.loc[
    df_resultados["silhouette"].idxmax()
]

print("=" * 50)
print("MELHOR RESULTADO")
print("=" * 50)

print(
    f"K: {int(melhor['k'])}"
)

print(
    f"Silhouette: "
    f"{melhor['silhouette']:.4f}"
)

print(
    f"Inércia: "
    f"{melhor['inercia']:.2f}"
)

print()
print(
    "Resultados salvos em:",
    arquivo_saida
)