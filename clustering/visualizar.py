from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap


BASE_DIR = Path(__file__).resolve().parent

ARQ_EMBEDDINGS = (
    BASE_DIR / "dados" / "embeddings_bert_conjuntos.npy"
)

ARQ_CLUSTERS = (
    BASE_DIR /
    "resultados" /
    "clusters_kmeans_k150.csv"
)

PASTA_RESULTADOS = (
    BASE_DIR / "resultados"
)


embeddings = np.load(
    ARQ_EMBEDDINGS
)

documentos = pd.read_csv(
    ARQ_CLUSTERS
)

print(
    "Embeddings:",
    embeddings.shape
)

print(
    "Documentos:",
    documentos.shape
)



print()
print("Reduzindo embeddings para 2D...")

redutor = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

embeddings_2d = redutor.fit_transform(
    embeddings
)

documentos["x"] = embeddings_2d[:, 0]
documentos["y"] = embeddings_2d[:, 1]



arquivo_coordenadas = (
    PASTA_RESULTADOS /
    "coordenadas_umap_2d.csv"
)

documentos.to_csv(
    arquivo_coordenadas,
    index=False,
    encoding="utf-8"
)


plt.figure(
    figsize=(14, 10)
)

plt.scatter(
    documentos["x"],
    documentos["y"],
    c=documentos["cluster"],
    cmap="tab20",
    s=12,
    alpha=0.6
)

plt.title(
    "Agrupamentos semânticos CNCT–CBO\n"
    "Sentence-BERT + K-Means + UMAP"
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.tight_layout()

arquivo_clusters = (
    PASTA_RESULTADOS /
    "visualizacao_clusters_kmeans.png"
)

plt.savefig(
    arquivo_clusters,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(
    figsize=(14, 10)
)

cbo = documentos[
    documentos["tipo"] == "CBO"
]

cnct = documentos[
    documentos["tipo"] == "CNCT"
]


plt.scatter(
    cbo["x"],
    cbo["y"],
    s=10,
    alpha=0.35,
    label="CBO"
)

plt.scatter(
    cnct["x"],
    cnct["y"],
    s=35,
    marker="^",
    alpha=0.9,
    label="CNCT"
)

plt.title(
    "Distribuição semântica dos documentos CNCT e CBO"
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.legend()

plt.tight_layout()

arquivo_fontes = (
    PASTA_RESULTADOS /
    "visualizacao_cnct_cbo.png"
)

plt.savefig(
    arquivo_fontes,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("Visualizações geradas:")
print(arquivo_clusters)
print(arquivo_fontes)
print(arquivo_coordenadas)