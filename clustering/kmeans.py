from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


BASE_DIR = Path(__file__).resolve().parent

ARQ_EMBEDDINGS = BASE_DIR / "dados" / "embeddings_bert_conjuntos.npy"
ARQ_DOCUMENTOS = BASE_DIR / "dados" / "documentos_bert_conjuntos.csv"

PASTA_RESULTADOS = BASE_DIR / "resultados"
PASTA_RESULTADOS.mkdir(exist_ok=True)

embeddings = np.load(ARQ_EMBEDDINGS)
documentos = pd.read_csv(ARQ_DOCUMENTOS)

print("Embeddings:", embeddings.shape)
print("Documentos:", documentos.shape)

if len(embeddings) != len(documentos):
    raise ValueError(
        "Quantidade de embeddings diferente "
        "da quantidade de documentos."
    )

valores_k = [80, 130, 150]


for k in valores_k:

    print()
    print("=" * 60)
    print(f"Executando K-Means com k={k}")
    print("=" * 60)

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = modelo.fit_predict(embeddings)

    resultado = documentos.copy()
    resultado["cluster"] = labels


    estatisticas = (
        resultado
        .groupby("cluster")
        .agg(
            total=("id", "count"),
            cnct=("tipo", lambda x: (x == "CNCT").sum()),
            cbo=("tipo", lambda x: (x == "CBO").sum())
        )
        .reset_index()
    )

    estatisticas["misto"] = (
        (estatisticas["cnct"] > 0) &
        (estatisticas["cbo"] > 0)
    )


    arquivo_clusters = (
        PASTA_RESULTADOS /
        f"clusters_kmeans_k{k}.csv"
    )

    arquivo_estatisticas = (
        PASTA_RESULTADOS /
        f"estatisticas_kmeans_k{k}.csv"
    )

    resultado.to_csv(
        arquivo_clusters,
        index=False,
        encoding="utf-8"
    )

    estatisticas.to_csv(
        arquivo_estatisticas,
        index=False,
        encoding="utf-8"
    )



    print(f"Número de clusters: {k}")

    print(
        "Menor cluster:",
        estatisticas["total"].min()
    )

    print(
        "Maior cluster:",
        estatisticas["total"].max()
    )

    print(
        "Tamanho médio:",
        round(estatisticas["total"].mean(), 2)
    )

    print(
        "Mediana:",
        estatisticas["total"].median()
    )

    print(
        "Clusters com CNCT:",
        (estatisticas["cnct"] > 0).sum()
    )

    print(
        "Clusters somente CBO:",
        (
            (estatisticas["cnct"] == 0) &
            (estatisticas["cbo"] > 0)
        ).sum()
    )

    print(
        "Clusters mistos CNCT + CBO:",
        estatisticas["misto"].sum()
    )

    print()
    print("Arquivos:")
    print(arquivo_clusters)
    print(arquivo_estatisticas)


print()
print("K-Means concluído.")