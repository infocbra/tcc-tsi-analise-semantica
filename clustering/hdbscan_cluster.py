from pathlib import Path

import numpy as np
import pandas as pd
import umap
import hdbscan


BASE_DIR = Path(__file__).resolve().parent

ARQ_EMBEDDINGS = BASE_DIR / "dados" / "embeddings.npy"
ARQ_DOCUMENTOS = BASE_DIR / "dados" / "documentos.csv"

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




print()
print("Executando UMAP...")

redutor = umap.UMAP(
    n_components=10,
    n_neighbors=15,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

embeddings_reduzidos = redutor.fit_transform(
    embeddings
)

print(
    "Shape após UMAP:",
    embeddings_reduzidos.shape
)


print()
print("Executando HDBSCAN...")

modelo = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

labels = modelo.fit_predict(
    embeddings_reduzidos
)

resultado = documentos.copy()
resultado["cluster"] = labels

# força/probabilidade de pertencimento
resultado["probabilidade_cluster"] = (
    modelo.probabilities_
)
ruido = (resultado["cluster"] == -1)

total_ruido = ruido.sum()

clusters_validos = resultado[
    resultado["cluster"] != -1
]

numero_clusters = (
    clusters_validos["cluster"].nunique()
)

estatisticas = (
    clusters_validos
    .groupby("cluster")
    .agg(
        total=("id", "count"),
        cnct=("tipo", lambda x: (x == "CNCT").sum()),
        cbo=("tipo", lambda x: (x == "CBO").sum()),
        probabilidade_media=(
            "probabilidade_cluster",
            "mean"
        )
    )
    .reset_index()
)

estatisticas["misto"] = (
    (estatisticas["cnct"] > 0)
    &
    (estatisticas["cbo"] > 0)
)


arquivo_resultado = (
    PASTA_RESULTADOS /
    "clusters_hdbscan.csv"
)

arquivo_estatisticas = (
    PASTA_RESULTADOS /
    "estatisticas_hdbscan.csv"
)

arquivo_umap = (
    PASTA_RESULTADOS /
    "embeddings_umap.npy"
)

resultado.to_csv(
    arquivo_resultado,
    index=False,
    encoding="utf-8"
)

estatisticas.to_csv(
    arquivo_estatisticas,
    index=False,
    encoding="utf-8"
)

np.save(
    arquivo_umap,
    embeddings_reduzidos
)


print()
print("=" * 60)
print("RESULTADO HDBSCAN")
print("=" * 60)

print(
    "Clusters encontrados:",
    numero_clusters
)

print(
    "Documentos classificados como ruído:",
    total_ruido
)

print(
    "Percentual de ruído:",
    f"{total_ruido / len(resultado) * 100:.2f}%"
)

if numero_clusters > 0:

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
        round(
            estatisticas["total"].mean(),
            2
        )
    )

    print(
        "Clusters com CNCT:",
        (estatisticas["cnct"] > 0).sum()
    )

    print(
        "Clusters somente CBO:",
        (
            (estatisticas["cnct"] == 0)
            &
            (estatisticas["cbo"] > 0)
        ).sum()
    )

    print(
        "Clusters mistos CNCT + CBO:",
        estatisticas["misto"].sum()
    )


print()
print("Arquivos gerados:")
print(arquivo_resultado)
print(arquivo_estatisticas)
print(arquivo_umap)