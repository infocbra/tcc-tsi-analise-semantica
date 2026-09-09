from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

from sklearn.metrics.pairwise import cosine_similarity

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

PASTA_FIGURAS = (
    PASTA_RESULTADOS / "figuras"
)

PASTA_FIGURAS.mkdir(
    exist_ok=True
)


def normalizar_texto(texto):
    """
    Remove acentos, converte para minúsculas
    e remove espaços excedentes.

    Usado apenas para facilitar a busca
    pelos nomes dos cursos.
    """

    texto = str(texto)

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    return texto.lower().strip()


def nome_arquivo(texto):
    """
    Converte um texto em um nome seguro
    para utilização como nome de arquivo.
    """

    texto = normalizar_texto(texto)

    texto = re.sub(
        r"[^a-z0-9]+",
        "_",
        texto
    )

    return texto.strip("_")


print("Carregando dados...")

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


if len(embeddings) != len(documentos):
    raise ValueError(
        "Quantidade de embeddings diferente "
        "da quantidade de documentos."
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



print()
print(
    "Gerando visualização global CNCT × CBO..."
)

cbo = documentos[
    documentos["tipo"] == "CBO"
]

cnct = documentos[
    documentos["tipo"] == "CNCT"
]


plt.figure(
    figsize=(14, 10)
)


# CBOs mais discretas
plt.scatter(
    cbo["x"],
    cbo["y"],
    s=10,
    alpha=0.25,
    label="CBO"
)


# CNCT em destaque
plt.scatter(
    cnct["x"],
    cnct["y"],
    s=45,
    marker="^",
    alpha=0.9,
    label="CNCT"
)


plt.title(
    "Distribuição semântica dos documentos CNCT e CBO\n"
    "Sentence-BERT + UMAP"
)

plt.xlabel(
    "UMAP 1"
)

plt.ylabel(
    "UMAP 2"
)

plt.legend()

plt.tight_layout()


arquivo_global = (
    PASTA_FIGURAS /
    "visualizacao_cnct_cbo.png"
)

plt.savefig(
    arquivo_global,
    dpi=300,
    bbox_inches="tight"
)

plt.close()



print(
    "Gerando visualização global dos clusters..."
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

plt.xlabel(
    "UMAP 1"
)

plt.ylabel(
    "UMAP 2"
)

plt.tight_layout()


arquivo_clusters = (
    PASTA_FIGURAS /
    "visualizacao_clusters_kmeans.png"
)

plt.savefig(
    arquivo_clusters,
    dpi=300,
    bbox_inches="tight"
)

plt.close()



def visualizar_curso(
    nome_curso,
    top_n=10
):
    """
    Localiza um curso CNCT e seleciona as CBOs
    semanticamente mais próximas utilizando
    similaridade do cosseno calculada sobre
    os embeddings originais de 768 dimensões.

    A posição dos pontos no gráfico é obtida
    pelo UMAP 2D.

    A cor das CBOs representa o cluster K-Means
    ao qual cada ocupação pertence.
    """

    print()
    print(
        "=" * 70
    )

    print(
        f"Gerando estudo de caso: {nome_curso}"
    )

    print(
        "=" * 70
    )


    nome_normalizado = normalizar_texto(
        nome_curso
    )

    nomes_normalizados = (
        documentos["nome"]
        .fillna("")
        .apply(normalizar_texto)
    )

    mascara_curso = (
        (documentos["tipo"] == "CNCT")
        &
        (
            nomes_normalizados
            == nome_normalizado
        )
    )

    indices_curso = documentos.index[
        mascara_curso
    ].tolist()


    if not indices_curso:

        print(
            f"Curso não encontrado: {nome_curso}"
        )

        print()
        print(
            "Tentando localizar correspondências parciais..."
        )

        mascara_parcial = (
            (documentos["tipo"] == "CNCT")
            &
            nomes_normalizados.str.contains(
                nome_normalizado,
                regex=False
            )
        )

        candidatos = documentos[
            mascara_parcial
        ]

        if not candidatos.empty:

            print()
            print(
                "Possíveis correspondências:"
            )

            for nome in candidatos["nome"]:

                print(
                    " -",
                    nome
                )

        return


    indice_curso = indices_curso[0]

    curso = documentos.loc[
        indice_curso
    ]


    print(
        "Curso encontrado:",
        curso["nome"]
    )

    print(
        "Cluster K-Means do curso:",
        int(curso["cluster"])
    )


    mascara_cbo = (
        documentos["tipo"] == "CBO"
    )

    indices_cbo = documentos.index[
        mascara_cbo
    ].to_numpy()


    embeddings_cbo = embeddings[
        indices_cbo
    ]


    embedding_curso = embeddings[
        indice_curso
    ].reshape(
        1,
        -1
    )


    similaridades = cosine_similarity(
        embedding_curso,
        embeddings_cbo
    )[0]


    # ordenar da maior para menor similaridade
    top_indices_locais = np.argsort(
        similaridades
    )[::-1][
        :top_n
    ]


    # converter índices relativos ao subconjunto CBO
    # em índices reais do dataframe
    top_indices_reais = indices_cbo[
        top_indices_locais
    ]


    cbo_proximas = documentos.loc[
        top_indices_reais
    ].copy()


    cbo_proximas[
        "similaridade"
    ] = similaridades[
        top_indices_locais
    ]


    cbo_proximas[
        "mesmo_cluster_curso"
    ] = (
        cbo_proximas["cluster"]
        == curso["cluster"]
    )


    print()
    print(
        f"Top {top_n} CBOs mais próximas:"
    )

    print()


    for posicao, (
        _,
        linha
    ) in enumerate(
        cbo_proximas.iterrows(),
        start=1
    ):

        if linha[
            "mesmo_cluster_curso"
        ]:

            indicador = (
                "MESMO CLUSTER"
            )

        else:

            indicador = (
                "cluster diferente"
            )


        print(
            f"{posicao:2d}. "
            f"{linha['nome']} | "
            f"sim={linha['similaridade']:.4f} | "
            f"cluster={int(linha['cluster'])} | "
            f"{indicador}"
        )



    plt.figure(
        figsize=(14, 10)
    )


    plt.scatter(
        cbo_proximas["x"],
        cbo_proximas["y"],
        c=cbo_proximas["cluster"],
        cmap="tab20",
        s=110,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3,
        label="CBOs semanticamente próximas"
    )




    plt.scatter(
        [curso["x"]],
        [curso["y"]],
        s=400,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label=curso["nome"],
        zorder=10
    )




    for posicao, (
        _,
        linha
    ) in enumerate(
        cbo_proximas.iterrows(),
        start=1
    ):

        texto = (
            f"{posicao}. "
            f"{linha['nome']}\n"
            f"sim={linha['similaridade']:.3f}"
        )


        plt.annotate(
            texto,
            (
                linha["x"],
                linha["y"]
            ),
            xytext=(
                7,
                7
            ),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9
        )



    plt.annotate(
        curso["nome"],
        (
            curso["x"],
            curso["y"]
        ),
        xytext=(
            10,
            12
        ),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold"
    )



    plt.title(
        f"{curso['nome']} e suas {top_n} CBOs "
        f"semanticamente mais próximas\n"
        "Sentence-BERT + Similaridade do Cosseno "
        "+ K-Means + UMAP"
    )

    plt.xlabel(
        "UMAP 1"
    )

    plt.ylabel(
        "UMAP 2"
    )

    plt.legend()

    plt.tight_layout()


    nome_saida = (
        "estudo_caso_similaridade_"
        + nome_arquivo(
            curso["nome"]
        )
        + ".png"
    )


    arquivo_saida = (
        PASTA_FIGURAS /
        nome_saida
    )


    plt.savefig(
        arquivo_saida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



    arquivo_csv = (
        PASTA_RESULTADOS /
        (
            "top_cbo_"
            + nome_arquivo(
                curso["nome"]
            )
            + ".csv"
        )
    )


    resultado_csv = cbo_proximas[
        [
            "id",
            "nome",
            "cluster",
            "x",
            "y",
            "similaridade",
            "mesmo_cluster_curso"
        ]
    ].copy()


    resultado_csv.insert(
        0,
        "curso",
        curso["nome"]
    )


    resultado_csv.insert(
        1,
        "cluster_curso",
        int(
            curso["cluster"]
        )
    )


    resultado_csv.insert(
        2,
        "posicao",
        range(
            1,
            len(resultado_csv) + 1
        )
    )


    resultado_csv.to_csv(
        arquivo_csv,
        index=False,
        encoding="utf-8"
    )


    quantidade_mesmo_cluster = (
        cbo_proximas[
            "mesmo_cluster_curso"
        ].sum()
    )


    print()
    print(
        "CBOs do Top-N no mesmo cluster do curso:",
        f"{quantidade_mesmo_cluster}/{top_n}"
    )

    print()
    print(
        "Figura:",
        arquivo_saida
    )

    print(
        "Ranking:",
        arquivo_csv
    )



visualizar_curso(
    "Técnico em Agrimensura",
    top_n=5
)

visualizar_curso(
    "Técnico Aeroportuário",
    top_n=5
)



print()
print(
    "=" * 70
)

print(
    "VISUALIZAÇÕES CONCLUÍDAS"
)

print(
    "=" * 70
)

print()
print(
    "Arquivos globais:"
)

print(
    arquivo_global
)

print(
    arquivo_clusters
)

print(
    arquivo_coordenadas
)

print()
print(
    "Figuras dos estudos de caso disponíveis em:"
)

print(
    PASTA_FIGURAS
)