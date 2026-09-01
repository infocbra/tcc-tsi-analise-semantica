import pandas as pd
import torch
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RAIZ_PROJETO = BASE_DIR.parent

nltk.download('stopwords')
nltk.download('punkt')

stopwords_pt = set(stopwords.words('portuguese'))

try:
    nlp = spacy.load('pt_core_news_sm')
except OSError:
    from spacy.cli import download
    download('pt_core_news_sm')
    nlp = spacy.load('pt_core_news_sm')

def preprocessar_texto(texto):
    if pd.isna(texto):
        return ""

    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúãõâêîôûç\s]', '', texto)

    doc = nlp(texto)
    tokens = [
        token.lemma_ for token in doc
        if token.text not in stopwords_pt and len(token.text) > 2 and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)


df_cursos = pd.read_csv(BASE_DIR / "catalogo_cnct.csv", sep=";", encoding="latin-1")
df_profissoes = pd.read_csv(BASE_DIR / "cbo_enriquecido.csv", sep=",", encoding="utf-8")

df_profissoes.rename(columns={
    "TITULO": "Nome Profissão",
    "TEXTO_ENRIQUECIDO": "Texto Enriquecido"
}, inplace=True)

colunas_para_concatenar = [
    "Denominação do Curso",
    "Perfil Profissional de Conclusão",
    "Campo de Atuação",
    "Itinerários Formativos",
    "Ocupações CBO Associadas"
]

for c in colunas_para_concatenar:
    if c not in df_cursos.columns:
        df_cursos[c] = ""

df_cursos["texto_enriquecido"] = df_cursos[colunas_para_concatenar].fillna("").agg(" ".join, axis=1)


df_cursos["texto_limpo"] = df_cursos["texto_enriquecido"].astype(str).apply(preprocessar_texto)
df_profissoes["texto_limpo"] = df_profissoes["Texto Enriquecido"].astype(str).apply(preprocessar_texto)


modelo = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings_cursos = modelo.encode(df_cursos["texto_limpo"].tolist(), convert_to_tensor=True)
embeddings_prof = modelo.encode(df_profissoes["texto_limpo"].tolist(), convert_to_tensor=True)


BASE_DIR = Path(__file__).resolve().parent
RAIZ_PROJETO = BASE_DIR.parent

PASTA_CLUSTERING = (
    RAIZ_PROJETO /
    "clustering" /
    "dados"
)

PASTA_CLUSTERING.mkdir(
    parents=True,
    exist_ok=True
)


# converter tensores para numpy
embeddings_cursos_np = (
    embeddings_cursos
    .detach()
    .cpu()
    .numpy()
)

embeddings_prof_np = (
    embeddings_prof
    .detach()
    .cpu()
    .numpy()
)


# salvar embeddings separadamente
np.save(
    PASTA_CLUSTERING /
    "embeddings_cursos_bert.npy",
    embeddings_cursos_np
)

np.save(
    PASTA_CLUSTERING /
    "embeddings_cbo_bert.npy",
    embeddings_prof_np
)


# salvar metadados dos cursos
df_cursos_cluster = pd.DataFrame({
    "id": [
        f"CNCT_{i}"
        for i in range(len(df_cursos))
    ],
    "tipo": "CNCT",
    "nome": df_cursos[
        "Denominação do Curso"
    ],
    "texto": df_cursos[
        "texto_limpo"
    ]
})

df_cursos_cluster.to_csv(
    PASTA_CLUSTERING /
    "documentos_cursos_bert.csv",
    index=False,
    encoding="utf-8"
)


# salvar metadados das CBOs
df_cbo_cluster = pd.DataFrame({
    "id": (
        "CBO_"
        + df_profissoes["CBO"].astype(str)
    ),
    "tipo": "CBO",
    "nome": df_profissoes[
        "Nome Profissão"
    ],
    "texto": df_profissoes[
        "texto_limpo"
    ]
})

df_cbo_cluster.to_csv(
    PASTA_CLUSTERING /
    "documentos_cbo_bert.csv",
    index=False,
    encoding="utf-8"
)


# criar também uma versão conjunta
embeddings_conjuntos = np.vstack([
    embeddings_cursos_np,
    embeddings_prof_np
])

np.save(
    PASTA_CLUSTERING /
    "embeddings_bert_conjuntos.npy",
    embeddings_conjuntos
)


documentos_conjuntos = pd.concat(
    [
        df_cursos_cluster,
        df_cbo_cluster
    ],
    ignore_index=True
)

documentos_conjuntos.to_csv(
    PASTA_CLUSTERING /
    "documentos_bert_conjuntos.csv",
    index=False,
    encoding="utf-8"
)


print()
print("Embeddings salvos para clustering:")
print(
    "Cursos:",
    embeddings_cursos_np.shape
)
print(
    "CBO:",
    embeddings_prof_np.shape
)
print(
    "Conjunto:",
    embeddings_conjuntos.shape
)


similaridade_cos = util.cos_sim(embeddings_cursos, embeddings_prof)

resultados = []
for i, curso in enumerate(df_cursos["Denominação do Curso"].tolist()):
    top_results = torch.topk(similaridade_cos[i], k=5)
    for score, idx in zip(top_results[0], top_results[1]):
        if score.item() > 0.55: 
            resultados.append({
                "Curso": curso,
                "CBO": df_profissoes["CBO"].iloc[idx.item()],
                "Profissão": df_profissoes["Nome Profissão"].iloc[idx.item()],
                "Similaridade": round(score.item(), 4)
            })

df_resultado = pd.DataFrame(resultados).sort_values(
    by=["Curso", "Similaridade"],
    ascending=[True, False]
)

df_resultado.to_csv(BASE_DIR / "resultado_correlacao1.csv", sep=";", index=False, encoding="utf-8")
print(df_resultado.head(10))
