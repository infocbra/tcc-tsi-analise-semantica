from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
RAIZ_PROJETO = BASE_DIR.parent

ARQ_CNCT = RAIZ_PROJETO / "BERT" / "catalogo_cnct.csv"
ARQ_CBO = RAIZ_PROJETO / "BERT" / "cbo_enriquecido.csv"

PASTA_DADOS = BASE_DIR / "dados"
PASTA_DADOS.mkdir(exist_ok=True)


cnct = pd.read_csv(
    ARQ_CNCT,
    sep=";",
    encoding="latin-1"
)

cbo = pd.read_csv(
    ARQ_CBO,
    sep=",",
    encoding="utf-8"
)



cnct_final = pd.DataFrame({
    "id": ["CNCT_" + str(i) for i in range(len(cnct))],
    "tipo": "CNCT",
    "nome": cnct["Denominação do Curso"],
    "texto": cnct["Perfil Profissional de Conclusão"].fillna("")
})

cbo_final = pd.DataFrame({
    "id": "CBO_" + cbo["CBO"].astype(str),
    "tipo": "CBO",
    "nome": cbo["TITULO"],
    "texto": cbo["TEXTO_ENRIQUECIDO"].fillna("")
})


documentos = pd.concat(
    [cnct_final, cbo_final],
    ignore_index=True
)

# remover textos vazios
documentos = documentos[
    documentos["texto"].str.strip() != ""
].reset_index(drop=True)

print("Quantidade de documentos:", len(documentos))
print()
print(documentos["tipo"].value_counts())


modelo = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2"
)


embeddings = modelo.encode(
    documentos["texto"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)
np.save(
    PASTA_DADOS / "embeddings.npy",
    embeddings
)

documentos.to_csv(
    PASTA_DADOS / "documentos.csv",
    index=False,
    encoding="utf-8"
)


print()
print("Shape dos embeddings:", embeddings.shape)
print("Arquivos gerados:")
print(PASTA_DADOS / "embeddings.npy")
print(PASTA_DADOS / "documentos.csv")