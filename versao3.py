#Correlação entre Cursos Técnicos (CNCT) e Ocupações (CBO)

import pandas as pd
import torch
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

#Baixa recursos do NLTK se necessário
nltk.download('stopwords')
nltk.download('punkt')

#Carrega stopwords em português
stopwords_pt = set(stopwords.words('portuguese'))

#Carrega modelo spaCy 
try:
    nlp = spacy.load('pt_core_news_sm')
except OSError:
    from spacy.cli import download
    download('pt_core_news_sm')
    nlp = spacy.load('pt_core_news_sm')

print("✅ Recursos carregados com sucesso!\n")

#Função de pré-processamento com lematização
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

#Leitura dos dados

df_cursos = pd.read_csv("catalogo_cnct.csv", sep=";", encoding="latin-1")
df_profissoes = pd.read_csv("cbo_enriquecido.csv", sep=",", encoding="utf-8")

#Renomeia colunas para garantir compatibilidade
df_profissoes.rename(columns={
    "TITULO": "Nome Profissão",
    "TEXTO_ENRIQUECIDO": "Texto Enriquecido"
}, inplace=True)

print("✅ Dados carregados:")
print(f"- {len(df_cursos)} cursos encontrados")
print(f"- {len(df_profissoes)} ocupações encontradas\n")

#Pré-processamento dos textos

print("🔄 Pré-processando textos...")
df_cursos["texto_limpo"] = df_cursos["Denominação do Curso"].astype(str).apply(preprocessar_texto)
df_profissoes["texto_limpo"] = df_profissoes["Texto Enriquecido"].astype(str).apply(preprocessar_texto)
print("✅ Textos pré-processados.\n")

#Embeddings com modelo BERT

print("🧠 Gerando embeddings...")
modelo = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings_cursos = modelo.encode(df_cursos["texto_limpo"].tolist(), convert_to_tensor=True)
embeddings_prof = modelo.encode(df_profissoes["texto_limpo"].tolist(), convert_to_tensor=True)
print("✅ Embeddings gerados.\n")

#Similaridade e ranking final

print("📊 Calculando similaridades...")
similaridade_cos = util.cos_sim(embeddings_cursos, embeddings_prof)

resultados = []
for i, curso in enumerate(df_cursos["Denominação do Curso"].tolist()):
    top_results = torch.topk(similaridade_cos[i], k=5)
    for score, idx in zip(top_results[0], top_results[1]):
        if score.item() > 0.55:  # limiar mínimo
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

#Salvando resultados

df_resultado.to_csv("resultado_correlacao.csv", sep=";", index=False, encoding="utf-8")
print("✅ Arquivo 'resultado_correlacao.csv' salvo com sucesso!\n")

#Exibe as 10 primeiras linhas
print(df_resultado.head(10))
