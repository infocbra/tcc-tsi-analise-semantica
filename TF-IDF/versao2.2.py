import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')

stopwords_pt = list(set(stopwords.words('portuguese')))

try:
    nlp = spacy.load('pt_core_news_sm')
except OSError:
    from spacy.cli import download
    download('pt_core_news_sm')
    nlp = spacy.load('pt_core_news_sm')


def preprocessar_texto(texto):
    """Normaliza, lematiza e remove stopwords."""
    if pd.isna(texto):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúãõâêîôûç\s]', ' ', texto)
    doc = nlp(texto)
    tokens = [
        token.lemma_ for token in doc
        if token.text not in stopwords_pt and len(token.text) > 2 and token.is_alpha
    ]
    return " ".join(tokens)

df_cursos = pd.read_csv("catalogo_cnct.csv", sep=";", encoding="latin-1")
df_profissoes = pd.read_csv("cbo_enriquecido.csv", sep=",", encoding="utf-8")

df_profissoes.rename(columns={
    "TITULO": "Nome Profissão",
    "TEXTO_ENRIQUECIDO": "Texto Enriquecido"
}, inplace=True)


df_cursos["texto_limpo"] = df_cursos["Perfil Profissional de Conclusão"].astype(str).apply(preprocessar_texto)
df_profissoes["texto_limpo"] = df_profissoes["Texto Enriquecido"].astype(str).apply(preprocessar_texto)

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    stop_words=stopwords_pt  
)

corpus = df_cursos["texto_limpo"].tolist() + df_profissoes["texto_limpo"].tolist()
tfidf_matrix = vectorizer.fit_transform(corpus)
tfidf_cursos = tfidf_matrix[:len(df_cursos)]
tfidf_profissoes = tfidf_matrix[len(df_cursos):]

sim_matrix = cosine_similarity(tfidf_cursos, tfidf_profissoes)

resultados = []
for i, curso in enumerate(df_cursos["Denominação do Curso"].tolist()):
    top_idx = sim_matrix[i].argsort()[-5:][::-1]
    for j in top_idx:
        resultados.append({
            "Curso": curso,
            "CBO": df_profissoes["CBO"].iloc[j],
            "Profissão": df_profissoes["Nome Profissão"].iloc[j],
            "Similaridade_TFIDF": round(float(sim_matrix[i, j]), 4)
        })

df_resultado = pd.DataFrame(resultados).sort_values(
    by=["Curso", "Similaridade_TFIDF"],
    ascending=[True, False]
)

df_resultado.to_csv("resultado_tfidf.csv", sep=";", index=False, encoding="utf-8")