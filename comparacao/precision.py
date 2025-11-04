import pandas as pd
import re
from rapidfuzz import fuzz

ARQ_CNCT = "catalogo_cnct.csv"
ARQ_MODELO = "resultado_tfidf.csv"

cnct = pd.read_csv(ARQ_CNCT, sep=",", encoding="latin1", quotechar='"', on_bad_lines="skip")
cnct = cnct.applymap(lambda x: x.encode('latin1').decode('utf-8') if isinstance(x, str) else x)
modelo = pd.read_csv(ARQ_MODELO, sep=";", encoding="utf-8")

def limpar_texto(txt):
    if pd.isna(txt):
        return ""
    txt = str(txt).lower()
    txt = re.sub(r"[^a-zà-ú0-9\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def extrair_ocupacoes(valor):
    if pd.isna(valor):
        return []
    txt = re.sub(r"\d{4,}-\d{2}\s*-\s*", "", str(valor))
    splits = re.split(
        r"(?=\b(técnico|gerente|supervisor|auxiliar|operador|analista|mecânico|inspetor|agente|desenhista|enfermeiro)\b)",
        txt,
        flags=re.IGNORECASE,
    )
    occ_list = [limpar_texto(o) for o in splits if len(o.strip().split()) > 1]
    return occ_list

cnct["Ocupações CNCT"] = cnct["OcupaÃ§Ãµes CBO Associadas"].apply(extrair_ocupacoes)
gabarito = cnct.set_index("DenominaÃ§Ã£o do Curso")["Ocupações CNCT"].to_dict()

modelo["Profissão limpa"] = modelo["Profissão"].apply(limpar_texto)
modelo_sorted = modelo.sort_values(by=["Curso", "Similaridade_TFIDF"], ascending=[True, False])

def precision_at_k_semantica(df_sorted, k, limiar=75):
    precisions = []
    curso_precisao = []
    for curso, group in df_sorted.groupby("Curso"):
        topk = group.head(k)["Profissão limpa"].tolist()
        occ_true_list = gabarito.get(curso, [])
        if not occ_true_list:
            continue
        occ_true_list = [limpar_texto(o) for o in occ_true_list if len(o.split()) > 1]
        occ_true_set = set(occ_true_list)
        acertos = 0
        matched_preds = []
        for pred in topk:
            pred_clean = limpar_texto(pred)
            scores = [fuzz.token_set_ratio(pred_clean, occ_true) for occ_true in occ_true_set]
            best_score = max(scores) if scores else 0
            if best_score >= limiar:
                acertos += 1
                matched_preds.append((pred, best_score))
        precisions.append(acertos / k)
        curso_precisao.append({
            "Curso": curso,
            f"Precision@{k}": acertos / k,
            "Ocupações CNCT": list(occ_true_set),
            "TopK_pred": topk,
            "Matches": matched_preds
        })
    media = sum(precisions) / len(precisions) if precisions else 0
    return media, pd.DataFrame(curso_precisao)

p1, df_p1 = precision_at_k_semantica(modelo_sorted, 1)
p3, df_p3 = precision_at_k_semantica(modelo_sorted, 3)
p5, df_p5 = precision_at_k_semantica(modelo_sorted, 5)

df_final = df_p1.merge(df_p3[["Curso", "Precision@3"]], on="Curso", how="left")
df_final = df_final.merge(df_p5[["Curso", "Precision@5"]], on="Curso", how="left")

df_final.to_csv("precision_semantica_por_curso.csv", sep=";", encoding="utf-8", index=False)
