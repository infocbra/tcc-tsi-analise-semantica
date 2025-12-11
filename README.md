# Análise Semântica de Correspondência entre CNCT e CBO

## Visão Geral do Projeto

Este repositório contém o código utilizado no Trabalho de Conclusão de Curso **“Análise Semântica de Correspondência entre Perfis Profissionais do Catálogo Nacional de Cursos Técnicos (CNCT) e Ocupações no Mundo do Trabalho (CBO)”**.

O objetivo central é desenvolver uma metodologia baseada em **Processamento de Linguagem Natural (PLN)** para correlacionar, de forma automática, escalável e sem vieses, os perfis profissionais do CNCT com as ocupações previstas na CBO.

As abordagens utilizadas na análise foram:

- TF-IDF (linha de base)
- SBERT — modelo *paraphrase-multilingual-mpnet-base-v2*

Os resultados produzidos incluem:

- Cálculo de similaridade semântica entre cursos e ocupações
- Rankings das ocupações mais próximas
- Avaliação quantitativa por meio da métrica Precision@k (k = 1, 3 e 5)
- Comparação de desempenho entre TF-IDF e SBERT

---

## Instalação e Uso

### 1. Clone o repositório

```bash
git clone https://github.com/infocbra/tcc-tsi-analise-semantica
cd tcc-tsi-analise-semantica
```
### 2. Crie o ambiente virtual

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```
### 3. Instale as dependências

```bash
pip install -r requirements.txt
```
---

## Execução do Código

### 1. Vetorização e cálculo de similaridades
```bash
python versao3.py
```
O script executa as seguintes etapas:

- Carregamento das bases CNCT e CBO
- Normalização, limpeza e concatenação dos textos
- Geração de vetores usando TF-IDF e SBERT
- Cálculo da similaridade do cosseno
- Exportação das ocupações mais similares por curso
- Geração do arquivo resultado_correlacao.csv

---

### 2. Como Calcular o Precision@k

```bash
python comparativo_precision.py
```
O script gera:

- Precision@1
- Precision@3
- Precision@5
- Comparativo entre TF-IDF e SBERT
- Arquivo comparativo_precision_final.csv

---

## Objetivo da Pesquisa

Este repositório foi desenvolvido para:

- Avaliar o alinhamento entre cursos técnicos e ocupações
- Identificar lacunas e possíveis atualizações no CNCT e na CBO
- Fornecer uma metodologia transparente e replicável
- Apoiar políticas públicas e análises de empregabilidade
- Modernizar o processo de correspondência entre formação e mercado de trabalho


## Autores

João Pedro Nunes Ramos\
Érika Campos Cassimiro
