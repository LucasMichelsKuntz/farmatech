# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href="https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Administração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# FarmTech Solutions — Fase 4: Machine Learning Aplicado à Agricultura de Precisão

## Grupo FarmTech Solutions

## 👨‍🎓 Integrantes:
- <a href="https://www.linkedin.com/in/">Lucas Michels Kuntz</a>
- <a href="https://www.linkedin.com/in/">Nome do Integrante 2</a>
- <a href="https://www.linkedin.com/in/">Nome do Integrante 3</a>
- <a href="https://www.linkedin.com/in/">Nome do Integrante 4</a>
- <a href="https://www.linkedin.com/in/">Nome do Integrante 5</a>

## 👩‍🏫 Professores:
### Tutor(a)
- <a href="https://www.linkedin.com/in/">Nome do Tutor</a>
### Coordenador(a)
- <a href="https://www.linkedin.com/in/">Nome do Coordenador</a>

---

## 📜 Descrição

A **FarmTech Solutions** é uma empresa de tecnologia agrícola que desenvolve soluções inteligentes para o agronegócio. Nesta Fase 4, o desafio proposto pela FIAP consiste em evoluir o sistema de monitoramento de sensores IoT desenvolvido nas fases anteriores, incorporando **Machine Learning supervisionado** para tomada de decisão autônoma no campo.

### Contexto e Problema

Produtores rurais enfrentam dois problemas recorrentes de alto impacto financeiro:

1. **Irrigação ineficiente**: irrigar em excesso eleva custos de energia e água e pode causar encharcamento; irrigar de menos causa stress hídrico e reduz a produtividade.
2. **Fertilização imprecisa**: a deficiência de nitrogênio (N) é a principal causa de queda de rendimento em culturas como milho, arroz e trigo, mas a superdosagem contamina o solo e o lençol freático.

O sistema de sensores IoT (simulado via Wokwi nas fases anteriores) coleta dados de **nitrogênio, fósforo, potássio, temperatura, umidade, pH e precipitação** para 22 culturas diferentes. Com 2.200 leituras reais (dataset Kaggle — *Crop Recommendation Dataset*), o desafio é treinar modelos de ML capazes de:

- **Prever a necessidade de irrigação** (regressão sobre `chuva_mm`)
- **Prever a necessidade de fertilização nitrogenada** (regressão sobre `nitrogenio`)
- **Recomendar qual cultura plantar** dado o conjunto de condições do solo e clima (classificação)

### O que foi construído

Desenvolvemos um **pipeline completo de Machine Learning** com as seguintes etapas:

**1. Ingestão de Dados**
O dataset CSV é carregado e persistido em um banco **SQLite** com schema relacional (`culturas` + `leituras_sensor`), simulando o fluxo real de dados IoT → banco de dados → modelo.

**2. Feature Engineering**
Além das 7 features originais dos sensores, criamos duas features derivadas:
- `npk_total` = N + P + K (fertilidade total do solo)
- `npk_ratio_nk` = N / (K + 1) (balanço nitrogênio-potássio)

**3. Modelos de Regressão (Irrigação e Fertilização)**
Para cada target, treinamos e comparamos três algoritmos com `Pipeline(StandardScaler + modelo)`:
| Modelo | Tipo | Hiperparâmetros |
|---|---|---|
| Regressão Linear | Linear | — |
| Ridge (L2) | Linear regularizado | alpha=1.0 |
| Random Forest | Ensemble não-linear | 150 estimadores, max_depth=10 |

Métricas calculadas: **MAE, MSE, RMSE, R²**. O modelo com maior R² é automaticamente selecionado como "melhor modelo" e salvo em `models/models.joblib`.

**4. Modelo de Classificação (Recomendação de Cultura)**
Um **Random Forest Classifier** (200 estimadores, max_depth=12) foi treinado para classificar entre 22 culturas. O modelo retorna as **Top 5 culturas recomendadas** com probabilidade de confiança para qualquer conjunto de condições.

**5. Dashboard Interativo (Streamlit)**
A interface web conta com 6 telas:

| Tela | Conteúdo |
|---|---|
| Visão Geral | Métricas do dataset, distribuição por cultura, estatísticas descritivas |
| Análise Exploratória | Histogramas, matriz de correlação, boxplots, dispersão interativa |
| Modelos de Regressão | Comparação de modelos (tabela de métricas), gráfico Real vs Previsto, importância de features |
| Previsões Interativas | Sliders para simular condições do campo; gauges de irrigação e N; análise de sensibilidade ao pH |
| Recomendações | Plano de manejo priorizado (CRITICA / ALTA / MEDIA / OK) gerado pelo modelo |
| Recomendação de Cultura | Top 5 culturas recomendadas com barra de confiança e importância de features |

### Tecnologias utilizadas
- **Python 3.11** — linguagem principal
- **Scikit-learn** — treinamento dos modelos de ML
- **Pandas / NumPy** — manipulação de dados
- **SQLite** — banco de dados relacional local
- **Streamlit** — dashboard web interativo
- **Plotly** — visualizações interativas
- **Joblib** — persistência dos modelos treinados

---

## 📁 Estrutura de pastas

```
farmtech-fase4/
│
├── assets/                  # Imagens e recursos estáticos
│   └── logo-fiap.png
│
├── dashboard/
│   └── app.py               # Aplicação Streamlit (6 telas)
│
├── data/
│   └── crop_dataset.csv     # Dataset Kaggle (2200 amostras, 22 culturas)
│
├── db/
│   ├── __init__.py
│   ├── connection.py        # Context manager para conexão SQLite
│   └── ingest.py            # Carga do CSV → banco de dados
│
├── document/
│   └── other/               # Documentos complementares
│
├── ml/
│   ├── __init__.py
│   ├── enums.py             # ModelType e Priority (str, Enum)
│   ├── features.py          # Constantes de features, load_data(), prepare()
│   ├── pipeline.py          # Orquestração do pipeline completo
│   ├── recommendations.py   # Geração do plano de recomendações
│   └── training.py          # Treinamento dos modelos (regressão + classificação)
│
├── models/                  # Modelos treinados (gerado automaticamente, não versionado)
│   └── models.joblib
│
├── config.py                # Caminhos centralizados (ROOT, CSV_PATH, DB_PATH, MODEL_PATH)
├── main.py                  # Ponto de entrada: setup do banco + inicialização do Streamlit
├── requirements.txt         # Dependências Python
└── README.md
```

### Descrição dos módulos principais

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Ponto de entrada: ingest condicional + `streamlit run` via subprocess |
| `config.py` | Única fonte de verdade para todos os caminhos do projeto |
| `db/connection.py` | Context manager `connection()` para SQLite (garante fechamento seguro) |
| `db/ingest.py` | Lê o CSV, cria as tabelas e popula o banco |
| `ml/enums.py` | `ModelType` (Linear, Ridge, Random Forest) e `Priority` (CRITICA, ALTA, MEDIA, OK) |
| `ml/features.py` | Carrega dados do banco, aplica LabelEncoder e cria features derivadas |
| `ml/training.py` | `RegressionResult` dataclass, `_build_pipeline()`, `train_regression()`, `train_classification()` |
| `ml/pipeline.py` | `PipelineResult` dataclass, `run_pipeline()` — orquestra tudo e salva o joblib |
| `ml/recommendations.py` | Compara leituras com faixas ótimas e gera recomendações priorizadas |
| `dashboard/app.py` | Interface Streamlit com 6 páginas, cache de dados e modelos |

---

## 🔧 Como executar o código

### Pré-requisitos

- Python 3.10 ou superior
- PowerShell (Windows) ou terminal Unix
- Git

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/farmtech-fase4.git
cd farmtech-fase4
```

### 2. Criar e ativar o ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Iniciar o dashboard

```bash
python main.py
```

O `main.py` verifica se o banco de dados já existe; caso contrário, executa a ingestão do CSV automaticamente antes de subir o Streamlit. Funciona em qualquer sistema operacional.

### 5. Treinar os modelos

Na primeira vez, acesse a tela **"Modelos de Regressão"** no menu lateral e clique em **"Treinar / Re-treinar todos os modelos"**. Os modelos são salvos em `models/models.joblib` e reutilizados nas demais telas.

### Acesso online (Streamlit Cloud)

O dashboard está publicado em: **[link do deploy]**

Não é necessário nenhum pré-requisito local — o banco e os modelos são gerados automaticamente na nuvem.

---

## 📊 Entregáveis

### Parte 1 — Integração ML + Dashboard Streamlit (3 + 2 pts)

- [x] Pipeline completo de Machine Learning com Scikit-Learn integrado ao dashboard Streamlit
- [x] Dashboard interativo com métricas de desempenho (MAE, MSE, RMSE, R²), gráficos de correlação e previsões em tempo real
- [x] Modelos de regressão treinados (Linear, Ridge, Random Forest) com comparação de desempenho e seleção automática do melhor modelo
- [x] Tela "Previsões Interativas" com sliders para simular condições do campo e visualizar previsões instantâneas

### Parte 2 — Algoritmos Preditivos para Ações de Manejo (3 pts)

- [x] Regressão para **irrigação** — previsão de `chuva_mm` necessária com base nas condições do solo/clima
- [x] Regressão para **fertilização** — previsão de necessidade de nitrogênio (`nitrogenio` mg/kg)
- [x] Avaliação com MAE, MSE, RMSE e R² para todos os modelos, com visualização Real vs Previsto
- [x] Tela "Recomendações" — plano de manejo priorizado (CRITICA / ALTA / MEDIA / OK) gerado a partir das previsões dos modelos
- [x] Feature engineering: `npk_total` e `npk_ratio_nk` para enriquecer o modelo

### IR ALÉM 1 — Integração IoT + Banco de Dados (bônus)

- [x] Banco de dados relacional SQLite com schema `culturas` + `leituras_sensor` (FK)
- [x] Ingestão automática do dataset CSV (2.200 amostras, 22 culturas) via `db/ingest.py`
- [x] Auto-setup: `main.py` detecta ausência do banco e executa a ingestão antes de subir o dashboard

### IR ALÉM 2 — Dashboard Analítico Interativo Online (bônus)

- [x] 6 telas interativas com Plotly: Visão Geral, EDA, Modelos de Regressão, Previsões Interativas, Recomendações, Recomendação de Cultura
- [x] Gráficos de correlação (matriz de Pearson com máscara diagonal), boxplots por cultura, dispersão interativa
- [x] Tendências de sensibilidade (ex.: impacto do pH na necessidade de N)
- [x] Classificação supervisionada — Random Forest Classifier recomenda Top 5 culturas com probabilidade de confiança
- [x] Deploy online no Streamlit Cloud (auto-ingest + auto-treino na primeira execução)

---

## 🗃 Histórico de lançamentos

* 1.0.0 — 17/06/2026
    * Pipeline de ML completo: regressão (irrigação + fertilização) e classificação (cultura)
    * Dashboard Streamlit com 6 telas interativas
    * Deploy no Streamlit Cloud com auto-ingestão
    * Refatoração arquitetural: `db/`, `ml/enums.py`, `ml/training.py`, `ml/features.py`, dataclasses, ENUMs
* 0.2.0 — Fase 3
    * Simulação de sensores IoT via Wokwi
    * Banco SQLite inicial com dados sintéticos
* 0.1.0 — Fase 1–2
    * Estrutura inicial do projeto FarmTech
    * Análise exploratória dos dados de solo e clima

---

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/agodoi/template">MODELO GIT FIAP</a> por <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://fiap.com.br">Fiap</a> está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>
