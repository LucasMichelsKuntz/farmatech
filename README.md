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
├── requirements.txt         # Dependências Python
├── run.ps1                  # Script de inicialização (Windows PowerShell)
└── README.md
```

### Descrição dos módulos principais

| Arquivo | Responsabilidade |
|---|---|
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

**Windows (script automático):**
```powershell
.\run.ps1
```

**Manual (qualquer OS):**
```bash
streamlit run dashboard/app.py
```

O script `run.ps1` verifica se o banco de dados já existe; caso contrário, executa a ingestão do CSV automaticamente. Ao abrir o dashboard em http://localhost:8501, se o banco ainda não existir (ex.: Streamlit Cloud), ele é criado automaticamente na primeira execução.

### 5. Treinar os modelos

Na primeira vez, acesse a tela **"Modelos de Regressão"** no menu lateral e clique em **"Treinar / Re-treinar todos os modelos"**. Os modelos são salvos em `models/models.joblib` e reutilizados nas demais telas.

### Acesso online (Streamlit Cloud)

O dashboard está publicado em: **[link do deploy]**

Não é necessário nenhum pré-requisito local — o banco e os modelos são gerados automaticamente na nuvem.

---

## 📊 Entregáveis

- [x] **Dataset real** integrado ao pipeline — Kaggle *Crop Recommendation Dataset* (2.200 amostras, 22 culturas, 7 features de sensores)
- [x] **Banco de dados relacional** — SQLite com tabelas `culturas` e `leituras_sensor`, populado automaticamente a partir do CSV
- [x] **Feature engineering** — `npk_total` e `npk_ratio_nk` calculados e integrados ao pipeline de ML
- [x] **Regressão supervisionada — Irrigação** — Linear, Ridge e Random Forest treinados e comparados por MAE/RMSE/R²; melhor modelo salvo
- [x] **Regressão supervisionada — Fertilização** — idem para previsão de necessidade de nitrogênio
- [x] **Classificação supervisionada** — Random Forest Classifier com 22 classes (culturas), retorna Top 5 com probabilidade
- [x] **Dashboard interativo** — 6 telas em Streamlit com Plotly: EDA, comparação de modelos, previsões em tempo real, plano de recomendações, recomendação de cultura
- [x] **Persistência de modelos** — `joblib` com chaves `irrigation`, `fertilization`, `classification`
- [x] **Código profissional** — ENUMs (`ModelType`, `Priority`), dataclasses (`RegressionResult`, `PipelineResult`, `Recommendation`), sem comentários redundantes, sem strings hardcoded para despacho de modelo
- [x] **Deploy** — Streamlit Cloud (auto-ingest na primeira execução)

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
