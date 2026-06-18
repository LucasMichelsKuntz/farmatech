# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href="https://www.fiap.com.br/"><img src="assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Administração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# FarmTech Solutions — Fase 4: Machine Learning Aplicado à Agricultura de Precisão

## Grupo FarmTech Solutions

## 👨‍🎓 Integrantes:
- <a href="https://www.linkedin.com/in/">Lucas Michels Kuntz</a> — RM 570050


## 👩‍🏫 Professores:
### Tutor(a)
- <a href="https://www.linkedin.com/in/">Sabrina Otoni</a>
### Coordenador(a)
- <a href="https://www.linkedin.com/in/">André Godoi</a>

---

## Video

[Assistir apresentação no Google Drive](https://drive.google.com/file/d/1YdioHfH0nju1Ygr8TNKsqscgm_wCHxAF/view?usp=sharing)

---

## Descrição

A **FarmTech Solutions** é uma empresa de tecnologia agrícola que desenvolve soluções inteligentes para o agronegócio. Nesta Fase 4, o desafio proposto pela FIAP consiste em evoluir o sistema de monitoramento de sensores IoT desenvolvido nas fases anteriores, incorporando **Machine Learning supervisionado** para tomada de decisão autônoma no campo.

### Contexto e Problema

Produtores rurais enfrentam dois problemas recorrentes de alto impacto financeiro:

1. **Irrigação ineficiente**: irrigar em excesso eleva custos de energia e água e pode causar encharcamento; irrigar de menos causa stress hídrico e reduz a produtividade.
2. **Fertilização imprecisa**: a deficiência de nitrogênio (N) é a principal causa de queda de rendimento em culturas como algodão, arroz e café, mas a superdosagem contamina o solo e o lençol freático.

O sistema de sensores IoT (simulado via Wokwi nas fases anteriores) coleta dados de **nitrogênio, fósforo, potássio, temperatura, umidade, pH e precipitação** para 22 culturas diferentes. Com 2.200 leituras do *Kaggle Crop Recommendation Dataset* (100 amostras por cultura), o desafio é treinar modelos de ML capazes de:

- **Prever a necessidade de irrigação** (regressão sobre `chuva_mm`)
- **Prever a necessidade de fertilização nitrogenada** (regressão sobre `nitrogenio`)
- **Recomendar qual cultura plantar** dado o conjunto de condições do solo e clima (classificação)

---

### O que foi construído

**1. Ingestão de Dados**
O dataset CSV é carregado e persistido em um banco **SQLite** com schema relacional (`culturas` + `leituras_sensor`), simulando o fluxo real de dados IoT → banco de dados → modelo.

**2. Feature Engineering**
Além das 7 features originais dos sensores, criamos duas features derivadas para o modelo de irrigação:
- `npk_total` = N + P + K (fertilidade total do solo)
- `npk_ratio_nk` = N / (K + 1) (balanço nitrogênio-potássio)

> Essas features derivadas **não são usadas no modelo de fertilização** (target = `nitrogenio`), pois npk_total e npk_ratio_nk codificam o próprio N — incluí-las causaria data leakage e faria o modelo simplesmente devolver o N atual como previsão.

**3. Modelos de Regressão (Irrigação e Fertilização)**
Para cada target, treinamos e comparamos três algoritmos com `Pipeline(StandardScaler + modelo)`:

| Modelo | Tipo | Hiperparâmetros |
|---|---|---|
| Regressão Linear | Linear | — |
| Ridge (L2) | Linear regularizado | alpha=1.0 |
| Random Forest | Ensemble não-linear | 150 estimadores, max_depth=10 |

O modelo com maior **Coeficiente de Determinação (R²)** é selecionado automaticamente como "melhor modelo" e salvo em `models/models.joblib`.

**4. Modelo de Classificação (Recomendação de Cultura)**
Um **Random Forest Classifier** (200 estimadores, max_depth=12) foi treinado para classificar entre 22 culturas. O modelo retorna as **Top 5 culturas recomendadas** com probabilidade de confiança.

**5. Dashboard Interativo (Streamlit)**
A interface web conta com 5 telas:

| Tela | Conteúdo |
|---|---|
| Visão Geral | Métricas do dataset, distribuição por cultura, estatísticas descritivas |
| Análise Exploratória | Histogramas, matriz de correlação, boxplots, dispersão interativa |
| Previsões Interativas | Sliders para simular condições do campo; gauges de irrigação e N; análise de sensibilidade ao pH |
| Recomendações | Plano de manejo priorizado (CRITICA / ALTA / MEDIA / OK) para N, P, K, irrigação, pH, umidade e temperatura |
| Recomendação de Cultura | Top 5 culturas recomendadas com barra de confiança e importância de features |

---

### O que os modelos aprenderam — validação agronômica

#### Por que o Random Forest supera em muito os modelos lineares

Correlações lineares entre as features e os targets são muito fracas (r < 0.11 para chuva_mm; r < 0.23 para nitrogenio). Isso explica o R² ≈ 0.03 dos modelos lineares: as relações no dataset não são lineares, são **cluster-based por cultura**. O Random Forest consegue R² = 0.81 (irrigação) e R² = 0.87 (fertilização) porque pode criar partições e capturar o padrão de cada cultura separadamente.

#### O que o modelo aprendeu sobre irrigação

O tipo de cultura (`cultura_cod`) responde por **47% da importância** do modelo de irrigação — e faz sentido: a necessidade de água é radicalmente diferente entre culturas. O modelo aprendeu os perfis corretos:

| Cultura | Chuva média (mm) | Validação agronômica |
|---|---|---|
| rice | 236 | Arroz cresce em solo inundado — maior demanda hídrica |
| coconut | 176 | Palmeira tropical, alto consumo |
| coffee | 158 | Cafeeiro precisa de chuva bem distribuída |
| chickpea | 80 | Leguminosa tolerante à seca |
| muskmelon | 25 | Clima árido, irrigação controlada |

> `fosforo` aparece com 30% de importância no modelo de irrigação — não por causalidade física, mas porque culturas como uva e maçã têm P muito alto (~132 mg/kg) e regime hídrico diferente de culturas como arroz (P ~47 mg/kg). O modelo usa P como identificador indireto de cultura.

#### O que o modelo aprendeu sobre fertilização nitrogenada

Após remover o data leakage (npk_total e npk_ratio_nk excluídos do modelo de N), as features secundárias mais importantes são agronômicamente válidas:

- **`chuva_mm` (19%)** — chuva intensa lixivia N do solo; maior precipitação → maior necessidade de reposição de N
- **`potassio` (14%)** — antagonismo K×N na absorção radicular: solos ricos em K competem com a absorção de N
- **`umidade` (13%)** — umidade do solo afeta a mineralização da matéria orgânica, principal fonte de N disponível

Os perfis de N por cultura também são corretos:

| Cultura | N médio (mg/kg) | Por quê |
|---|---|---|
| cotton | 118 | Algodão é grande consumidor de N |
| coffee | 101 | Cafeeiro exige alta adubação nitrogenada |
| rice | 80 | Demanda moderada-alta |
| lentil / mungbean | ~19–21 | Leguminosas fixam N biologicamente — menor demanda |
| apple / grapes | ~21–23 | Frutíferas priorizam P e K para formação do fruto |

#### O que o modelo aprendeu sobre classificação de culturas

Acurácia de **99.3%** no conjunto de teste. Features mais importantes: `chuva_mm` (22%) e `umidade` (21%) — o que faz sentido: o bioma/clima onde uma cultura prospera é o maior discriminador. pH (5.5%) tem o menor peso, pois a maioria das culturas tolera uma faixa relativamente ampla.

---

### Tecnologias utilizadas
- **Python 3.11** — linguagem principal
- **Scikit-learn** — treinamento dos modelos de ML
- **Pandas / NumPy** — manipulação de dados
- **SQLite** — banco de dados relacional local
- **Streamlit** — dashboard web interativo
- **Plotly** — visualizações interativas
- **Joblib** — persistência dos modelos treinados

---

## Estrutura de pastas

```
farmtech-fase4/
│
├── assets/                  # Imagens e recursos estáticos
│   └── logo-fiap.png
│
├── dashboard/
│   └── app.py               # Aplicação Streamlit (5 telas)
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
├── main.py                  # Entrypoint: ingestão automática + inicialização do Streamlit
├── requirements.txt         # Dependências Python
└── README.md
```

### Descrição dos módulos principais

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Entrypoint: verifica banco, executa ingestão e sobe o Streamlit |
| `config.py` | Única fonte de verdade para todos os caminhos do projeto |
| `db/connection.py` | Context manager `connection()` para SQLite (garante fechamento seguro) |
| `db/ingest.py` | Lê o CSV, cria as tabelas e popula o banco |
| `ml/enums.py` | `ModelType` (Linear, Ridge, Random Forest) e `Priority` (CRITICA, ALTA, MEDIA, OK) |
| `ml/features.py` | Carrega dados do banco, aplica LabelEncoder e cria features derivadas |
| `ml/training.py` | `RegressionResult` dataclass, `_build_pipeline()`, `train_regression()`, `train_classification()` |
| `ml/pipeline.py` | `PipelineResult` dataclass, `run_pipeline()` — orquestra tudo e salva o joblib |
| `ml/recommendations.py` | Compara leituras do campo com faixas ótimas agronômicas e previsões do modelo; gera plano de manejo priorizado |
| `dashboard/app.py` | Interface Streamlit com 5 páginas, cache de dados e modelos |

---

## Como executar o código

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

O script verifica se o banco já existe, executa a ingestão do CSV caso necessário e sobe o Streamlit automaticamente. Equivalente a executar `streamlit run dashboard/app.py` após popular o banco manualmente.

### 5. Treinamento dos modelos

O pipeline de ML (ingestão → feature engineering → treino dos 3 modelos por target → seleção do melhor → persistência em `models/models.joblib`) é executado automaticamente na primeira vez que qualquer tela de previsão for acessada. Nas execuções seguintes os modelos são lidos do disco.

### Acesso online (Streamlit Cloud)

O dashboard está publicado em: **[link do deploy]**

Não é necessário nenhum pré-requisito local — o banco e os modelos são gerados automaticamente na nuvem.

---

## Entregáveis

- [x] **Dataset real** integrado ao pipeline — Kaggle *Crop Recommendation Dataset* (2.200 amostras, 22 culturas, 7 features de sensores)
- [x] **Banco de dados relacional** — SQLite com tabelas `culturas` e `leituras_sensor`, populado automaticamente a partir do CSV
- [x] **Feature engineering** — `npk_total` e `npk_ratio_nk` (usadas no modelo de irrigação; excluídas do modelo de fertilização para evitar data leakage)
- [x] **Regressão supervisionada — Irrigação** — Linear, Ridge e Random Forest treinados e comparados; Random Forest R² = 0.81; melhor modelo salvo automaticamente
- [x] **Regressão supervisionada — Fertilização** — idem; Random Forest R² = 0.87
- [x] **Classificação supervisionada** — Random Forest Classifier 99.3% de acurácia; 22 classes; retorna Top 5 com probabilidade
- [x] **Dashboard interativo** — 5 telas em Streamlit com Plotly: EDA, previsões em tempo real, plano de recomendações completo (N, P, K, irrigação, pH, umidade, temperatura), recomendação de cultura
- [x] **Persistência de modelos** — `joblib` com chaves `irrigation`, `fertilization`, `classification`
- [x] **Validação agronômica** — importâncias de features e perfis por cultura conferem com literatura agronômica (leguminosas com N baixo, arroz com alta demanda hídrica, frutíferas com alto P/K)

---

## Histórico de lançamentos

* 1.0.0 — 17/06/2026
    * Pipeline de ML completo: regressão (irrigação + fertilização) e classificação (cultura)
    * Dashboard Streamlit com 6 telas interativas
    * Correção de data leakage no modelo de fertilização (npk features excluídas)
    * Plano de recomendações expandido: N, P, K, irrigação, pH, umidade e temperatura
    * Validação agronômica das importâncias de features e perfis por cultura
    * Deploy no Streamlit Cloud com auto-ingestão
* 0.2.0 — Fase 3
    * Simulação de sensores IoT via Wokwi
    * Banco SQLite inicial com dados sintéticos
* 0.1.0 — Fase 1–2
    * Estrutura inicial do projeto FarmTech
    * Análise exploratória dos dados de solo e clima
