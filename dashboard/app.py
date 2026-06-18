import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CSV_PATH, DB_PATH, MODEL_PATH
from ml.enums import ModelType, Priority
from ml.features import load_data, prepare
from ml.recommendations import generate_recommendations

VERDE   = "#2E7D32"
AZUL    = "#1565C0"
AMARELO = "#F9A825"
LARANJA = "#E65100"
VERMELHO = "#C62828"

st.set_page_config(
    page_title="FarmTech IA",
    layout="wide",
    initial_sidebar_state="expanded",
)

_REC_STYLE = {
    Priority.CRITICAL: (VERMELHO, "#FFEBEE"),
    Priority.HIGH:     (LARANJA,  "#FFF3E0"),
    Priority.MEDIUM:   (AMARELO,  "#FFFDE7"),
    Priority.OK:       (VERDE,    "#E8F5E9"),
}


@st.cache_resource(show_spinner="Preparando banco de dados...")
def _ensure_db():
    if not DB_PATH.exists():
        from db.ingest import ingest
        ingest()


@st.cache_data(show_spinner="Carregando dados do banco...")
def _data():
    _ensure_db()
    df = load_data()
    df, le = prepare(df)
    return df, le


@st.cache_resource(show_spinner="Treinando modelos de IA...")
def _train():
    from ml.pipeline import run_pipeline
    return run_pipeline()


def _load_models():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


with st.sidebar:
    st.markdown("## FarmTech IA")
    st.caption("Assistente Agricola Inteligente — Fase 4 FIAP")
    st.divider()
    page = st.radio("Navegacao", [
        "Visao Geral",
        "Analise Exploratoria",
        "Modelos de Regressao",
        "Previsoes Interativas",
        "Recomendacoes",
        "Recomendacao de Cultura",
    ])
    st.divider()
    st.success("Banco conectado")

df, le = _data()
crops = sorted(df["cultura"].unique())

if page == "Visao Geral":
    st.title("FarmTech — Assistente Agricola Inteligente")
    st.caption("Pipeline: Sensores IoT -> SQLite -> Machine Learning -> Recomendacoes de Manejo")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amostras no Banco", f"{len(df):,}")
    c2.metric("Culturas Catalogadas", df["cultura"].nunique())
    c3.metric("Media de Chuva (mm)", f"{df['chuva_mm'].mean():.1f}")
    c4.metric("Media de pH", f"{df['ph'].mean():.2f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuicao de Amostras por Cultura")
        cnt = df["cultura"].value_counts().reset_index()
        cnt.columns = ["Cultura", "Amostras"]
        fig = px.bar(cnt, x="Amostras", y="Cultura", orientation="h",
                     color="Amostras", color_continuous_scale="Greens")
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Necessidade de Chuva por Cultura (mediana)")
        med = df.groupby("cultura")["chuva_mm"].median().sort_values(ascending=False).reset_index()
        fig2 = px.bar(med, x="chuva_mm", y="cultura", orientation="h",
                      color="chuva_mm", color_continuous_scale="Blues",
                      labels={"chuva_mm": "Chuva mediana (mm)", "cultura": "Cultura"})
        fig2.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Estatisticas do Dataset")
    desc = (
        df[["nitrogenio", "fosforo", "potassio", "temperatura", "umidade", "ph", "chuva_mm"]]
        .describe().round(2)
    )
    desc.index = ["qtd amostras", "media", "desvio padrao", "minimo",
                  "percentil 25%", "mediana (50%)", "percentil 75%", "maximo"]
    st.dataframe(desc, use_container_width=True)

elif page == "Analise Exploratoria":
    st.title("Analise Exploratoria dos Dados (EDA)")
    st.info("Etapa fundamental antes do treinamento: entender distribuicoes, outliers e correlacoes.")

    tab1, tab2, tab3, tab4 = st.tabs(["Distribuicoes", "Correlacoes", "Boxplots por Cultura", "Dispersao"])

    features_num = ["nitrogenio", "fosforo", "potassio", "temperatura", "umidade", "ph", "chuva_mm"]

    with tab1:
        col = st.selectbox("Variavel:", features_num)
        crop_filter = st.multiselect("Filtrar culturas:", crops, default=crops[:5])
        df_f = df[df["cultura"].isin(crop_filter)] if crop_filter else df
        fig = px.histogram(df_f, x=col, color="cultura", nbins=40, barmode="overlay",
                           opacity=0.7, labels={col: col, "cultura": "Cultura"})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Matriz de Correlacao de Pearson")
        corr = df[features_num].corr()
        corr_masked = corr.mask(np.eye(len(corr), dtype=bool))
        fig_c = px.imshow(corr_masked, text_auto=".2f", aspect="auto",
                          color_continuous_scale="RdYlGn",
                          title="Correlacao entre variaveis numericas")
        st.plotly_chart(fig_c, use_container_width=True)

        st.caption(
            "**Interpretacao:** valores proximos de +/-1 indicam forte correlacao linear. "
            "Features com alta correlacao com o target sao mais preditivas para o modelo."
        )
        target_sel = st.selectbox("Ver correlacao com:", features_num, index=6)
        corr_t = corr[target_sel].drop(target_sel).sort_values(ascending=False)
        fig_b = px.bar(x=corr_t.values, y=corr_t.index, orientation="h",
                       color=corr_t.values, color_continuous_scale="RdYlGn",
                       labels={"x": f"Correlacao com {target_sel}", "y": "Feature"})
        fig_b.update_layout(showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

    with tab3:
        feat_box = st.selectbox("Feature para boxplot:", features_num, key="box")
        crops_box = st.multiselect("Culturas:", crops, default=crops[:8], key="box_c")
        df_b = df[df["cultura"].isin(crops_box)] if crops_box else df
        fig_box = px.box(df_b, x="cultura", y=feat_box, color="cultura",
                         labels={"cultura": "Cultura", feat_box: feat_box})
        fig_box.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        x_ax = st.selectbox("Eixo X:", features_num, index=0)
        y_ax = st.selectbox("Eixo Y:", features_num, index=6)
        crops_sc = st.multiselect("Culturas:", crops, default=crops[:6], key="sc_c")
        df_sc = df[df["cultura"].isin(crops_sc)] if crops_sc else df
        fig_sc = px.scatter(df_sc, x=x_ax, y=y_ax, color="cultura",
                            opacity=0.6, labels={x_ax: x_ax, y_ax: y_ax, "cultura": "Cultura"})
        st.plotly_chart(fig_sc, use_container_width=True)

elif page == "Modelos de Regressao":
    st.title("Modelos de Regressao")
    st.info(
        "Comparacao de tres algoritmos por target. "
        "O melhor por Coeficiente de Determinacao e salvo automaticamente."
    )

    res   = _train()
    saved = _load_models()
    if saved is None:
        st.warning("Modelos nao encontrados. Recarregue a pagina para treinar.")
        st.stop()

    def _show_results(label, result, saved_key):
        st.subheader(f"Comparacao de Modelos — {label}")
        st.dataframe(result.metrics, use_container_width=True, hide_index=True)

        best_row = result.metrics.loc[
            result.metrics["Coeficiente de Det. (R²)"].idxmax()
        ]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Melhor modelo",              best_row["model"])
        c2.metric("Erro Medio Absoluto",        best_row["Erro Medio Absoluto"])
        c3.metric("Raiz do Erro Quadratico",    best_row["Raiz do Erro Quadratico"])
        c4.metric("Coeficiente de Determinacao", best_row["Coeficiente de Det. (R²)"])

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Real vs Previsto")
            y_pred = saved[saved_key]["model"].predict(result.X_test)
            fig = px.scatter(
                x=result.y_test, y=y_pred,
                labels={"x": "Valor Real", "y": "Valor Previsto"},
                opacity=0.6, color_discrete_sequence=[AZUL],
            )
            lim = [
                min(result.y_test.min(), y_pred.min()),
                max(result.y_test.max(), y_pred.max()),
            ]
            fig.add_shape(
                type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                line=dict(color=VERMELHO, dash="dash"),
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            mdl = saved[saved_key]["model"].named_steps["model"]
            if hasattr(mdl, "feature_importances_"):
                st.subheader("Importancia das Features (Random Forest)")
                feats  = saved[saved_key]["features"]
                imp_df = pd.DataFrame({"Feature": feats, "Importancia": mdl.feature_importances_})
                imp_df = imp_df.sort_values("Importancia", ascending=True)
                fig2 = px.bar(
                    imp_df, x="Importancia", y="Feature", orientation="h",
                    color="Importancia", color_continuous_scale="Greens",
                )
                fig2.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

    tab_irr, tab_fer = st.tabs(["Irrigacao (chuva_mm)", "Fertilizacao (nitrogenio)"])
    with tab_irr:
        _show_results("Irrigacao", res.irr, "irrigation")
    with tab_fer:
        _show_results("Fertilizacao", res.fer, "fertilization")

elif page == "Previsoes Interativas":
    st.title("Previsoes Interativas")
    st.info("Ajuste os parametros do campo e veja as previsoes do modelo em tempo real.")

    res   = _train()
    saved = _load_models()
    if saved is None:
        st.warning("Modelos nao encontrados. Recarregue a pagina para treinar.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Nutrientes do Solo")
        nitrogenio = st.slider("Nitrogenio — N", 0, 140, 50)
        fosforo    = st.slider("Fosforo — P", 5, 145, 50)
        potassio   = st.slider("Potassio — K", 5, 205, 50)

    with col2:
        st.subheader("Clima")
        temperatura = st.slider("Temperatura (C)", 8.0, 44.0, 25.0, 0.5)
        umidade     = st.slider("Umidade (%)", 14.0, 100.0, 65.0, 0.5)
        chuva_mm    = st.slider("Chuva atual (mm)", 20.0, 300.0, 100.0, 1.0)

    with col3:
        st.subheader("Quimica")
        ph         = st.slider("pH do Solo", 3.5, 10.0, 6.5, 0.1)
        crop_sel   = st.selectbox("Cultura atual:", crops)
        cultura_cod = int(le.transform([crop_sel])[0])

    npk_total    = nitrogenio + fosforo + potassio
    npk_ratio_nk = nitrogenio / (potassio + 1)

    input_irr = pd.DataFrame([{
        "nitrogenio": nitrogenio, "fosforo": fosforo, "potassio": potassio,
        "temperatura": temperatura, "umidade": umidade, "ph": ph,
        "chuva_mm": chuva_mm, "cultura_cod": cultura_cod,
        "npk_total": npk_total, "npk_ratio_nk": npk_ratio_nk,
    }])[res.irr.features]

    input_fer = pd.DataFrame([{
        "nitrogenio": nitrogenio, "fosforo": fosforo, "potassio": potassio,
        "temperatura": temperatura, "umidade": umidade, "ph": ph,
        "chuva_mm": chuva_mm, "cultura_cod": cultura_cod,
        "npk_total": npk_total, "npk_ratio_nk": npk_ratio_nk,
    }])[res.fer.features]

    irr_pred = saved["irrigation"]["model"].predict(input_irr)[0]
    fer_pred = saved["fertilization"]["model"].predict(input_fer)[0]

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Irrigacao Prevista")
        delta_irr = irr_pred - chuva_mm
        st.metric(
            "Volume de Chuva/Irrigacao Necessario",
            f"{irr_pred:.1f} mm",
            delta=f"{delta_irr:+.1f} mm vs atual",
            delta_color="inverse",
        )
        fig_g1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=irr_pred,
            number={"suffix": " mm", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 300]},
                "bar": {"color": AZUL},
                "steps": [
                    {"range": [0, 60],    "color": "#FFF9C4"},
                    {"range": [60, 150],  "color": "#B3E5FC"},
                    {"range": [150, 300], "color": "#90CAF9"},
                ],
            },
        ))
        fig_g1.update_layout(height=260, margin=dict(t=20, b=10))
        st.plotly_chart(fig_g1, use_container_width=True)

    with c2:
        st.subheader("Nitrogenio Previsto")
        delta_fer = fer_pred - nitrogenio
        st.metric(
            "Necessidade de Nitrogenio (N)",
            f"{fer_pred:.1f} mg/kg",
            delta=f"{delta_fer:+.1f} mg/kg vs atual",
            delta_color="inverse",
        )
        fig_g2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fer_pred,
            number={"suffix": " mg/kg", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 140]},
                "bar": {"color": VERDE},
                "steps": [
                    {"range": [0, 40],    "color": "#FFCDD2"},
                    {"range": [40, 100],  "color": "#C8E6C9"},
                    {"range": [100, 140], "color": "#A5D6A7"},
                ],
            },
        ))
        fig_g2.update_layout(height=260, margin=dict(t=20, b=10))
        st.plotly_chart(fig_g2, use_container_width=True)

    st.divider()
    st.subheader("Sensibilidade: como o pH afeta a necessidade de N?")
    phs = np.linspace(3.5, 10.0, 60)
    ph_preds = []
    for p in phs:
        e = pd.DataFrame([{
            "nitrogenio": nitrogenio, "fosforo": fosforo, "potassio": potassio,
            "temperatura": temperatura, "umidade": umidade, "ph": p,
            "chuva_mm": chuva_mm, "cultura_cod": cultura_cod,
            "npk_total": npk_total, "npk_ratio_nk": npk_ratio_nk,
        }])[res.fer.features]
        ph_preds.append(saved["fertilization"]["model"].predict(e)[0])

    fig_ph = px.line(x=phs, y=ph_preds,
                     labels={"x": "pH do Solo", "y": "N Previsto (mg/kg)"},
                     color_discrete_sequence=[VERDE])
    fig_ph.add_vline(x=ph, line_dash="dash", line_color=VERMELHO,
                      annotation_text=f"pH atual: {ph}")
    fig_ph.add_vrect(x0=5.5, x1=7.5, fillcolor=VERDE, opacity=0.08,
                      annotation_text="Faixa otima")
    st.plotly_chart(fig_ph, use_container_width=True)

elif page == "Recomendacoes":
    st.title("Plano de Recomendacoes de Manejo")

    res   = _train()
    saved = _load_models()
    if saved is None:
        st.warning("Modelos nao encontrados. Recarregue a pagina para treinar.")
        st.stop()

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("Parametros do Campo")
        crop_r = st.selectbox("Cultura:", crops)
        n_r    = st.slider("Nitrogenio — N (mg/kg)", 0, 140, 50)
        p_r    = st.slider("Fosforo — P (mg/kg)",    5, 145, 50)
        k_r    = st.slider("Potassio — K (mg/kg)",   5, 205, 50)
        t_r    = st.slider("Temperatura (C)",        8.0, 44.0, 25.0, 0.5)
        u_r    = st.slider("Umidade (%)",           14.0, 100.0, 65.0, 0.5)
        ph_r   = st.slider("pH do Solo",             3.5, 10.0, 6.5, 0.1)
        rain_r = st.slider("Chuva atual (mm)",      20.0, 300.0, 100.0, 1.0)

    crop_cod = int(le.transform([crop_r])[0])
    npk_t    = n_r + p_r + k_r
    npk_rk   = n_r / (k_r + 1)

    input_irr = pd.DataFrame([{
        "nitrogenio": n_r, "fosforo": p_r, "potassio": k_r,
        "temperatura": t_r, "umidade": u_r, "ph": ph_r,
        "chuva_mm": rain_r, "cultura_cod": crop_cod,
        "npk_total": npk_t, "npk_ratio_nk": npk_rk,
    }])[res.irr.features]

    input_fer = pd.DataFrame([{
        "nitrogenio": n_r, "fosforo": p_r, "potassio": k_r,
        "temperatura": t_r, "umidade": u_r, "ph": ph_r,
        "chuva_mm": rain_r, "cultura_cod": crop_cod,
        "npk_total": npk_t, "npk_ratio_nk": npk_rk,
    }])[res.fer.features]

    irr_pred = saved["irrigation"]["model"].predict(input_irr)[0]
    fer_pred = saved["fertilization"]["model"].predict(input_fer)[0]

    reading = {
        "nitrogenio": n_r, "fosforo": p_r, "potassio": k_r,
        "temperatura": t_r, "umidade": u_r, "ph": ph_r, "chuva_mm": rain_r,
    }
    recs = generate_recommendations(reading, irr_pred, fer_pred)

    with col_out:
        st.subheader("Resultado do Modelo")
        m1, m2 = st.columns(2)
        m1.metric("Irrigacao necessaria", f"{irr_pred:.1f} mm")
        m2.metric("Nitrogenio necessario", f"{fer_pred:.1f} mg/kg")
        st.caption(f"Cultura selecionada: **{crop_r}**")

        st.subheader("Acoes Recomendadas")
        for rec in recs:
            border, bg = _REC_STYLE.get(rec.priority, (VERDE, "#E8F5E9"))
            imp = f" | Impacto estimado: **+{rec.impact_pct}%**" if rec.impact_pct else ""
            code_style = "background:rgba(0,0,0,0.08);padding:1px 4px;border-radius:3px;font-family:monospace;color:#212121"
            st.markdown(f"""
<div style="background-color:{bg};border-left:4px solid {border};padding:10px;border-radius:4px;margin:6px 0;color:#212121">
  <strong>[{rec.priority.value}] {rec.parameter}</strong><br>
  Atual: <span style="{code_style}">{rec.current_value}</span> | Faixa otima: <span style="{code_style}">{rec.optimal_range}</span><br>
  {rec.action}{imp}
</div>
""", unsafe_allow_html=True)

elif page == "Recomendacao de Cultura":
    st.title("Recomendacao de Cultura (Classificacao)")
    st.info("""
**IR ALEM** — Alem das regressoes, treinamos um **Random Forest Classifier**
para recomendar qual cultura plantar com base nas condicoes atuais do solo e clima.
""")

    res   = _train()
    saved = _load_models()
    if saved is None:
        st.warning("Modelos nao encontrados. Treine primeiro.")
        st.stop()
    st.metric("Accuracy do Classificador", f"{res.acc_clf:.2%}")

    st.divider()
    st.subheader("Qual cultura plantar nas minhas condicoes?")

    c1, c2 = st.columns(2)
    with c1:
        nc  = st.slider("N (mg/kg)",        0, 140,   50,       key="cn")
        pc  = st.slider("P (mg/kg)",        5, 145,   50,       key="cp")
        kc  = st.slider("K (mg/kg)",        5, 205,   50,       key="ck")
        phc = st.slider("pH",             3.5, 10.0,  6.5, 0.1, key="cph")
    with c2:
        tc  = st.slider("Temperatura (C)", 8.0,  44.0,  25.0, 0.5, key="ct")
        uc  = st.slider("Umidade (%)",    14.0, 100.0,  65.0, 0.5, key="cu")
        rc  = st.slider("Chuva (mm)",     20.0, 300.0, 100.0, 1.0, key="cr")

    clf_model = saved["classification"]["model"]
    clf_feats = saved["classification"]["features"]
    clf_le    = saved["classification"]["le"]

    input_clf = pd.DataFrame([{
        "nitrogenio": nc, "fosforo": pc, "potassio": kc,
        "temperatura": tc, "umidade": uc, "ph": phc, "chuva_mm": rc,
    }])[clf_feats]

    proba     = clf_model.predict_proba(input_clf)[0]
    top5_idx  = proba.argsort()[::-1][:5]
    top5_crop = clf_le.inverse_transform(top5_idx)
    top5_prob = proba[top5_idx]

    st.subheader("Top 5 Culturas Recomendadas")
    ranks = ["1.", "2.", "3.", "4.", "5."]
    for i, (crop, prob) in enumerate(zip(top5_crop, top5_prob)):
        st.progress(float(prob), text=f"{ranks[i]} **{crop}** — {prob:.1%} de confianca")

    fig_top = px.bar(
        x=top5_prob * 100, y=top5_crop, orientation="h",
        labels={"x": "Confianca (%)", "y": "Cultura"},
        color=top5_prob, color_continuous_scale="Greens",
    )
    fig_top.update_layout(showlegend=False)
    st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Importancia das Features para a Classificacao")
    imp_c = pd.Series(clf_model.feature_importances_, index=clf_feats).sort_values(ascending=False)
    fig_ic = px.bar(x=imp_c.values, y=imp_c.index, orientation="h",
                     color=imp_c.values, color_continuous_scale="Greens",
                     labels={"x": "Importancia", "y": "Feature"})
    fig_ic.update_layout(showlegend=False)
    st.plotly_chart(fig_ic, use_container_width=True)
