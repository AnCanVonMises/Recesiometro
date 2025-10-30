import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import requests
import plotly.express as px

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(page_title="📊 Recesiómetro IA", layout="wide")
st.markdown("<h1 style='text-align:center; color:#1f77b4;'>📊 Recesiómetro IA</h1>", unsafe_allow_html=True)

# =========================
# CLAVES API
# =========================
FRED_API_KEY = st.secrets["FRED_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

fred = Fred(api_key=FRED_API_KEY)

# =========================
# INDICADORES CLAVE
# =========================
key_indicators = {
    "Real GDP": "GDPC1",
    "Unemployment": "UNRATE",
    "CPI": "CPIAUCSL",
    "Industrial Production": "INDPRO",
    "Yield Curve (10y-3m)": ["GS10", "TB3MS"],
    "Consumer Confidence": "UMCSENT"
}

# =========================
# DESCARGA DE DATOS
# =========================
df = pd.DataFrame()

for name, code in key_indicators.items():
    try:
        if isinstance(code, list):
            df["10Y"] = fred.get_series(code[0])
            df["3M"] = fred.get_series(code[1])
            df["Yield Curve (10y-3m)"] = df["10Y"] - df["3M"]
        else:
            df[name] = fred.get_series(code)
    except:
        st.warning(f"No disponible: {name}")

df = df.asfreq('D').interpolate(method='linear')
df.sort_index(inplace=True)

# =========================
# CÁLCULO RIESGO
# =========================
weights = {
    "Yield Curve (10y-3m)": 35,
    "Real GDP": 30,
    "Unemployment": 30,
    "CPI": 25,
    "Industrial Production": 15,
    "Consumer Confidence": 10
}

risk_direction = {
    "Yield Curve (10y-3m)": False,
    "Real GDP": False,
    "Unemployment": True,
    "CPI": True,
    "Industrial Production": False,
    "Consumer Confidence": False
}

df_risk = df.copy()
df_risk = df_risk.ffill()
risk = pd.Series(index=df_risk.index, dtype=float)
pct_change = df_risk.pct_change().fillna(0)

for date in df_risk.index:
    value = 0
    for var in weights.keys():
        if var not in df_risk.columns:
            continue
        change = pct_change.loc[date, var]
        if not risk_direction[var]:
            change = -change
        change = np.clip(change, -0.2, 0.2)
        value += change * weights[var]
    # Penalización yield curve invertida
    if "Yield Curve (10y-3m)" in df_risk.columns and df_risk.loc[date, "Yield Curve (10y-3m)"] < 0:
        value += weights["Yield Curve (10y-3m)"] * 0.5
    risk[date] = max(0, min(100, value))

df_risk["Risk (%)"] = risk

# =========================
# EVENTOS CLAVE
# =========================
df_risk["Delta"] = df_risk["Risk (%)"].diff()
events = df_risk[df_risk["Delta"] > 5]

# =========================
# GRAFICA
# =========================
fig = px.line(df_risk, y="Risk (%)", title="Riesgo de Recesión (%)")
for i, row in events.iterrows():
    # Indicador que más contribuyó al aumento
    delta_vars = {var: pct_change.loc[i, var] for var in weights.keys() if var in pct_change.columns}
    top_var = max(delta_vars, key=lambda k: abs(delta_vars[k]))
    
    fig.add_annotation(
        x=i,
        y=row["Risk (%)"] + 1,
        text=f"⬆ {top_var} sube",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-30,
        font=dict(color="white", size=10),
        align="center",
        bgcolor="red",
        bordercolor="black"
    )

st.plotly_chart(fig, use_container_width=True)

# =========================
# FUNCION EXPLICACION IA
# =========================
def explain_risk_with_llm(risk_value, context_vars):
    prompt = f"""
Eres un analista siguiendo los principios de Ray Dalio.
Riesgo de recesión actual: {risk_value:.1f}%
Indicadores recientes: {context_vars}

Proporciona una explicación detallada de por qué el riesgo es así, en qué se basa y cuál es el porcentaje actual de riesgo de recesión.
Responde en texto plano.
"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "prompt": prompt,
        "model": "llama3-70b",
        "max_output_tokens": 300,
        "temperature": 0.7
    }
    try:
        response = requests.post("https://api.groq.com/v1/generate", headers=headers, json=payload)
        return response.json().get("completion", response.text)
    except Exception as e:
        return f"⚠️ AI error: {e}"

latest_risk = df_risk["Risk (%)"].iloc[-1]
context_vars = df_risk.tail(1).to_dict(orient="records")[0]

explanation = explain_risk_with_llm(latest_risk, context_vars)

st.markdown("### 🤖 Evaluación de Riesgo (IA)")
st.write(explanation)
st.markdown(f"**Riesgo de recesión al {df_risk.index[-1].date()}: {latest_risk:.1f}%**")

