import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import requests

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="📊 Global Recession Meter", layout="wide")
st.markdown("<h1 style='text-align:center; color:#1f77b4;'>📊 Global Recession Meter</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#555;'>Predicting recession risk using yield curve and macroeconomic indicators</h4>", unsafe_allow_html=True)

# =========================
# API KEYS
# =========================
FRED_API_KEY = st.secrets["FRED_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

fred = Fred(api_key=FRED_API_KEY)

# =========================
# INDICADORES FRED USA
# =========================
codes = {
    "USA": {
        "Real GDP": "GDPC1",
        "Unemployment": "UNRATE",
        "CPI": "CPIAUCSL",
        "Industrial Production": "INDPRO",
        "Retail Sales": "RSAFS",
        "10-Year Rate": "GS10", 
        "3-Month Rate": "TB3MS",
        "Consumer Confidence": "UMCSENT",
        "Nonfarm Payrolls": "PAYEMS",
        "Building Permits": "PERMIT",
        "S&P 500": "SP500",
        "WTI Oil": "DCOILWTICO",
        "30Y Fixed Mortgage": "MORTGAGE30US",
        "Mortgage Delinquencies": "QBPAMISM",
        "Personal Consumption Expenditure": "PCE",
        "Personal Income": "PI",
        "Durable Goods Orders": "DGORDER",
        "ISM Manufacturing": "NAPM",
        "ISM Services": "SERVPMI",
        "M2 Money Supply": "M2SL",
        "Corporate Profits": "CP"
    }
}

weights = {
    "Yield Curve (10y-3m)": 35,
    "Real GDP": 30,
    "Unemployment": 30,
    "CPI": 25,
    "Industrial Production": 15,
    "Retail Sales": 15,
    "10-Year Rate": 5,
    "3-Month Rate": 5,
    "Consumer Confidence": 10,
    "Nonfarm Payrolls": 15,
    "Building Permits": 5,
    "S&P 500": 10,
    "WTI Oil": 5,
    "30Y Fixed Mortgage": 5,
    "Mortgage Delinquencies": 10,
    "Personal Consumption Expenditure": 10,
    "Personal Income": 10,
    "Durable Goods Orders": 10,
    "ISM Manufacturing": 15,
    "ISM Services": 15,
    "M2 Money Supply": 5,
    "Corporate Profits": 15
}

risk_direction = {
    "Yield Curve (10y-3m)": False,
    "Real GDP": False,
    "Unemployment": True,
    "CPI": True,
    "Industrial Production": False,
    "Retail Sales": False,
    "10-Year Rate": True,
    "3-Month Rate": True,
    "Consumer Confidence": False,
    "Nonfarm Payrolls": False,
    "Building Permits": False,
    "S&P 500": False,
    "WTI Oil": False,
    "30Y Fixed Mortgage": True,
    "Mortgage Delinquencies": True,
    "Personal Consumption Expenditure": False,
    "Personal Income": False,
    "Durable Goods Orders": False,
    "ISM Manufacturing": False,
    "ISM Services": False,
    "M2 Money Supply": False,
    "Corporate Profits": False
}

# =========================
# SELECCION DE PAISES
# =========================
selected_countries = st.multiselect(
    "Select countries",
    options=list(codes.keys()),
    default=["USA"]
)
if not selected_countries:
    st.stop()

# =========================
# DESCARGA DE DATOS
# =========================
all_data = {}
for country in selected_countries:
    indicators = codes[country]
    df_country = pd.DataFrame()
    for name, code in indicators.items():
        try:
            serie = fred.get_series(code)
            serie.index = pd.to_datetime(serie.index)
            df_country[name] = serie
        except Exception:
            st.warning(f"⚠️ Not available: {name} ({code}) in {country}")

    if not df_country.empty:
        df_country = df_country.asfreq('D').interpolate(method='linear')
        if "CPI" in df_country.columns:
            df_country["Annual Inflation (%)"] = df_country["CPI"].pct_change(365) * 100
        if {"10-Year Rate", "3-Month Rate"} <= set(df_country.columns):
            df_country["Yield Curve (10y-3m)"] = df_country["10-Year Rate"] - df_country["3-Month Rate"]
        df_country.sort_index(inplace=True)
        all_data[country] = df_country

# =========================
# CALCULO DE RIESGO
# =========================
def calculate_risk_advanced(df):
    df = df.ffill()
    risk = pd.Series(index=df.index, dtype=float)
    pct_change = df.pct_change().fillna(0)

    for date in df.index:
        value = 0
        for var in codes["USA"]:
            if var not in df.columns:
                continue
            change = pct_change.loc[date, var]
            if not risk_direction[var]:
                change = -change
            change = np.clip(change, -0.2, 0.2)
            value += change * weights[var]
        if "Yield Curve (10y-3m)" in df.columns and df.loc[date, "Yield Curve (10y-3m)"] < 0:
            value += weights["Yield Curve (10y-3m)"] * 0.5
        risk[date] = max(0, min(100, value))
    df["Risk (%)"] = risk
    return df

# =========================
# FUNCION IA / GROQ
# =========================
def explain_risk_with_llm(risk_value, context_vars):
    prompt = f"""
Eres un analista siguiendo los principios de Ray Dalio.
Riesgo de recesión calculado: {risk_value:.1f}%
Variables recientes:
{context_vars}

Responde en este formato:
- Porcentaje de riesgo: X%
- Explicación: breve, clara y estilo Ray Dalio
"""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama3-70b-8192",
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.7
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/completions", headers=headers, json=payload)
        return response.json()["choices"][0]["text"]
    except Exception as e:
        return f"⚠️ AI error: {e}"

# =========================
# DISPLAY
# =========================
for country, df in all_data.items():
    df = calculate_risk_advanced(df)
    st.subheader(f"{country} Risk Index")
    st.line_chart(df["Risk (%)"])

    latest_risk = df["Risk (%)"].iloc[-1]
    context_vars = df.tail(1).to_dict(orient="records")[0]

    explanation = explain_risk_with_llm(latest_risk, context_vars)

    st.markdown("### 🤖 AI Risk Assessment")
    st.write(explanation)
    st.dataframe(df.tail(10))
