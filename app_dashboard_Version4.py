import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings('ignore')

import streamlit as st

# ==============================================================================
# PASO 1: INSTALACIÓN Y CARGA DE LIBRERÍAS Y DATOS
# ==============================================================================
st.set_page_config(page_title="Dashboard de Tesis", layout="wide")
st.title("Dashboard de Tesis: Análisis y Simulación")
st.write("Use el selector para cambiar el activo en las pestañas.")

url = 'https://raw.githubusercontent.com/jaimeisaac2020/Python-analsisis-basicos/mi-github/dataset_completo_google_amazon.csv'
df = pd.read_csv(url, index_col='Date', parse_dates=True)
st.success("✅ Datos cargados correctamente.")

# ==============================================================================
# PASO 2: INGENIERÍA DE CARACTERÍSTICAS
# ==============================================================================
df_eng = df.copy()
for asset in ['GOOGL', 'AMZN']:
    df_eng[f'{asset}_Return_1d'] = df_eng[f'{asset}_Close'].pct_change()
    df_eng[f'{asset}_Volatility_20'] = df_eng[f'{asset}_Return_1d'].rolling(window=20).std()
    for lag in range(1, 6):
        df_eng[f'{asset}_Close_Lag_{lag}'] = df_eng[f'{asset}_Close'].shift(lag)

df_final = df_eng.dropna()
st.success("✅ Ingeniería de características completada.")

# ==============================================================================
# PASO 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ==============================================================================

results = {}
models = {}
X_train_dict = {}
X_test_dict = {}
features_base_all = ['VIX', 'SP500', 'NASDAQ', 'Treasury_10Y', 'Inflacion_T5YIE']
arima_predictions = {}

for asset in ['GOOGL', 'AMZN']:
    results[asset] = {}
    models[asset] = {}

    train_size = int(len(df_final) * 0.8)
    train_df, test_df = df_final.iloc[:train_size], df_final.iloc[train_size:]
    y_train, y_test = train_df[f'{asset}_Close'], test_df[f'{asset}_Close']

    features_macro_base = ['VIX', 'SP500', 'NASDAQ']
    rf_base = RandomForestRegressor(random_state=42).fit(train_df[features_macro_base], y_train)
    preds = rf_base.predict(test_df[features_macro_base])
    results[asset]['RF_Base_Macro'] = {'MSE': mean_squared_error(y_test, preds), 'R2': r2_score(y_test, preds)}

    features_con_tasa = features_macro_base + ['Treasury_10Y']
    rf_con_tasa = RandomForestRegressor(random_state=42).fit(train_df[features_con_tasa], y_train)
    preds = rf_con_tasa.predict(test_df[features_con_tasa])
    results[asset]['RF_Con_Tasa'] = {'MSE': mean_squared_error(y_test, preds), 'R2': r2_score(y_test, preds)}

    features_con_inflacion = features_con_tasa + ['Inflacion_T5YIE']
    rf_con_inf = RandomForestRegressor(random_state=42).fit(train_df[features_con_inflacion], y_train)
    preds = rf_con_inf.predict(test_df[features_con_inflacion])
    results[asset]['RF_Con_Inflacion'] = {'MSE': mean_squared_error(y_test, preds), 'R2': r2_score(y_test, preds)}

    features_enriched = [f for f in df_final.columns if f.startswith(asset) or f in features_base_all]
    features_enriched = [f for f in features_enriched if f not in [f'{asset}_Close', f'{asset}_Return_1d']]
    X_train_enriched, y_train_enriched = train_df[features_enriched], train_df[f'{asset}_Close']
    X_test_enriched, y_test_enriched = test_df[features_enriched], test_df[f'{asset}_Close']
    X_train_dict[asset], X_test_dict[asset] = X_train_enriched, X_test_enriched

    models[asset]['Random Forest'] = RandomForestRegressor(random_state=42).fit(X_train_enriched, y_train_enriched)
    models[asset]['Gradient Boosting'] = GradientBoostingRegressor(random_state=42).fit(X_train_enriched, y_train_enriched)
    models[asset]['Regresión Lineal'] = LinearRegression().fit(X_train_enriched, y_train_enriched)

    for name, model in models[asset].items():
        preds = model.predict(X_test_enriched)
        results[asset][f'{name}_Enriquecido'] = {'MSE': mean_squared_error(y_test_enriched, preds), 'R2': r2_score(y_test_enriched, preds)}

    arima_order = (1, 1, 1) if asset == 'GOOGL' else (0, 1, 0)
    arima_model = ARIMA(y_train_enriched, order=arima_order).fit()
    arima_predictions[asset] = arima_model.forecast(steps=len(y_test_enriched))
    results[asset]['ARIMA_Enriquecido'] = {'MSE': mean_squared_error(y_test_enriched, arima_predictions[asset]), 'R2': r2_score(y_test_enriched, arima_predictions[asset])}
    models[asset]['ARIMA_pred_estatica'] = arima_predictions[asset].iloc[0]

st.success("✅ Todos los modelos han sido entrenados y evaluados.")

# ==============================================================================
# FUNCIONES DE PESTAÑAS STREAMLIT
# ==============================================================================

def tab_exploration(df):
    st.subheader("Primeros y Últimos 5 registros")
    st.write(df.head())
    st.write(df.tail())
    st.subheader("Gráficos de Series de Tiempo")
    for col in df.columns:
        if col not in ['GOOGL_Return_1d', 'AMZN_Return_1d', 'GOOGL_Volatility_20', 'AMZN_Volatility_20']:
            fig, ax = plt.subplots(figsize=(8, 2))
            df[col].plot(ax=ax, title=col)
            st.pyplot(fig)

def tab_objective1(asset):
    res_base = results[asset]['RF_Base_Macro']
    res_completo = results[asset]['RF_Con_Tasa']
    df_res = pd.DataFrame([res_base, res_completo], index=['Modelo Base', 'Modelo con Tasa Int.'])
    st.subheader("Validación H2: Impacto de Tasas de Interés")
    st.write(df_res.style.format('{:.4f}').highlight_min(subset='MSE', color='#d4edda'))
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(df_final.index, df_final[f'{asset}_Close'], color='blue', label=f'Precio {asset}')
    ax1.set_ylabel(f'Precio {asset} (USD)', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(df_final.index, df_final['Treasury_10Y'], color='red', linestyle='--', label='Tasa 10Y')
    ax2.set_ylabel('Tasa Treasury 10Y (%)', color='red')
    plt.title(f'Contexto: Precio de {asset} vs. Tasa de Interés')
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(df_res.index, df_res['MSE'], color=['gray', 'skyblue'])
    ax.set_title(f'Reducción de Error (MSE)')
    ax.bar_label(bars, fmt='{:,.2f}')
    st.pyplot(fig)

def tab_objective2(asset):
    res_sin = results[asset]['RF_Con_Tasa']
    res_con = results[asset]['RF_Con_Inflacion']
    df_res = pd.DataFrame([res_sin, res_con], index=['Modelo SIN Inflación', 'Modelo CON Inflación'])
    incremento_error = (res_sin['MSE'] / res_con['MSE'] - 1) * 100 if res_con['MSE'] > 0 else 0
    st.subheader("Validación H3: Impacto de la Inflación")
    st.write(df_res.style.format('{:.4f}').highlight_min(subset='MSE', color='#d4edda'))
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(df_final.index, df_final[f'{asset}_Close'], color='blue', label=f'Precio {asset}')
    ax1.set_ylabel(f'Precio {asset} (USD)', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(df_final.index, df_final['Inflacion_T5YIE'], color='green', linestyle='--', label='Inflación 5A')
    ax2.set_ylabel('Expectativa Inflación 5A (%)', color='green')
    plt.title(f'Contexto: Precio de {asset} vs. Expectativas de Inflación')
    st.pyplot(fig)
    st.info(f"Eliminar la inflación aumentó el error en un {incremento_error:.2f}% para {asset}.")

def tab_objective3(asset):
    res_base = results[asset]['RF_Base_Macro']
    # CORREGIDO: Usar el nombre correcto de la clave enriquecida
    res_enriquecido = results[asset]['Random Forest_Enriquecido']
    df_res = pd.DataFrame([res_base, res_enriquecido], index=['Modelo SIN Históricos', 'Modelo CON Históricos'])
    r2_color = 'red' if df_res.loc['Modelo CON Históricos', 'R2'] < 0 else 'green'
    if asset == 'AMZN':
        st.success(f"¡Resultado Clave para {asset}! El R² pasó de {res_base['R2']:.2f} a {res_enriquecido['R2']:.2f}")
    st.subheader("Validación H4: Impacto de Datos Históricos")
    st.write(df_res.style.format('{:.4f}').highlight_min(subset='MSE', color='#d4edda'))

def tab_h1_validation(asset):
    asset_results = {
        'ARIMA': results[asset]['ARIMA_Enriquecido'],
        'Random Forest': results[asset]['Random Forest_Enriquecido'],
        'Gradient Boosting': results[asset]['Gradient Boosting_Enriquecido']
    }
    df_res = pd.DataFrame(asset_results).T
    reduction_vs_arima = (1 - df_res.loc['Random Forest', 'MSE'] / df_res.loc['ARIMA', 'MSE']) * 100
    st.header("Validación Hipótesis 1: Superioridad de Modelos Supervisados")
    st.write(df_res.style.format('{:.4f}').highlight_min(subset='MSE', color='#d4edda'))
    st.success(f"Random Forest reduce el error (MSE) en un {reduction_vs_arima:.2f}% en comparación con ARIMA para {asset}.")
    fig, ax = plt.subplots(figsize=(10, 5))
    test_size = int(len(df_final) * 0.8)
    test_df = df_final.iloc[test_size:]
    test_dates = test_df.index
    ax.plot(test_dates, test_df[f'{asset}_Close'], label='Precio Real', color='black', linewidth=2)
    ax.plot(test_dates, arima_predictions[asset], label='ARIMA (Tradicional)', color='red', linestyle='--')
    ax.plot(test_dates, models[asset]['Random Forest'].predict(X_test_dict[asset]), label='Random Forest (Supervisado)', color='blue', linestyle='--')
    ax.set_title(f'Comparación Visual de Predicciones para {asset}')
    ax.legend()
    st.pyplot(fig)
    st.info("El gráfico muestra cómo ARIMA (rojo) falla en capturar la tendencia, mientras que Random Forest (azul) se adapta mejor al precio real.")

def tab_interpretability(asset):
    st.header("Interpretabilidad del Modelo Random Forest")
    model = models[asset]['Random Forest']
    X_test = X_test_dict[asset]
    X_train = X_train_dict[asset]
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    st.subheader("Gráfico SHAP")
    fig_shap = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    plt.title(f'SHAP Summary Plot para {asset}')
    st.pyplot(fig_shap)
    # PDP
    st.subheader("Partial Dependence Plots (PDP)")
    features_to_plot = ['Treasury_10Y', 'Inflacion_T5YIE', 'VIX', f'{asset}_Close_Lag_1']
    fig_pdp, ax = plt.subplots(1, len(features_to_plot), figsize=(15, 3))
    PartialDependenceDisplay.from_estimator(model, X_train, features_to_plot, ax=ax)
    plt.suptitle(f'Partial Dependence Plots (PDP) para {asset}')
    st.pyplot(fig_pdp)

def tab_simulador():
    asset = st.selectbox("Seleccione el Activo:", ['AMZN', 'GOOGL'])
    tasa_interes = st.slider("Tasa Interés:", min_value=2.0, max_value=6.0, value=4.0, step=0.05)
    inflacion = st.slider("Inflación:", min_value=1.5, max_value=4.0, value=2.5, step=0.05)
    vix = st.slider("VIX:", min_value=10, max_value=40, value=15, step=1)
    precio_anterior = st.slider("Precio Anterior:", min_value=150.0, max_value=250.0, value=float(df_final['AMZN_Close'].iloc[-1]), step=0.5)
    last_known = df_final.iloc[-1].copy()
    input_data = pd.DataFrame([last_known])
    input_data['Treasury_10Y'] = tasa_interes
    input_data['Inflacion_T5YIE'] = inflacion
    input_data['VIX'] = vix
    input_data[f'{asset}_Close_Lag_1'] = precio_anterior
    for lag in range(2, 6):
        input_data[f'{asset}_Close_Lag_{lag}'] = precio_anterior
    features_enriched = [f for f in df_final.columns if f.startswith(asset) or f in features_base_all]
    features_enriched = [f for f in features_enriched if f not in [f'{asset}_Close', f'{asset}_Return_1d']]
    input_data_final = input_data[features_enriched]
    preds = {name: model.predict(input_data_final)[0] for name, model in models[asset].items() if name != 'ARIMA_pred_estatica'}
    preds['ARIMA'] = models[asset]['ARIMA_pred_estatica']
    st.subheader("Tabla de Predicciones")
    table_pred = pd.DataFrame.from_dict(preds, orient='index', columns=['Precio Predicho'])
    st.write(table_pred)
    st.subheader("Gráfico de Predicciones")
    historic_data = df_final[f'{asset}_Close'].tail(100)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(historic_data.index, historic_data.values, label=f'Histórico de {asset}', color='black', alpha=0.7)
    last_date = historic_data.index[-1]
    prediction_date = last_date + pd.Timedelta(days=1)
    colors = {'ARIMA': 'red', 'Regresión Lineal': 'orange', 'Gradient Boosting': 'purple', 'Random Forest': 'blue'}
    best_model_name = "Random Forest"
    for model_name, prediction in preds.items():
        is_best = (model_name == best_model_name)
        ax.plot([last_date, prediction_date], [precio_anterior, prediction], linestyle='--', marker='o', color=colors[model_name],
                label=f'{model_name}: {prediction:,.2f}', linewidth=3.0 if is_best else 1.5, alpha=1.0 if is_best else 0.8)
    ax.set_title(f'Pronósticos para {asset} bajo Escenario Simulado', fontsize=16)
    ax.set_ylabel('Precio (USD)')
    ax.legend(loc='best')
    st.pyplot(fig)

# ==============================================================================
# STREAMLIT TABS
# ==============================================================================

tabs = st.tabs([
    "0. Exploración Datos",
    "1. Validación H1",
    "2. Impacto Tasas Int.",
    "3. Impacto Inflación",
    "4. Impacto D. Históricos",
    "5. Interpretabilidad",
    "6. DEMO INTERACTIVA"
])

with tabs[0]:
    tab_exploration(df)

asset = st.sidebar.selectbox("Activo para Análisis (Tabs 1-5):", ['AMZN', 'GOOGL'])

with tabs[1]:
    tab_h1_validation(asset)
with tabs[2]:
    tab_objective1(asset)
with tabs[3]:
    tab_objective2(asset)
with tabs[4]:
    tab_objective3(asset)
with tabs[5]:
    tab_interpretability(asset)
with tabs[6]:
    tab_simulador()