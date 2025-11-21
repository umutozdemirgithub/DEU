import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import shap
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(page_title="XAI Solar Forecasting", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'res_store' not in st.session_state:
    st.session_state['res_store'] = {}
if 'trained' not in st.session_state:
    st.session_state['trained'] = False

# --- 1. HELPER FUNCTIONS ---

def st_shap(plot, height=None):
    """Helper function to display SHAP JS plots in Streamlit"""
    shap_html = f"{shap.getjs()}{plot.html()}"
    components.html(shap_html, height=height)

def apply_outlier_handling(df, cols, methods):
    df_clean = df.copy()
    if "IQR Capping" in methods:
        for col in cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
            
    if "Z-Score Capping" in methods:
        for col in cols:
            z = stats.zscore(df_clean[col])
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            upper = mean_val + 3 * std_val
            lower = mean_val - 3 * std_val
            df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
            df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
            
    if "Isolation Forest (Drop)" in methods:
        iso = IsolationForest(contamination=0.05, random_state=42)
        yhat = iso.fit_predict(df_clean[cols])
        mask = yhat != -1
        df_clean = df_clean[mask]

    return df_clean

def apply_scaling(df, cols, methods):
    df_scaled = df.copy()
    if "Log Transformation (np.log1p)" in methods:
        for col in cols:
            if (df_scaled[col] >= 0).all():
                df_scaled[col] = np.log1p(df_scaled[col])

    if "Min-Max Scaling (0-1)" in methods:
        df_scaled[cols] = MinMaxScaler().fit_transform(df_scaled[cols])
    if "Standard Scaling (Z-Score)" in methods:
        df_scaled[cols] = StandardScaler().fit_transform(df_scaled[cols])
    if "Robust Scaling (IQR based)" in methods:
        df_scaled[cols] = RobustScaler().fit_transform(df_scaled[cols])
    if "MaxAbs Scaling (-1 to 1)" in methods:
        df_scaled[cols] = MaxAbsScaler().fit_transform(df_scaled[cols])
        
    return df_scaled

def perform_hpo(X_train, y_train, method):
    base_model = HistGradientBoostingRegressor(random_state=42)
    
    # --- ZENGİNLEŞTİRİLMİŞ PARAMETRE UZAYLARI ---
    # Random Search için Olasılık Dağılımları (Daha geniş ve esnek)
    random_space = {
        'learning_rate': stats.loguniform(0.001, 0.5), # Logaritmik dağılım (önemli)
        'max_iter': stats.randint(100, 1000),          # Ağaç sayısı
        'max_leaf_nodes': stats.randint(15, 127),      # Ağaç karmaşıklığı
        'max_depth': [None, 5, 10, 20, 30],            # Derinlik sınırı
        'min_samples_leaf': stats.randint(10, 100),    # Yaprak başına min örnek (Overfitting önler)
        'l2_regularization': stats.uniform(0.0, 10.0), # L2 Regularizasyon
        'max_bins': [50, 100, 255]                     # Veri ayrıştırma hassasiyeti
    }
    
    # Grid Search için Sabit Listeler (Kombinasyon sayısı kontrollü tutuldu)
    grid_space = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [200, 500, 1000],
        'max_leaf_nodes': [31, 63],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [20, 50],
        'l2_regularization': [0.0, 0.1, 1.0]
    }

    if method == "Random Search":
        # n_iter artırıldı (10 -> 30) daha fazla deneme yapması için
        search = RandomizedSearchCV(
            base_model, 
            random_space, 
            n_iter=30, 
            cv=3, 
            scoring='neg_mean_squared_error', 
            random_state=42,
            n_jobs=-1
        )
    elif method == "Grid Search":
        search = GridSearchCV(
            base_model, 
            grid_space, 
            cv=3, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
    else:
        return base_model
        
    with st.spinner(f"Optimizing ({method})... This may take a moment."):
        search.fit(X_train, y_train)
        
    return search.best_estimator_

def train_and_evaluate(X, y, test_size, hpo_method=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if hpo_method:
        model = perform_hpo(X_train, y_train, hpo_method)
    else:
        model = HistGradientBoostingRegressor(random_state=42)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MSE': mse
    }
    return metrics, model, X_train, X_test, y_test, y_pred

# --- 2. MAIN UI ---
st.title("☀️ Explainable AI Assisted Solar Energy Forecasting")
st.markdown("**Based on the research:** *Explainable Artificial Intelligence Assisted Solar Energy Forecasting*")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("👋 Please upload a CSV file.")
    st.stop()

try:
    df_original = pd.read_csv(uploaded_file)
    st.success("Data Loaded.")

    st.sidebar.subheader("1.1 Date Column")
    date_cols = [c for c in df_original.columns if 'date' in c.lower() or 'time' in c.lower()]
    idx = (list(df_original.columns).index(date_cols[0]) + 1) if date_cols else 0
    date_col = st.sidebar.selectbox("Date Column", [None] + list(df_original.columns), index=idx)
    
    if date_col:
        df_original[date_col] = pd.to_datetime(df_original[date_col])
        df_original.set_index(date_col, inplace=True)

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- CONFIGURATION ---
st.sidebar.header("2. Configuration")
cols = df_original.columns.tolist()
target_def = 'Power' if 'Power' in cols else cols[-1]
target_col = st.sidebar.selectbox("Target", cols, index=cols.index(target_def))

feats_def = [c for c in cols if c != target_col]
feature_cols = st.sidebar.multiselect("Features", cols, default=feats_def)

if not feature_cols:
    st.stop()

st.sidebar.header("3. Preprocessing")
outlier_methods = st.sidebar.multiselect("Outlier Methods", ["IQR Capping", "Z-Score Capping", "Isolation Forest (Drop)"])
scaling_methods = st.sidebar.multiselect("Scaling Methods", ["Min-Max Scaling (0-1)", "Standard Scaling (Z-Score)", "Robust Scaling (IQR based)", "MaxAbs Scaling (-1 to 1)", "Log Transformation (np.log1p)"])

st.sidebar.header("4. HPO & Analysis")
hpo_methods = st.sidebar.multiselect("HPO Methods", ["Random Search", "Grid Search"])

# --- YARDIM BÖLÜMÜ (YENİ) ---
if hpo_methods:
    with st.expander("ℹ️ HPO Parametreleri Hakkında Bilgi"):
        st.markdown("""
        `HistGradientBoostingRegressor` için kullanılan temel HPO parametreleri ve anlamları:
        
        | Parametre | Açıklama |
        | :--- | :--- |
        | **`learning_rate`** | Modelin her iterasyonda ne kadar hızlı öğrendiğini kontrol eder. Düşük değerler daha sağlam modeller üretir. |
        | **`max_iter`** | Oluşturulacak ağaç sayısıdır. Modelin karmaşıklığını ve öğrenme süresini belirler. |
        | **`max_leaf_nodes`** | Her bir ağaçtaki maksimum yaprak sayısıdır. Ağaçların karmaşıklığını doğrudan kontrol eder. |
        | **`max_depth`**... | Ağaçların maksimum derinliğini sınırlar. Aşırı öğrenmeyi (overfitting) engellemeye yardımcı olur. |
        | **`min_samples_leaf`** | Bir yaprağın (leaf) geçerli sayılması için gereken minimum örnek (veri noktası) sayısıdır. |
        | **`l2_regularization`** | Aşırı öğrenmeyi engellemek için kullanılan bir ceza (penalty) terimidir. |
        | **`max_bins`** | Özelliklerin (features) kaç farklı gruba ayrılacağını belirler. Modelin hızını ve doğruluğunu etkiler. |
        
        **Not:** `Random Search` daha geniş bir aralıkta rastgele arama yaparken, `Grid Search` belirlediğiniz spesifik değerleri dener.
        """)

st.sidebar.header("5. XAI")
run_shap = st.sidebar.checkbox("Run SHAP Analysis?", value=False)

st.sidebar.header("6. Visualization Settings")
show_combined = st.sidebar.checkbox("Show Combined Forecast Plot", value=True)
show_individual = st.sidebar.checkbox("Show Individual Forecast Plots", value=True)
show_diagnostic = st.sidebar.checkbox("Show Diagnostic Plots", value=True)

# --- TRAINING ---
test_size = st.sidebar.slider("Test Split", 0.1, 0.5, 0.2)

if st.sidebar.button("Train & Compare All Scenarios"):
    with st.spinner("Training..."):
        res_store = {}
        
        # 1. Default
        X_def = df_original[feature_cols]
        y_def = df_original[target_col]
        m, mod, Xt, Xte, yte, ypr = train_and_evaluate(X_def, y_def, test_size)
        res_store['Default'] = {'metrics': m, 'model': mod, 'Xt': Xt, 'Xte': Xte, 'yte': yte, 'ypr': ypr}
        
        # 2. Preprocessed
        df_b = apply_outlier_handling(df_original, feature_cols, outlier_methods) if outlier_methods else df_original.copy()
        df_p = apply_scaling(df_b, feature_cols, scaling_methods) if scaling_methods else df_b
        
        label_p = "Processed" if (outlier_methods or scaling_methods) else None
        
        if label_p:
            Xp, yp = df_p[feature_cols], df_p[target_col]
            m, mod, Xt, Xte, yte, ypr = train_and_evaluate(Xp, yp, test_size)
            res_store[label_p] = {'metrics': m, 'model': mod, 'Xt': Xt, 'Xte': Xte, 'yte': yte, 'ypr': ypr}
            
        # 3. HPO
        data_hpo = df_p if label_p else df_original
        for hpo in hpo_methods:
            lbl = f"HPO ({hpo})"
            if label_p:
                lbl = f"Processed + {lbl}"
            
            Xh, yh = data_hpo[feature_cols], data_hpo[target_col]
            m, mod, Xt, Xte, yte, ypr = train_and_evaluate(Xh, yh, test_size, hpo)
            res_store[lbl] = {'metrics': m, 'model': mod, 'Xt': Xt, 'Xte': Xte, 'yte': yte, 'ypr': ypr}
            
        st.session_state['res_store'] = res_store
        st.session_state['trained'] = True

# --- RESULTS ---
if st.session_state['trained']:
    results = st.session_state['res_store']
    
    # --- METRICS & PARAMS (Redesigned) ---
    st.markdown("---")
    st.subheader("📊 Model Performance & Configuration Report")
    
    # 1. Verileri Hazırla
    m_data = []
    p_data = []
    
    # Yeni parametreleri listeye ekleyin
    k_params = [
        'learning_rate', 'max_iter', 'max_leaf_nodes', 'max_depth', 
        'min_samples_leaf', 'l2_regularization', 'max_bins'
    ]

    for n, r in results.items():
        # Metrik Verisi
        me = r['metrics']
        m_data.append({
            "Scenario": n,
            "RMSE": me['RMSE'],
            "MAE": me['MAE'],
            "R2": me['R2'],
            "MSE": me['MSE']
        })
        
        # Parametre Verisi
        p = r['model'].get_params()
        row = {"Scenario": n}
        for k in k_params:
            row[k] = p.get(k, '-')
        p_data.append(row)

    df_metrics = pd.DataFrame(m_data).set_index("Scenario")
    df_params = pd.DataFrame(p_data).set_index("Scenario")

    # 2. Sekmeli Görünüm
    tab_leaderboard, tab_params, tab_visuals = st.tabs([
        "🏆 Leaderboard (Metrics)", 
        "⚙️ Hyperparameters",
        "📈 Visual Comparison"
    ])
    
    with tab_leaderboard:
        st.markdown("Modellerin performans karşılaştırması. **R²** için yüksek, **Hata (Error)** metrikleri için düşük değerler iyidir.")
        # Pandas Styler ile Renklendirme (Highlight Best)
        st.dataframe(
            df_metrics.style
            .highlight_max(subset=['R2'], color='#d1e7dd') # En iyi R2 Yeşil
            .highlight_min(subset=['RMSE', 'MAE', 'MSE'], color='#cff4fc') # En düşük Hata Mavi
            .format("{:.4f}"),
            use_container_width=True
        )

    with tab_params:
        st.markdown("Modellerin eğitiminde kullanılan hiperparametreler.")
        st.dataframe(df_params, use_container_width=True)
        
        # HATA BURADAYDI: Bu blok 'if st.session_state['trained']:' içine taşındı.
        with st.expander("🔍 See Full Raw Parameters (JSON)"):
            for n, r in results.items():
                st.text(f"{n}: {r['model'].get_params()}")

    with tab_visuals:
        st.markdown("#### ⚔️ Model Performance Benchmark")
        fig = go.Figure()
        # --- RMSE Bar ---
        fig.add_trace(go.Bar(
            x=df_metrics.index, 
            y=df_metrics["RMSE"], 
            name="RMSE", 
            marker=dict(color="rgba(255, 99, 71, 0.75)"), # soft tomato
            hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}"
        ))
        # --- R2 Line ---
        fig.add_trace(go.Scatter(
            x=df_metrics.index, 
            y=df_metrics["R2"], 
            name="R² Score", 
            yaxis="y2", 
            mode="lines+markers", 
            marker=dict(color="rgba(34, 139, 34, 0.9)", size=10), # forest green
            line=dict(width=3)
        ))
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Scenario",
            yaxis=dict(title="RMSE (Lower is Better)", side="left"),
            yaxis2=dict(title="R² Score (Higher is Better)", side="right", overlaying="y", range=[0, 1.1]),
            legend=dict(orientation="h", y=1.1),
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- FORECAST PLOTS ---
    st.markdown("---")
    st.subheader("📉 Forecast & Analysis")
    
    model_names = list(results.keys())
    selected_model_name = st.selectbox("Select Scenario to Analyze", model_names)
    
    if selected_model_name:
        res = results[selected_model_name]
        model = res['model']
        X_test = res['Xte']
        y_test = res['yte']
        y_pred = res['ypr']
        
        # 1. Combined Plot
        if show_combined:
            st.markdown("#### Actual vs Predicted (Combined)")
            fig_comb = go.Figure()
            # Gerçek
            fig_comb.add_trace(go.Scatter(
                y=y_test.values, 
                mode='lines', 
                name='Actual', 
                line=dict(color='black', width=2),
                opacity=0.6
            ))
            # Tahmin
            fig_comb.add_trace(go.Scatter(
                y=y_pred, 
                mode='lines', 
                name='Predicted', 
                line=dict(color='#007bff', width=2)
            ))
            fig_comb.update_layout(
                title=f"Actual vs Predicted: {selected_model_name}",
                xaxis_title="Samples",
                yaxis_title="Power",
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig_comb, use_container_width=True)

        # 2. Diagnostic Plots
        if show_diagnostic:
            st.markdown("#### Model Diagnostics")
            diag_tabs = st.tabs(["Actual vs Predicted", "Residual Distribution", "Residuals vs Predicted"])
            
            with diag_tabs[0]:
                # Scatter Actual vs Predicted
                fig_scat = px.scatter(
                    x=y_test, y=y_pred, 
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title=f"Actual vs Predicted ({selected_model_name})",
                    trendline="ols",
                    trendline_color_override="red"
                )
                st.plotly_chart(fig_scat, use_container_width=True)
                
            with diag_tabs[1]:
                # Residuals Distribution
                residuals = y_test - y_pred
                fig_res = px.histogram(
                    residuals, 
                    nbins=50, 
                    title=f"Residuals Distribution ({selected_model_name})",
                    labels={'value': 'Residual Error'},
                    color_discrete_sequence=['#6c757d']
                )
                st.plotly_chart(fig_res, use_container_width=True)
                
            with diag_tabs[2]:
                # Residuals vs Predicted
                fig_rvp = px.scatter(
                    x=y_pred, y=residuals,
                    labels={'x': 'Predicted', 'y': 'Residuals'},
                    title=f"Residuals vs Predicted ({selected_model_name})"
                )
                fig_rvp.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_rvp, use_container_width=True)

        # 3. SHAP
        if run_shap:
            st.markdown("---")
            st.subheader("🤖 XAI: SHAP Explanation")
            
            with st.spinner("Calculating SHAP values..."):
                try:
                    # TreeExplainer HistGradientBoosting ile bazen uyumsuz olabilir, KernelExplainer fallback
                    # Ancak HistGradientBoosting scikit-learn sürümüne göre değişir.
                    # Genel yaklaşım:
                    explainer = shap.Explainer(model, X_test)
                    shap_values = explainer(X_test)

                    # Tabs for SHAP
                    shap_tab1, shap_tab2, shap_tab3 = st.tabs(["Summary (Beeswarm)", "Feature Importance (Bar)", "Feature Dependence"])
                    
                    with shap_tab1:
                        st.markdown("Features pushing the prediction higher (red) or lower (blue).")
                        # SHAP Beeswarm Plot
                        fig, ax = plt.subplots()
                        shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(fig)
                        
                    with shap_tab2:
                        st.markdown("Global importance of each feature.")
                        # SHAP Bar Plot
                        fig, ax = plt.subplots()
                        shap.plots.bar(shap_values, show=False)
                        st.pyplot(fig)
                        
                    with shap_tab3:
                        st.markdown("Scatter plot of feature effect vs feature value.")
                        # En önemli özelliği bul
                        top_feature = X_test.columns[0] # Basitçe ilkini al veya logic ekle
                        fig, ax = plt.subplots()
                        shap.plots.scatter(shap_values[:, top_feature], color=shap_values, show=False)
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.warning(f"SHAP calculation failed: {e}")
                    st.info("Note: Some complex models or data structures might cause issues with SHAP directly.")

else:
    st.info("👈 Please upload data and click 'Train & Compare' in the sidebar to start.")
