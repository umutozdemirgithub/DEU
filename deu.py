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
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
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
            if (df_scaled[col] >= 0).all(): df_scaled[col] = np.log1p(df_scaled[col])
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
        'learning_rate': stats.loguniform(0.001, 0.5),  # Logaritmik dağılım (önemli)
        'max_iter': stats.randint(100, 1000),           # Ağaç sayısı
        'max_leaf_nodes': stats.randint(15, 127),       # Ağaç karmaşıklığı
        'max_depth': [None, 5, 10, 20, 30],             # Derinlik sınırı
        'min_samples_leaf': stats.randint(10, 100),     # Yaprak başına min örnek (Overfitting önler)
        'l2_regularization': stats.uniform(0.0, 10.0),  # L2 Regularizasyon
        'max_bins': [50, 100, 255]                      # Veri ayrıştırma hassasiyeti
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
            base_model, random_space, n_iter=30, cv=3, 
            scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
        )
    elif method == "Grid Search":
        search = GridSearchCV(
            base_model, grid_space, cv=3, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
    else:
        return base_model

    with st.spinner(f"Optimizing ({method})... This may take a moment."):
        search.fit(X_train, y_train)
        
    return search.best_estimator_


def train_and_evaluate(X, y, test_size, hpo_method=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if hpo_method: model = perform_hpo(X_train, y_train, hpo_method)
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
if not feature_cols: st.stop()

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
        | **`max_depth`** | Ağaçların maksimum derinliğini sınırlar. Aşırı öğrenmeyi (overfitting) engellemeye yardımcı olur. |
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
            if label_p: lbl = f"Processed + {lbl}"
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
        .highlight_max(subset=['R2'], color='#d1e7dd')  # En iyi R2 Yeşil
        .highlight_min(subset=['RMSE', 'MAE', 'MSE'], color='#cff4fc') # En düşük Hata Mavi
        .format("{:.4f}"),
        use_container_width=True
    )
    
    # # CSV İndirme Butonu
    # csv_metrics = df_metrics.to_csv().encode('utf-8')
    # st.download_button(
    #     label="📥 Download Metrics CSV",
    #     data=csv_metrics,
    #     file_name='model_metrics.csv',
    #     mime='text/csv',
    # )

with tab_params:
    st.markdown("Modellerin eğitiminde kullanılan hiperparametreler.")
    st.dataframe(df_params, use_container_width=True)
    
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
        marker=dict(color="rgba(255, 99, 71, 0.75)"),  # soft tomato
        hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>"
    ))

    # --- MSE Bar ---
    fig.add_trace(go.Bar(
        x=df_metrics.index,
        y=df_metrics["MSE"],
        name="MSE",
        marker=dict(color="rgba(255, 159, 64, 0.75)"),  # pastel orange
        hovertemplate="<b>%{x}</b><br>MSE: %{y:.4f}<extra></extra>"
    ))

    # --- MAE Bar ---
    fig.add_trace(go.Bar(
        x=df_metrics.index,
        y=df_metrics["MAE"],
        name="MAE",
        marker=dict(color="rgba(255, 205, 86, 0.75)"),  # lemon pastel
        hovertemplate="<b>%{x}</b><br>MAE: %{y:.4f}<extra></extra>"
    ))

    # --- R² Line + Marker ---
    fig.add_trace(go.Scatter(
        x=df_metrics.index,
        y=df_metrics["R2"],
        name="R²",
        mode="lines+markers",
        marker=dict(size=13, color="#007aff", line=dict(width=2, color="white")),  # iOS blue
        line=dict(width=4, color="#007aff"),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>"
    ))

    # --- Layout: Apple Minimal Theme ---
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="📊 Model Performance Overview ",
            x=0.5,
            font=dict(size=22, color="#333", family="Helvetica Neue")
        ),
        xaxis=dict(
            title="Scenario",
            tickfont=dict(size=12, family="Helvetica Neue"),
            showgrid=False
        ),
        yaxis=dict(
            title="RMSE / MSE / MAE",
            titlefont=dict(size=14, color="#555", family="Helvetica Neue"),
            tickfont=dict(size=12, color="#666", family="Helvetica Neue"),
            gridcolor="rgba(0,0,0,0.07)"
        ),
        yaxis2=dict(
            title="R²",
            overlaying="y",
            side="right",
            range=[0, 1],
            titlefont=dict(size=14, color="#007aff"),
            tickfont=dict(size=12, color="#007aff"),
        ),
        barmode="group",
        bargap=0.25,
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(
            orientation="h",
            x=0.02, y=1.07,
            font=dict(size=12, family="Helvetica Neue")
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Analysis ---
    best_r2_model = df_metrics["R2"].idxmax()
    best_r2_val = df_metrics["R2"].max()
    best_rmse_model = df_metrics["RMSE"].idxmin()
    best_mse_model = df_metrics["MSE"].idxmin()
    best_mae_model = df_metrics["MAE"].idxmin()

    st.info(f"""
    **💡 Otomatik Analiz:**
    * **En yüksek R²:** `{best_r2_model}` (R² = **{best_r2_val:.4f}**)
    * **En düşük RMSE:** `{best_rmse_model}`
    * **En düşük MSE:** `{best_mse_model}`
    * **En düşük MAE:** `{best_mae_model}`
    
    **📌 Genel öneri:**  
    *Güvenlik kritik durumlarda RMSE & MAE daha önemlidir,  
    genel performans değerlendirmede ise R² belirleyicidir.*
    """)

        
    # --- VISUALIZATION ---
    st.subheader("Analysis")
    scenarios = list(results.keys())
    
    # Ana Tablar (Combined + Senaryolar)
    main_tabs_labels = []
    if show_combined: main_tabs_labels.append("📈 Combined Plot")
    main_tabs_labels.extend(scenarios)
    
    if not main_tabs_labels:
        st.warning("No plots selected in sidebar.")
    else:
        main_tabs = st.tabs(main_tabs_labels)
        curr_tab_idx = 0
        
        # 1. Combined Plot
        if show_combined:
            with main_tabs[curr_tab_idx]:
                ref = scenarios[0]
                max_p = len(results[ref]['yte'])
                n_pts = st.number_input("Points", 1, max_p, min(48, max_p), key="comb")
                fig = go.Figure()
                
                # Actual
                y_ref = results[ref]['yte'].sort_index().head(n_pts)
                fig.add_trace(go.Scatter(x=y_ref.index, y=y_ref, name="Actual", line=dict(color='black', width=3)))
                
                # Preds
                colors = px.colors.qualitative.Plotly
                for i, (name, r) in enumerate(results.items()):
                    yp = pd.Series(r['ypr'], index=r['yte'].index).sort_index().head(n_pts)
                    fig.add_trace(go.Scatter(x=yp.index, y=yp, name=f"Pred: {name}", line=dict(color=colors[i%len(colors)], dash='dot')))
                
                fig.update_layout(template="plotly_white", title="Combined Forecast")
                st.plotly_chart(fig, use_container_width=True)
            curr_tab_idx += 1
            
        # 2. Scenario Tabs
        for i, sc_name in enumerate(scenarios):
            with main_tabs[curr_tab_idx + i]:
                dat = results[sc_name]
                y_test = dat['yte']
                y_pred = dat['ypr']
                resid = y_test - y_pred
                
                # Alt Tablar (Feature bazlı)
                sub_labels = []
                if show_individual: sub_labels.append("Forecast")
                if show_diagnostic: sub_labels.append("Diagnostics")
                if run_shap: sub_labels.append("XAI (SHAP)")
                
                if not sub_labels:
                    st.info("Select plots from sidebar.")
                else:
                    sub_tabs = st.tabs(sub_labels)
                    s_idx = 0
                    
                    # A. Forecast (Enhanced Interactive)
                    if show_individual:
                        with sub_tabs[s_idx]:
                            # 1. Kontrol Paneli (Slicing Controls)
                            st.markdown("#### 🕹️ Interactive Forecast Explorer")
                            c1, c2 = st.columns([1, 3])
                            
                            total_points = len(y_test)
                            with c1:
                                # Pencere Boyutu (Ne kadar veri göreceğiz?)
                                window_size = st.slider(
                                    f"Window Size ({sc_name})", 
                                    min_value=24, 
                                    max_value=min(1000, total_points), 
                                    value=min(100, total_points), 
                                    step=24,
                                    key=f"win_{i}"
                                )
                            with c2:
                                # Başlangıç Noktası (Veri içinde kaydır)
                                start_idx = st.slider(
                                    f"Start Index (Scroll through time) - {sc_name}", 
                                    min_value=0, 
                                    max_value=max(0, total_points - window_size), 
                                    value=0, 
                                    step=1,
                                    key=f"start_{i}"
                                )
                            
                            # 2. Veriyi Hazırla
                            # Veriyi indexe göre sıralayıp kesiyoruz (Slicing)
                            df_viz = pd.DataFrame({'Act': y_test, 'Pred': y_pred}, index=y_test.index).sort_index()
                            df_slice = df_viz.iloc[start_idx : start_idx + window_size]
                            
                            # 3. Anlık Metrikler (Sadece bu pencere için)
                            slice_mae = mean_absolute_error(df_slice['Act'], df_slice['Pred'])
                            slice_r2 = r2_score(df_slice['Act'], df_slice['Pred'])
                            
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Slice MAE", f"{slice_mae:.3f}", delta_color="inverse")
                            m2.metric("Slice R²", f"{slice_r2:.3f}")
                            m3.markdown(f"**Start Date:** {df_slice.index[0]}")
                            m4.markdown(f"**End Date:** {df_slice.index[-1]}")

                            # 4. Profesyonel Çizim (Subplots)
                            from plotly.subplots import make_subplots # Import'u buraya ekledim, global'e de taşınabilir.
                            
                            fig_forecast = make_subplots(
                                rows=2, cols=1, 
                                shared_xaxes=True, 
                                vertical_spacing=0.1, 
                                row_heights=[0.7, 0.3],
                                subplot_titles=("Actual vs Predicted Trend", "Residuals (Difference)")
                            )
                            
                            # Trace 1: Actual (Gerçek)
                            fig_forecast.add_trace(go.Scatter(
                                x=df_slice.index, y=df_slice['Act'], 
                                name='Actual',
                                mode='lines',
                                line=dict(color='black', width=2)
                            ), row=1, col=1)
                            
                            # Trace 2: Predicted (Tahmin)
                            fig_forecast.add_trace(go.Scatter(
                                x=df_slice.index, y=df_slice['Pred'], 
                                name='Predicted',
                                mode='lines',
                                line=dict(color='#FF8C00', width=2, dash='solid'), # Solar Orange
                                fill='tonexty', # Aradaki farkı boyar (isteğe bağlı)
                                fillcolor='rgba(255, 140, 0, 0.1)' 
                            ), row=1, col=1)
                            
                            # Trace 3: Residuals (Hata Çubukları)
                            residuals = df_slice['Act'] - df_slice['Pred']
                            colors = ['crimson' if val < 0 else 'royalblue' for val in residuals]
                            
                            fig_forecast.add_trace(go.Bar(
                                x=df_slice.index, y=residuals, 
                                name='Residuals',
                                marker_color=colors,
                                opacity=0.8
                            ), row=2, col=1)
                            
                            # Layout Ayarları
                            fig_forecast.update_layout(
                                height=600, 
                                template="plotly_white",
                                hovermode="x unified",
                                xaxis2_title="Date/Time",
                                yaxis_title="Power",
                                yaxis2_title="Error",
                                showlegend=True,
                                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
                            )
                            
                            # Range Slider (Alternatif gezinme)
                            fig_forecast.update_xaxes(rangeslider_visible=False) # Kendi slider'ımız olduğu için kapattım
                            
                            st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            with st.expander("See Raw Data for Selected Window"):
                                st.dataframe(df_slice)

                        s_idx += 1

                        
                    # B. Diagnostics (Advanced & Statistical)
                    if show_diagnostic:
                        with sub_tabs[s_idx]:
                            st.markdown("#### 🩺 Model Diagnostics & Statistical Validation")
                            
                            # Tanısal sekmeleri oluşturuyoruz (6 Sekme)
                            diag_tabs = st.tabs([
                                "🔍 Actual vs Predicted", 
                                "📊 Residual Dist & Q-Q", 
                                "📉 Time Series Check",
                                "🕸️ Residuals vs Predicted",
                                "📦 Error Box Plot",
                                "🧮 Statistical Analysis"
                            ])
                            
                            # 1. Actual vs Predicted (Scatter + Color by Error)
                            with diag_tabs[0]:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    fig_scat = go.Figure()
                                    error_abs = np.abs(y_test - y_pred)
                                    
                                    fig_scat.add_trace(go.Scatter(
                                        x=y_test, y=y_pred,
                                        mode='markers',
                                        name='Data Points',
                                        marker=dict(
                                            color=error_abs,
                                            colorscale='Viridis',
                                            showscale=True,
                                            colorbar=dict(title="Abs Error"),
                                            opacity=0.7,
                                            line=dict(width=0.5, color='DarkSlateGrey')
                                        ),
                                        hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>Error: %{marker.color:.2f}<extra></extra>"
                                    ))
                                    
                                    # Perfect Fit Line
                                    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                                    fig_scat.add_trace(go.Scatter(
                                        x=[min_val, max_val], y=[min_val, max_val],
                                        mode='lines', name='Perfect Fit',
                                        line=dict(color='red', dash='dash', width=3)
                                    ))
                                    
                                    fig_scat.update_layout(
                                        title="Actual vs Predicted (Color by Error Magnitude)",
                                        xaxis_title="Actual Power",
                                        yaxis_title="Predicted Power",
                                        template="plotly_white",
                                        height=500,
                                        hovermode="closest"
                                    )
                                    st.plotly_chart(fig_scat, use_container_width=True)
                                
                                with col2:
                                    st.markdown("#### Quick Metrics")
                                    st.metric("Mean Abs Error", f"{mean_absolute_error(y_test, y_pred):.3f}")
                                    st.metric("Max Error", f"{np.max(error_abs):.3f}")
                                    # MAPE Calculation (Handling division by zero)
                                    mask = y_test != 0
                                    if np.sum(mask) > 0:
                                        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                                        st.metric("MAPE", f"{mape:.2f}%")
                                    else:
                                        st.metric("MAPE", "N/A")

                            # 2. Residual Histogram + Q-Q Plot
                            with diag_tabs[1]:
                                c_hist, c_qq = st.columns(2)
                                
                                # Histogram with KDE
                                with c_hist:
                                    fig_hist = go.Figure()
                                    fig_hist.add_trace(go.Histogram(
                                        x=resid, name='Residuals', histnorm='probability density',
                                        marker=dict(color='rgba(100, 149, 237, 0.7)', line=dict(color='black', width=0.5))
                                    ))
                                    try:
                                        kde = stats.gaussian_kde(resid)
                                        x_range = np.linspace(resid.min(), resid.max(), 200)
                                        fig_hist.add_trace(go.Scatter(
                                            x=x_range, y=kde(x_range), mode='lines', name='KDE', line=dict(color='darkorange', width=3)
                                        ))
                                    except: pass
                                    fig_hist.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
                                    fig_hist.update_layout(title="Residual Distribution", height=500, template="plotly_white")
                                    st.plotly_chart(fig_hist, use_container_width=True)

                                # Q-Q Plot
                                with c_qq:
                                    qq_fig = go.Figure()
                                    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm", plot=None)
                                    qq_fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Residuals'))
                                    qq_fig.add_trace(go.Scatter(
                                        x=osm, y=slope*osm + intercept, mode='lines', name='Normal Fit',
                                        line=dict(color='red', width=2)
                                    ))
                                    qq_fig.update_layout(
                                        title="Q-Q Plot (Normality Check)",
                                        xaxis_title="Theoretical Quantiles",
                                        yaxis_title="Sample Quantiles",
                                        height=500,
                                        template="plotly_white"
                                    )
                                    st.plotly_chart(qq_fig, use_container_width=True)

                            # 3. Time Series Check (Zoomable)
                            with diag_tabs[2]:
                                st.markdown("**Zoomable Forecast Check (Ordered by Index)**")
                                # Veriyi indexe göre sıralayıp (eğer zaman ise) çizelim
                                df_ts = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).sort_index()
                                
                                fig_ts = go.Figure()
                                fig_ts.add_trace(go.Scatter(x=df_ts.index, y=df_ts['Actual'], name='Actual', line=dict(color='black', width=1)))
                                fig_ts.add_trace(go.Scatter(x=df_ts.index, y=df_ts['Predicted'], name='Predicted', line=dict(color='red', width=1, dash='dot')))
                                
                                fig_ts.update_layout(
                                    title="Actual vs Predicted Over Time (Zoom In/Out)",
                                    xaxis_title="Date/Index",
                                    yaxis_title="Power",
                                    height=500,
                                    template="plotly_white",
                                    xaxis_rangeslider_visible=True # Alt tarafa mini slider ekler
                                )
                                st.plotly_chart(fig_ts, use_container_width=True)

                                # Residuals over Time
                                fig_res_time = go.Figure()
                                fig_res_time.add_trace(go.Scatter(
                                    x=df_ts.index, 
                                    y=df_ts['Actual'] - df_ts['Predicted'],
                                    mode='markers',
                                    marker=dict(size=4, color='gray'),
                                    name='Residuals'
                                ))
                                fig_res_time.add_hline(y=0, line_color='red')
                                fig_res_time.update_layout(
                                    title="Residuals Over Time (Homoscedasticity Check)",
                                    height=350,
                                    template="plotly_white"
                                )
                                st.plotly_chart(fig_res_time, use_container_width=True)

                            # 4. Residuals vs Predicted
                            with diag_tabs[3]:
                                fig_res = go.Figure()
                                fig_res.add_trace(go.Scatter(
                                    x=y_pred, y=resid, mode='markers',
                                    marker=dict(color=resid, colorscale='RdBu', cmid=0, showscale=True, opacity=0.8)
                                ))
                                fig_res.add_hline(y=0, line_color="black", line_width=2, line_dash="dash")
                                fig_res.update_layout(
                                    title="Residuals vs Predicted",
                                    xaxis_title="Predicted Values",
                                    yaxis_title="Residuals",
                                    template="plotly_white",
                                    height=500
                                )
                                st.plotly_chart(fig_res, use_container_width=True)

                            # 5. Error Box Plot
                            with diag_tabs[4]:
                                fig_box = go.Figure()
                                fig_box.add_trace(go.Box(y=resid, name="Residuals", boxmean='sd', marker_color='indianred'))
                                fig_box.add_trace(go.Box(y=error_abs, name="Abs Errors", boxmean='sd', marker_color='lightseagreen'))
                                fig_box.update_layout(
                                    title="Error Distribution (Box Plot)",
                                    yaxis_title="Error Value",
                                    template="plotly_white",
                                    height=500
                                )
                                st.plotly_chart(fig_box, use_container_width=True)

                            # 6. Statistical Analysis Section
                            with diag_tabs[5]:
                                st.markdown("### 🧬 Statistical Validation of Residuals")
                                
                                # 1. Normality Tests
                                shapiro_stat, shapiro_p = stats.shapiro(resid[:5000]) 
                                k2, p_k2 = stats.normaltest(resid)
                                skew_val = stats.skew(resid)
                                kurt_val = stats.kurtosis(resid)
                                
                                # 2. Autocorrelation (Durbin-Watson approximation)
                                resid_diff = np.diff(resid)
                                if np.sum(resid ** 2) > 0:
                                    dw_stat = np.sum(resid_diff ** 2) / np.sum(resid ** 2)
                                else:
                                    dw_stat = 0
                                
                                # 3. Homoscedasticity Check
                                corr_het, _ = stats.pearsonr(y_pred, np.abs(resid))
                                
                                # Tablo Hazırlama
                                stat_data = [
                                    {"Test": "Shapiro-Wilk (Normality)", "Statistic": f"{shapiro_stat:.4f}", "P-Value": f"{shapiro_p:.4e}", "Result": "Normal" if shapiro_p > 0.05 else "Not Normal"},
                                    {"Test": "D'Agostino's K^2 (Normality)", "Statistic": f"{k2:.4f}", "P-Value": f"{p_k2:.4e}", "Result": "Normal" if p_k2 > 0.05 else "Not Normal"},
                                    {"Test": "Durbin-Watson (Autocorr)", "Statistic": f"{dw_stat:.4f}", "P-Value": "-", "Result": "No Autocorr" if 1.5 < dw_stat < 2.5 else "Autocorrelation Detected"},
                                    {"Test": "Homoscedasticity (Corr)", "Statistic": f"{corr_het:.4f}", "P-Value": "-", "Result": "Homoscedastic" if abs(corr_het) < 0.1 else "Heteroscedastic Signs"},
                                ]
                                
                                df_stats = pd.DataFrame(stat_data)
                                
                                # Renklendirme Fonksiyonu
                                def color_results(val):
                                    color = 'red' 
                                    if val in ['Normal', 'No Autocorr', 'Homoscedastic']:
                                        color = 'green'
                                    elif val == '-':
                                        color = 'black'
                                    return f'color: {color}; font-weight: bold'

                                st.table(df_stats.style.applymap(color_results, subset=['Result']))
                                
                                c_s1, c_s2 = st.columns(2)
                                with c_s1:
                                    st.info(f"**Skewness:** {skew_val:.4f} (0 = Symmetric)")
                                with c_s2:
                                    st.info(f"**Kurtosis:** {kurt_val:.4f} (3 = Normal for some defs, 0 for Fisher)")
                                
                                st.caption("""
                                * **Shapiro-Wilk:** P-value > 0.05 assumes normal distribution.
                                * **Durbin-Watson:** Values around 2.0 indicate no autocorrelation.
                                * **Homoscedasticity:** Checks correlation between predicted values and absolute errors.
                                """)
                            
                            s_idx += 1


                        
                    # C. SHAP (GÜNCELLENEN KISIM: Violin Eklendi)
                    if run_shap:
                        with sub_tabs[s_idx]:
                            try:
                                X_bg = dat['Xt'].sample(min(100, len(dat['Xt'])), random_state=42)
                                exp = shap.Explainer(dat['model'].predict, X_bg)
                                sv = exp(dat['Xte'])
                                
                                # SHAP için 5 alt sekme (Violin Eklendi)
                                shap_tabs = st.tabs(["Beeswarm", "Violin", "Bar Plot", "Force Plot (Local)", "Data Table"])
                                
                                # 1. Beeswarm
                                with shap_tabs[0]:
                                    st.markdown("**Summary Plot (Beeswarm)**")
                                    fig_beeswarm, ax_beeswarm = plt.subplots()
                                    shap.summary_plot(sv, dat['Xte'], show=False)
                                    st.pyplot(fig_beeswarm)
                                    
                                # 2. Violin Plot (YENİ)
                                with shap_tabs[1]:
                                    st.markdown("**Violin Summary Plot**")
                                    fig_violin, ax_violin = plt.subplots()
                                    shap.summary_plot(sv, dat['Xte'], plot_type="violin", show=False)
                                    st.pyplot(fig_violin)
                                
                                # 3. Bar Plot
                                with shap_tabs[2]:
                                    st.markdown("**Feature Importance (Bar)**")
                                    fig_bar, ax_bar = plt.subplots()
                                    shap.summary_plot(sv, dat['Xte'], plot_type="bar", show=False)
                                    st.pyplot(fig_bar)
                                        
                                # 4. Local Explanation
                                with shap_tabs[3]:
                                    idx = st.slider(f"Select Index ({sc_name})", 0, len(dat['Xte'])-1, 0, key=f"slider_{i}")
                                    st.write(f"Index: **{idx}**")
                                    
                                    if hasattr  (sv, 'base_values'):
                                         base = sv.base_values[idx]
                                         val = sv.values[idx]
                                    else:
                                         base = exp.expected_value
                                         val = sv[idx]
                                    p = shap.force_plot(base, val, dat['Xte'].iloc[idx], matplotlib=False)
                                    st_shap(p, height=120)
                                    
                                # 5. Data Table
                                with shap_tabs[4]:
                                    if hasattr(sv, 'values'):
                                        shap_df = pd.DataFrame(sv.values, columns=dat['Xte'].columns, index=dat['Xte'].index)
                                    else:
                                        shap_df = pd.DataFrame(sv, columns=dat['Xte'].columns, index=dat['Xte'].index)
                                    st.dataframe(shap_df.head(50))

                            except Exception as e: st.error(f"SHAP Error: {e}")
