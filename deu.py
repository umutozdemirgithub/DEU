import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn.inspection import permutation_importance
from scipy import stats
import shap
import streamlit.components.v1 as components
import warnings

import statsmodels.api as sm
from scipy.stats import gaussian_kde, probplot
from scipy.stats import shapiro
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import time

# Optional / best-effort imports
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from lime import lime_tabular
except Exception:
    lime_tabular = None

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# -------------------------
# Utilities
# -------------------------
def st_shap(plot, height=None):
    """Helper to render shap plots in Streamlit (JS-based)."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def safe_model_factory(name, random_state=42):
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=random_state)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=random_state)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=random_state)
    if name == "XGBoost":
        if XGBRegressor is None: raise ImportError("XGBoost not installed")
        return XGBRegressor(random_state=random_state, verbosity=0)
    if name == "LightGBM":
        if LGBMRegressor is None: raise ImportError("LightGBM not installed")
        return LGBMRegressor(random_state=random_state, verbose=-1)
    if name == "CatBoost":
        if CatBoostRegressor is None: raise ImportError("CatBoost not installed")
        return CatBoostRegressor(random_state=random_state, verbose=0)
    return HistGradientBoostingRegressor(random_state=random_state)

# -------------------------
# HPO Logic (FIXED: Time Series Awareness)
# -------------------------
def perform_hpo(X_train, y_train, method, model_name, use_timesplit=False):
    base_model = safe_model_factory(model_name)
    random_space = {}
    grid_space = {}
    
    # CV Stratejisi Belirleme (FIXED: Time Series Leakage Prevention)
    if use_timesplit:
        # Zaman serisi için özel split (Shuffle yok)
        cv_strategy = TimeSeriesSplit(n_splits=3)
    else:
        # Standart 3-fold
        cv_strategy = 3

    # 1. HistGradientBoosting & GradientBoosting
    if model_name in ["HistGradientBoosting", "GradientBoosting"]:
        
        # HistGradientBoosting için daha geniş ve "Default"u kapsayan bir arama uzayı
        if model_name == "HistGradientBoosting":
            random_space = {
                # Learning rate'i çok düşürürsen iterasyonu artırman gerekir. 
                # Loguniform ile 0.01 ile 0.3 arasında dengeli arama yapıyoruz.
                'learning_rate': stats.loguniform(0.01, 0.5), 
                
                # Default max_iter 100'dür. HPO'nun daha fazlasını denemesine izin verelim.
                'max_iter': stats.randint(100, 1000),
                
                # Max depth None (sınırsız) genellikle iyidir ama overfitting yaratabilir.
                # Hem sığ (3-5) hem derin (10-20) hem de None denetelim.
                'max_depth': [None, 3, 5, 10, 20],
                
                # Yaprak başına minimum örnek sayısı genelleme için kritiktir.
                'min_samples_leaf': stats.randint(20, 100),
                
                # ÖNEMLİ EKLEME: Regularization (L2). Overfitting'i engeller.
                'l2_regularization': stats.loguniform(1e-9, 10) 
            }
        else:
            # Standart GradientBoosting (daha yavaştır, max_iter düşük tutulmalı)
            random_space = {
                'learning_rate': stats.loguniform(0.01, 0.3),
                'n_estimators': stats.randint(100, 300), # GB'de max_iter yerine n_estimators kullanılır
                'max_depth': [3, 5, 8],
                'min_samples_leaf': stats.randint(20, 50)
            }

        # Grid Search Uzayı (Küçük ama etkili kombinasyonlar)
        grid_space = {
            'learning_rate': [0.01, 0.05, 0.1], 
            'max_iter': [100, 300, 500] if model_name=="HistGradientBoosting" else [100, 200],
            'max_depth': [None, 5, 10] if model_name=="HistGradientBoosting" else [3, 5],
            'l2_regularization': [0, 1.0] if model_name=="HistGradientBoosting" else [0] # Sadece HistGB için
        }

    # 2. RandomForest
    elif model_name == "RandomForest":
        random_space = {
            'n_estimators': stats.randint(50, 300), 
            'max_depth': stats.randint(3, 20),
            'min_samples_split': stats.randint(2, 10)
        }
        grid_space = {
            'n_estimators': [100, 200], 
            'max_depth': [5, 10, None]
        }

    # 3. CatBoost
    elif model_name == "CatBoost":
        random_space = {
            'depth': stats.randint(4, 10), 
            'learning_rate': stats.loguniform(0.01, 0.3), 
            'iterations': stats.randint(100, 500),
            'l2_leaf_reg': stats.randint(1, 10)
        }
        grid_space = {
            'depth': [6, 8], 
            'learning_rate': [0.03, 0.1],
            'iterations': [200]
        }
    
    # 4. XGBoost
    elif model_name == "XGBoost":
        random_space = {
            'n_estimators': stats.randint(100, 500),
            'max_depth': stats.randint(3, 10),
            'learning_rate': stats.loguniform(0.01, 0.3),
            'subsample': stats.uniform(0.5, 0.5),
            'colsample_bytree': stats.uniform(0.5, 0.5)
        }
        grid_space = {
            'n_estimators': [100, 300],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }

    # 5. LightGBM
    elif model_name == "LightGBM":
        random_space = {
            'n_estimators': stats.randint(100, 500),
            'learning_rate': stats.loguniform(0.01, 0.3),
            'num_leaves': stats.randint(20, 100),
            'max_depth': stats.randint(-1, 15)
        }
        grid_space = {
            'n_estimators': [100, 300],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 50]
        }

    # HPO Uygulama
    if method == "Random Search" and random_space:
        st.write(f"⚙️ Tuning {model_name} with Random Search ({'TimeSeries' if use_timesplit else 'KFold'})...")
        search = RandomizedSearchCV(base_model, random_space, n_iter=30, cv=cv_strategy, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_
        
    elif method == "Grid Search" and grid_space:
        st.write(f"⚙️ Tuning {model_name} with Grid Search ({'TimeSeries' if use_timesplit else 'KFold'})...")
        search = GridSearchCV(base_model, grid_space, cv=cv_strategy, scoring='neg_mean_squared_error', n_jobs=-1)
        search.fit(X_train, y_train)
        return search.best_estimator_
    
    else:
        return base_model

# -------------------------
# Training & evaluation (FIXED: Data Leakage Prevention)
# -------------------------
def train_and_evaluate(X, y, test_size, model_name, hpo_method=None, use_timesplit=False, 
                       progress_callback=None, active_metrics=None, 
                       outlier_methods=None, scaling_methods=None):
    
    # 1. SPLIT (Önce ayır, sonra işlem yap -> Leakage Fix)
    if use_timesplit:
        tss = int(len(X)*(1-test_size))
        X_train, X_test = X.iloc[:tss].copy(), X.iloc[tss:].copy()
        y_train, y_test = y.iloc[:tss].copy(), y.iloc[tss:].copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        X_train, X_test = X_train.copy(), X_test.copy()
        y_train, y_test = y_train.copy(), y_test.copy()

    # 2. OUTLIER HANDLING (Sadece Train verisine uygulanır)
    if outlier_methods:
        # Outlier temizliği için geçici birleşme (sadece X_train ve y_train için)
        train_df = pd.concat([X_train, y_train], axis=1)
        target_col_name = y_train.name
        feature_cols = X_train.columns.tolist()
        
        # Temizlik
        df_clean = train_df.copy()
        cols_to_check = feature_cols # Genellikle featurelarda aranır
        
        if "IQR Capping" in outlier_methods:
            for col in cols_to_check:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])

        if "Z-Score Capping" in outlier_methods:
            for col in cols_to_check:
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                upper = mean_val + 3 * std_val
                lower = mean_val - 3 * std_val
                df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
                df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])

        if "Isolation Forest (Drop)" in outlier_methods:
            iso = IsolationForest(contamination=0.05, random_state=42)
            yhat = iso.fit_predict(df_clean[feature_cols])
            mask = yhat != -1
            df_clean = df_clean[mask]

        # Temizlenmiş veriyi geri ayır
        X_train = df_clean[feature_cols]
        y_train = df_clean[target_col_name]

    # 3. SCALING (Train fit -> Train transform -> Test transform)
    if scaling_methods:
        cols_to_scale = X_train.columns.tolist()
        
        # Çoklu seçim olabilir, sırayla uygular (Genelde tek bir scaler seçilmesi önerilir ama yapı destekler)
        if "Log Transformation (np.log1p)" in scaling_methods:
            for col in cols_to_scale:
                if (X_train[col] >= 0).all() and (X_test[col] >= 0).all():
                    X_train[col] = np.log1p(X_train[col])
                    X_test[col] = np.log1p(X_test[col])
        
        scaler = None
        if "Min-Max Scaling (0-1)" in scaling_methods:
            scaler = MinMaxScaler()
        elif "Standard Scaling (Z-Score)" in scaling_methods:
            scaler = StandardScaler()
        elif "Robust Scaling (IQR based)" in scaling_methods:
            scaler = RobustScaler()
        elif "MaxAbs Scaling (-1 to 1)" in scaling_methods:
            scaler = MaxAbsScaler()
            
        if scaler:
            # Sadece Train üzerine FIT
            X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
            # Test üzerine TRANSFORM (Sızıntı yok)
            X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # 4. MODEL TRAINING / HPO
    if hpo_method:
        # HPO fonksiyonuna use_timesplit bilgisini de gönderiyoruz
        model = perform_hpo(X_train, y_train, hpo_method, model_name, use_timesplit=use_timesplit)
    else:
        model = safe_model_factory(model_name)
        model.fit(X_train, y_train)

    # Progress simulation
    if progress_callback:
        for p in [20,50,80,100]:
            progress_callback(p)
            time.sleep(0.01)

    # 5. PREDICTION & METRICS
    y_pred = model.predict(X_test)
    
    metrics = {}
    if active_metrics is None: 
        active_metrics = ["RMSE", "MAE", "R2"]

    if "MSE" in active_metrics or "RMSE" in active_metrics:
        mse_val = mean_squared_error(y_test, y_pred)
        if "MSE" in active_metrics: metrics['MSE'] = mse_val
        if "RMSE" in active_metrics: metrics['RMSE'] = np.sqrt(mse_val)
    if "MAE" in active_metrics:
        metrics['MAE'] = mean_absolute_error(y_test, y_pred)
    if "R2" in active_metrics:
        metrics['R2'] = r2_score(y_test, y_pred)
    if "MAPE" in active_metrics:
        metrics['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)
    if "MedAE" in active_metrics:
        metrics['MedAE'] = median_absolute_error(y_test, y_pred)

    return metrics, model, X_train, X_test, y_test, y_pred

# -------------------------
# Diagnostics & Visuals
# -------------------------
def display_forecast_plots(y_test, y_pred, scenario_name, key_prefix="forecast"):
    df_viz = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
    df_viz = df_viz.sort_index()
    limit = min(120, len(df_viz))
    df_viz_subset = df_viz.head(limit)
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df_viz_subset.index, y=df_viz_subset['Actual'], name="Actual", mode='lines', line=dict(width=2)))
    fig_f.add_trace(go.Scatter(x=df_viz_subset.index, y=df_viz_subset['Predicted'], name="Predicted", mode='lines', line=dict(width=2, dash='dot')))
    fig_f.update_layout(title=f"Actual vs Predicted (first {limit} points)", template="plotly_white", height=420)
    st.plotly_chart(fig_f, use_container_width=True, key=f"{key_prefix}_{scenario_name}")

def display_diagnostic_plots(y_test, y_pred, key_prefix="diag"):
    st.markdown("## 🔍 Model Diagnostic Suite (Advanced)")
    resid = np.array(y_test) - np.array(y_pred)
    std_resid = (resid - np.mean(resid)) / np.std(resid)
    # Influence
    try:
        X = sm.add_constant(pd.Series(np.array(y_pred)))
        model = sm.OLS(pd.Series(np.array(y_test)), X).fit()
        influence = model.get_influence()
        leverage = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]
    except Exception:
        leverage = np.zeros_like(resid)
        cooks_d = np.zeros_like(resid)

    tabs = st.tabs(["Residuals","Distribution","QQ","Std Resid","Influence"])
    with tabs[0]:
        lowess = sm.nonparametric.lowess(resid, y_pred, frac=0.3)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=resid, mode='markers'))
        fig.add_hline(y=0, line_dash='dash')
        fig.add_trace(go.Scatter(x=lowess[:,0], y=lowess[:,1], mode='lines', line=dict(width=3)))
        fig.update_layout(template='plotly_white', height=360)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_resid")

    with tabs[1]:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=resid, nbinsx=40, opacity=0.7))
        try:
            kde = gaussian_kde(resid)
            x_vals = np.linspace(resid.min(), resid.max(), 200)
            fig.add_trace(go.Scatter(x=x_vals, y=kde(x_vals)*len(resid)*(resid.max()-resid.min())/40, mode='lines'))
        except Exception:
            pass
        fig.update_layout(template='plotly_white', height=360)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_dist")

    with tabs[2]:
        qq = probplot(resid, dist='norm')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers'))
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines'))
        fig.update_layout(template='plotly_white', height=360)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_qq")
        
    with tabs[3]:
        fig = px.scatter(x=y_pred, y=std_resid, labels={'x':'Predicted','y':'Std Residuals'})
        fig.add_hline(y=0, line_dash='dash')
        fig.add_hline(y=3, line_dash='dot', line_color='red')
        fig.add_hline(y=-3, line_dash='dot', line_color='red')
        fig.update_layout(template='plotly_white', height=360)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_std_resid")
        
    with tabs[4]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=leverage, y=cooks_d, mode='markers'))
        fig.update_layout(template='plotly_white', height=360)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_influence")

def interpret_model_diagnostics(y_test, y_pred):
    resid = np.array(y_test) - np.array(y_pred)
    resid_mean = float(np.mean(resid))
    resid_std = float(np.std(resid))
    hetero_corr = float(np.corrcoef(resid, y_pred)[0,1]) if len(resid)>1 else 0.0
    outlier_ratio = float(np.sum(np.abs(resid)>3*resid_std)/len(resid)) if resid_std>0 else 0.0
    try:
        _, p_value = shapiro(resid) if len(resid)>=3 else (None,1.0)
    except Exception:
        p_value=1.0
    trend_corr = float(np.corrcoef(np.arange(len(resid)), np.sort(resid))[0,1]) if len(resid)>1 else 0.0

    # Cards
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Residual Mean", f"{resid_mean:.4f}")
    c2.metric("Residual Std", f"{resid_std:.4f}")
    c3.metric("Hetero Corr", f"{hetero_corr:.3f}")
    c4.metric("Outlier Ratio", f"{outlier_ratio*100:.2f}%")
    c5.metric("Shapiro p", f"{p_value:.3f}")

    # Summary
    def label(cond): return "✔" if cond=="good" else ("⚠" if cond=="warn" else "❌")
    bias = "good" if abs(resid_mean)<resid_std*0.05 else ("warn" if abs(resid_mean)<resid_std*0.15 else "bad")
    hetero = "good" if abs(hetero_corr)<0.15 else ("warn" if abs(hetero_corr)<0.35 else "bad")
    normal = "good" if p_value>0.05 else "warn"
    outl = "good" if outlier_ratio<0.01 else ("warn" if outlier_ratio<0.05 else "bad")
    trend = "good" if abs(trend_corr)<0.15 else "warn"

    st.markdown("### 🧩 Summary")
    st.markdown(f"**Bias:** {label(bias)}  |  **Heteroskedasticity:** {label(hetero)}  |  **Normality:** {label(normal)}  |  **Outliers:** {label(outl)}  |  **Trend:** {label(trend)}")

    stats_df = pd.DataFrame({"Metric":["Resid Mean","Resid Std","Hetero Corr","Outlier Ratio","Shapiro p","Trend Corr"],"Value":[resid_mean,resid_std,hetero_corr,outlier_ratio,p_value,trend_corr]})
    st.dataframe(stats_df, use_container_width=True)

    return {"resid":resid, "mean":resid_mean, "std":resid_std, "hetero":hetero_corr, "outlier_ratio":outlier_ratio, "p_value":p_value, "trend":trend_corr}

# XAI (SHAP/PFI/LIME) - kept robust
def run_xai_analysis(model, X_train, X_test, y_test, y_pred, methods, key_suffix):
    if not methods:
        st.info("No XAI methods selected.")
        return

    xai_tabs = st.tabs(methods)

    # ----- SHAP -----
    if "SHAP" in methods:
        with xai_tabs[methods.index("SHAP")]:
            st.info("Computing SHAP (Tree explainer preferred)")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test) if hasattr(explainer, 'shap_values') else explainer(X_test)
                
                # Summary plot
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"SHAP failed: {e}")

    # ----- PFI -----
    if "PFI" in methods:
        with xai_tabs[methods.index("PFI")]:
            st.info("Permutation Feature Importance")
            try:
                r_pfi = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
                idx = r_pfi.importances_mean.argsort()[::-1]
                pfi_df = pd.DataFrame({
                    "Feature": X_test.columns[idx],
                    "Importance": r_pfi.importances_mean[idx]
                })
                
                fig = px.bar(pfi_df, x='Importance', y='Feature', orientation='h', height=400)
                st.plotly_chart(fig, use_container_width=True, key=f'pfi_chart_{key_suffix}')
                st.dataframe(pfi_df, use_container_width=True, key=f'pfi_df_{key_suffix}')
            except Exception as e:
                st.error(f"PFI error: {e}")

    # ----- LIME -----
    if "LIME" in methods:
        with xai_tabs[methods.index("LIME")]:
            if lime_tabular is None:
                st.warning("LIME not installed.")
            else:
                st.info("Generating LIME explanation for a selected instance")
                idx = st.number_input(
                    "Instance index", 
                    min_value=0, 
                    max_value=max(0, len(X_test) - 1), 
                    value=0, 
                    key=f'lime_idx_{key_suffix}'
                )

                explainer = lime_tabular.LimeTabularExplainer(
                    X_train.values, 
                    feature_names=X_train.columns, 
                    mode='regression'
                )
                exp = explainer.explain_instance(
                    X_test.iloc[idx].values,
                    lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns)),
                    num_features=10
                )
                
                components.html(exp.as_html(), height=500, scrolling=True)

# -------------------------
# PDF Report generator
# -------------------------
def generate_pdf_report(results):
    buf = BytesIO()
    plt.ioff() 
    
    with PdfPages(buf) as pdf:
        # 1. Leaderboard
        fig_table, ax_table = plt.subplots(figsize=(11.69, 8.27)) 
        ax_table.axis('off')
        ax_table.set_title("Model Leaderboard & Summary", fontsize=16, fontweight='bold', y=1.02)
        
        dfm = pd.DataFrame([{**{'Scenario':k}, **v['metrics']} for k,v in results.items()])
        numeric_cols = dfm.select_dtypes(include=[np.number]).columns
        dfm[numeric_cols] = dfm[numeric_cols].round(4)
        
        table = ax_table.table(cellText=dfm.values, colLabels=dfm.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        pdf.savefig(fig_table, bbox_inches='tight')
        plt.close(fig_table)

        # 2. Leaderboard Chart
        fig_chart, ax_chart = plt.subplots(figsize=(11.69, 8.27))
        
        bar_width = 0.2
        index = np.arange(len(dfm))
        colors = plt.cm.Pastel1.colors 
        color_idx = 0
        
        for metric in numeric_cols:
            if metric != 'R2':
                ax_chart.bar(index + color_idx*bar_width, dfm[metric], bar_width, label=metric, color=colors[color_idx % len(colors)])
                color_idx += 1
        
        if 'R2' in numeric_cols:
            ax_chart.plot(index + bar_width, dfm['R2'], color='crimson', marker='o', linestyle='--', linewidth=2, label='R2')
        
        ax_chart.set_xticks(index + bar_width*(color_idx-1)/2)
        ax_chart.set_xticklabels(dfm['Scenario'], rotation=45, ha='right')
        ax_chart.set_ylabel("Score")
        ax_chart.set_title("Leaderboard Metrics & R2", fontsize=14, fontweight='bold')
        ax_chart.legend()
        ax_chart.grid(axis='y', linestyle='--', alpha=0.7)
        
        pdf.savefig(fig_chart, bbox_inches='tight')
        plt.close(fig_chart)

        # 3. Individual Scenarios
        for scenario_name, res in results.items():
            model = res['model']
            X_test = res['Xte']
            y_test = res['yte']
            y_pred = res['ypr']
            metrics = res['metrics']
            resid = y_test - y_pred

            fig_perf, axs = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig_perf.suptitle(f"Performance: {scenario_name}", fontsize=14, fontweight='bold')

            limit = min(300, len(y_test))
            axs[0].plot(y_test.index[:limit], y_test[:limit], label='Actual', color='black', linewidth=1.5)
            axs[0].plot(y_test.index[:limit], y_pred[:limit], label='Predicted', color='blue', linestyle='--', alpha=0.7)
            axs[0].set_title(f"Forecast (First {limit} points)")
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)

            axs[1].scatter(y_pred, resid, alpha=0.5, color='purple', s=10)
            axs[1].axhline(0, color='red', linestyle='--', linewidth=1)
            axs[1].set_xlabel("Predicted")
            axs[1].set_ylabel("Residuals")
            axs[1].set_title("Residual Analysis")
            axs[1].grid(True, alpha=0.3)

            axs[2].hist(resid, bins=30, color='green', edgecolor='black', alpha=0.7)
            axs[2].set_title("Error Distribution")
            
            stats_text = " | ".join([f"{k}: {v:.4f}" for k,v in metrics.items()])
            fig_perf.text(0.5, 0.02, stats_text, ha='center', fontsize=12, bbox=dict(facecolor='lightgrey', alpha=0.5))
            
            plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.4)
            pdf.savefig(fig_perf)
            plt.close(fig_perf)

            fig_xai = plt.figure(figsize=(8.27, 11.69))
            fig_xai.suptitle(f"XAI Analysis: {scenario_name}", fontsize=14, fontweight='bold')
            
            gs = fig_xai.add_gridspec(2, 1, hspace=0.3)
            ax_imp = fig_xai.add_subplot(gs[0])
            
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[-10:] 
                    ax_imp.barh(range(len(indices)), importances[indices], align='center', color='teal')
                    ax_imp.set_yticks(range(len(indices)))
                    ax_imp.set_yticklabels([X_test.columns[i] for i in indices])
                    ax_imp.set_title("Top 10 Feature Importances (Built-in)")
                else:
                    r_pfi = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
                    idx = r_pfi.importances_mean.argsort()[-10:]
                    ax_imp.barh(range(len(idx)), r_pfi.importances_mean[idx], align='center', color='orange')
                    ax_imp.set_yticks(range(len(idx)))
                    ax_imp.set_yticklabels([X_test.columns[i] for i in idx])
                    ax_imp.set_title("Top 10 Feature Importances (Permutation)")
            except Exception as e:
                ax_imp.text(0.5, 0.5, f"Feature Importance Error: {str(e)}", ha='center')

            ax_shap = fig_xai.add_subplot(gs[1])
            ax_shap.set_title("SHAP Summary Plot (Top Interactions)")
            
            try:
                sample_size = min(100, len(X_test))
                X_sample = X_test.iloc[:sample_size]
                
                explainer = None
                shap_values = None
                
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                except:
                    pass

                if shap_values is not None:
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0] 
                    
                    plt.sca(ax_shap) 
                    shap.summary_plot(shap_values, X_sample, show=False, plot_size=None, color_bar=False)
                else:
                    ax_shap.text(0.5, 0.5, "SHAP Explainer not compatible or too slow.", ha='center')

            except Exception as e:
                ax_shap.text(0.5, 0.5, f"SHAP Error: {str(e)}", ha='center')

            pdf.savefig(fig_xai)
            plt.close(fig_xai)

    buf.seek(0)
    return buf.read()

def display_leaderboard(results: dict) -> pd.DataFrame:
    df_results = pd.DataFrame([{**{'Scenario': k}, **v['metrics']} for k, v in results.items()])
    numeric_cols = df_results.select_dtypes(include=['float', 'int']).columns.tolist()
    
    with st.expander("### 🏆 Model Leaderboard", expanded=False):
        tabs = st.tabs(["📊 Table", "📈 Chart", "⚙️ Parameters"])
        
        with tabs[0]:
            st.dataframe(
                df_results.style.format(formatter="{:.4f}", subset=numeric_cols),
                use_container_width=True
            )
        
        with tabs[1]:
            import plotly.express as px
            import plotly.graph_objects as go

            fig = go.Figure()
            colors = px.colors.qualitative.Pastel
            color_idx = 0
            
            for metric in numeric_cols:
                if metric != 'R2':
                    fig.add_trace(
                        go.Bar(
                            x=df_results['Scenario'],
                            y=df_results[metric],
                            name=metric,
                            text=df_results[metric].round(4),
                            textposition='outside',
                            marker_color=colors[color_idx % len(colors)]
                        )
                    )
                    color_idx += 1
            
            if 'R2' in numeric_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df_results['Scenario'],
                        y=df_results['R2'],
                        mode='lines+markers',
                        name='R2',
                        line=dict(dash='dash', color='crimson', width=3),
                        marker=dict(size=10, color='crimson'),
                        text=df_results['R2'].round(4),
                        textposition='top center'
                    )
                )
            
            fig.update_layout(
                title="Leaderboard - All Metrics & R2",
                yaxis=dict(title="Score"),
                xaxis=dict(title="Model"),
                barmode='group',
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            st.markdown("#### ⚙️ Model Hyperparameters Configuration")
            param_list = []
            
            for name, res in results.items():
                model_obj = res['model']
                try:
                    params = model_obj.get_params()
                except Exception:
                    params = {"info": str(model_obj)}
                
                clean_params = {'Scenario': name}
                for k, v in params.items():
                    clean_params[k] = str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                
                param_list.append(clean_params)
            
            if param_list:
                df_params = pd.DataFrame(param_list)
                cols = ['Scenario'] + [c for c in df_params.columns if c != 'Scenario']
                df_params = df_params[cols]
                df_params = df_params.fillna("-")
                st.dataframe(df_params, use_container_width=True)
            else:
                st.info("No parameter data available.")

    return df_results

# -------------------------
# Streamlit App
# -------------------------

st.set_page_config(page_title="Model Diagnostic Dashboard – Pro + PDF + Models", layout="wide")
st.title("🚀 Model Diagnostic Dashboard – Full Edition")
st.markdown("All-in-one: UI polish, PDF export, extra models, TimeSeriesSplit, progress bars, SHAP/LIME/PFI.")

# Sidebar
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV (must include target column)", type=['csv'])
if not uploaded_file:
    st.info("Please upload a CSV file to proceed. You can use the existing deu.py as example data if you prefer.")
    st.stop()

try:
    df_original = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded CSV")
except Exception as e:
    st.sidebar.error(f"Failed reading CSV: {e}")
    st.stop()

# Date handling
cols = df_original.columns.tolist()
date_guess = [c for c in cols if 'date' in c.lower() or 'time' in c.lower()]
date_col = st.sidebar.selectbox('Date column (optional)', [None] + cols, index=0)
if date_col:
    try:
        df_original[date_col] = pd.to_datetime(df_original[date_col])
        df_original.set_index(date_col, inplace=True)
        df_original = df_original.sort_index()
    except Exception:
        st.sidebar.warning('Could not parse date column - leaving as-is')

# Target & features
target_default = 'Power' if 'Power' in cols else cols[-1]
target_col = st.sidebar.selectbox('Target column', cols, index=cols.index(target_default))
feature_cols = st.sidebar.multiselect('Feature columns', [c for c in cols if c!=target_col], default=[c for c in cols if c!=target_col][:6])
if not feature_cols:
    st.sidebar.error('Select at least one feature')
    st.stop()

# Models
available = ["HistGradientBoosting","RandomForest","GradientBoosting","CatBoost","XGBoost","LightGBM"]
avail_filtered = [m for m in available if not (m=="XGBoost" and XGBRegressor is None) and not (m=="LightGBM" and LGBMRegressor is None) and not (m=="CatBoost" and CatBoostRegressor is None)]
selected_models = st.sidebar.multiselect('Models', avail_filtered, default=[avail_filtered[0]])

# Metrics
metric_options = ["RMSE", "MAE", "R2", "MSE", "MAPE", "MedAE"]
selected_metrics = st.sidebar.multiselect('Evaluation Metrics', metric_options, default=["MSE", "RMSE", "MAE", "R2"])
if not selected_metrics:
    st.sidebar.error("Please select at least one metric.")
    st.stop()

# Preproc
outlier_methods = st.sidebar.multiselect('Outlier handling (Training Data Only)', ["IQR Capping","Z-Score Capping","Isolation Forest (Drop)"])
scaling_methods = st.sidebar.multiselect('Scaling (Fit on Train, Transform Test)', ["Min-Max Scaling (0-1)","Standard Scaling (Z-Score)","Robust Scaling (IQR based)","MaxAbs Scaling (-1 to 1)","Log Transformation (np.log1p)"])

# HPO and XAI
hpo_methods = st.sidebar.multiselect('HPO Methods', ['Random Search','Grid Search'])
xai_methods = st.sidebar.multiselect('XAI Methods', ['SHAP','PFI','LIME'])

st.sidebar.markdown("---")
st.sidebar.markdown("**View Options**")
use_timesplit = st.sidebar.checkbox('Use Time-series split (train on earliest, test on latest)', value=True)
show_combined = st.sidebar.checkbox('Show combined forecast', value=True)
show_forecasts = st.sidebar.checkbox('Show scenario forecasts', value=True)
show_diags = st.sidebar.checkbox('Show diagnostics', value=True)

train_btn = st.sidebar.button('Train & Run All')

# --- EĞİTİM ALGORİTMASI (FIXED: Clean Loop) ---
if train_btn:
    st.info('Starting training — this may take some time depending on models and HPO.')
    results = {}
    progress = st.progress(0)
    
    # Senaryo Hazırlığı (Döngü Karmaşıklığını Azaltma)
    # Her model için hangi varyasyonların çalışacağını belirliyoruz
    scenarios = []
    
    # 1. Default (Her zaman var)
    scenarios.append({
        "suffix": "Default",
        "hpo": None,
        "use_prep": False
    })

    # 2. Preprocessed (Eğer seçildiyse)
    if outlier_methods or scaling_methods:
        scenarios.append({
            "suffix": "Preprocessed",
            "hpo": None,
            "use_prep": True
        })
        
    # 3. HPO (Eğer seçildiyse)
    # Not: HPO'yu genellikle clean data (Preprocessed) üzerinde yapmak mantıklıdır.
    # Eğer preprocessing seçili değilse raw data üzerinde çalışır.
    for hpo in hpo_methods:
        scenarios.append({
            "suffix": f"HPO ({hpo})",
            "hpo": hpo,
            "use_prep": True if (outlier_methods or scaling_methods) else False
        })

    # Toplam işlem sayısı progress bar için
    total_tasks = len(selected_models) * len(scenarios)
    task_idx = 0

    # Ana Döngü
    for model_name in selected_models:
        for scen in scenarios:
            # Etiket oluşturma
            label = f"{model_name} - {scen['suffix']}"
            
            # Parametre ayarı
            current_outlier = outlier_methods if scen['use_prep'] else None
            current_scaling = scaling_methods if scen['use_prep'] else None
            
            # Veri Hazırlığı
            # Veri artık fonksiyon içinde split edilip işleniyor, buraya raw gönderiyoruz.
            X_raw = df_original[feature_cols]
            y_raw = df_original[target_col]
            
            def prog_cb(p): 
                # Progress bar güncelleme alt fonksiyonu
                val = min(100, int((task_idx/total_tasks*100) + p/total_tasks))
                progress.progress(val)

            try:
                # Fonksiyon Çağrısı (Refactored)
                metrics, model, Xt, Xte, yte, ypr = train_and_evaluate(
                    X_raw, y_raw, 
                    test_size=0.2, 
                    model_name=model_name, 
                    hpo_method=scen['hpo'], 
                    use_timesplit=use_timesplit, 
                    progress_callback=prog_cb, 
                    active_metrics=selected_metrics,
                    outlier_methods=current_outlier,
                    scaling_methods=current_scaling
                )
                
                results[label] = {'metrics':metrics,'model':model,'Xt':Xt,'Xte':Xte,'yte':yte,'ypr':ypr}
                
            except Exception as e:
                st.error(f"Error in {label}: {e}")
            
            task_idx += 1
            progress.progress(min(100, int(task_idx/total_tasks*100)))

    st.success('Training complete')
    st.session_state['results_all'] = results

# --- SONUÇLARI GÖSTERME (EĞİTİM BUTTONUNUN DIŞINDA) ---
if 'results_all' in st.session_state:
    results = st.session_state['results_all']
    
    df_results = display_leaderboard(results)

    # 1. BÖLÜM: FORECASTS
    if show_combined or show_forecasts:
        with st.expander("### 📈 Forecast Overview", expanded=False):
            tabs_to_create = []
            if show_combined: tabs_to_create.append("Combined Comparison")
            if show_forecasts: tabs_to_create.append("Individual Scenarios")
            
            if tabs_to_create:
                main_tabs = st.tabs(tabs_to_create)
                tab_map = dict(zip(tabs_to_create, main_tabs))

                if show_combined and "Combined Comparison" in tab_map:
                    with tab_map["Combined Comparison"]:
                        st.markdown("#### All Models vs Actual")
                        fig = go.Figure()
                        first = next(iter(results.values()))
                        n = min(200, len(first['yte']))
                        
                        fig.add_trace(go.Scatter(x=first['yte'].index[:n], y=first['yte'][:n], name='Actual', line=dict(width=3, color='black')))
                        for name, r in results.items():
                            yp = pd.Series(r['ypr'], index=r['yte'].index)[:n]
                            fig.add_trace(go.Scatter(x=yp.index, y=yp, name=name, line=dict(dash='dot')))
                        fig.update_layout(template='plotly_white', height=450, margin=dict(t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                if show_forecasts and "Individual Scenarios" in tab_map:
                    with tab_map["Individual Scenarios"]:
                        scenario_names = list(results.keys())
                        if scenario_names:
                            sub_tabs = st.tabs(scenario_names)
                            for i, scenario in enumerate(scenario_names):
                                with sub_tabs[i]:
                                    sel = results[scenario]
                                    display_forecast_plots(sel['yte'], sel['ypr'], scenario, key_prefix=f"ind_forecast_{i}")

    # 2. BÖLÜM: DIAGNOSTICS
    if show_diags:
        with st.expander("### 🔍 Model Diagnostics", expanded=False):
            scenario_names = list(results.keys())
            if scenario_names:
                d_tabs = st.tabs(scenario_names)
                for i, scenario in enumerate(scenario_names):
                    with d_tabs[i]:
                        sel = results[scenario]
                        display_diagnostic_plots(sel['yte'], sel['ypr'], key_prefix=f"diag_{i}")
                        interpret_model_diagnostics(sel['yte'], sel['ypr'])

    # 3. BÖLÜM: XAI
    if xai_methods:
        with st.expander("🧠 Explainable AI (XAI) Analysis", expanded=False):        
            scenario_names = list(results.keys())
            if scenario_names:
                xai_tabs = st.tabs(scenario_names)
                for i, scenario in enumerate(scenario_names):
                    with xai_tabs[i]:
                        sel_xai = results[scenario]
                        safe_key = scenario.replace(" ","_")
                        run_xai_analysis(sel_xai['model'], sel_xai['Xt'], sel_xai['Xte'], sel_xai['yte'], sel_xai['ypr'], xai_methods, key_suffix=f"xai_tab_{i}_{safe_key}")

    # 4. BÖLÜM: EXPORT (PDF & CSV)
    st.markdown("---")
    st.markdown("### 📥 Export Full Report")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Metrics (CSV)",
            data=csv_data,
            file_name="leaderboard.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col2:       
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf_data = generate_pdf_report(results)
                    st.session_state['pdf_ready'] = pdf_data
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

        if 'pdf_ready' in st.session_state:
            st.download_button(
                label="📕 Download Ready PDF",
                data=st.session_state['pdf_ready'],
                file_name="comprehensive_model_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

st.info('You can re-run training with different options from the sidebar.')
