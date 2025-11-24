import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, learning_curve
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn.inspection import permutation_importance
from scipy import stats
import shap
import streamlit.components.v1 as components
import warnings
import statsmodels.api as sm
from scipy.stats import gaussian_kde, probplot, shapiro
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import time
from sklearn.pipeline import Pipeline
from sklearn.base import is_classifier

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

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

st.set_page_config(page_title="Model Diagnostic Dashboard", layout="wide")

# Utilities
@st.cache_data(ttl=3600)  # 1 saat boyunca cache'de tut
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            return None
    return None

def st_shap(plot, height=None):
    shap_html = f"{shap.getjs()}{plot.html()}"
    components.html(shap_html, height=height)

def safe_model_factory(name, random_state=42):
    defaults = MODEL_DEFAULT_PARAMS.get(name, {})

    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=random_state, **defaults)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=random_state, **defaults)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=random_state, **defaults)
    if name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError("XGBoost not installed")
        return XGBRegressor(random_state=random_state, verbosity=0, **defaults)
    if name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("LightGBM not installed")
        return LGBMRegressor(random_state=random_state, verbose=-1, **defaults)
    if name == "CatBoost":
        if CatBoostRegressor is None:
            raise ImportError("CatBoost not installed")
        return CatBoostRegressor(random_state=random_state, verbose=0, allow_writing_files=False, **defaults)

    return HistGradientBoostingRegressor(random_state=random_state)

MODEL_DEFAULT_PARAMS = {
    "HistGradientBoosting": {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_depth": None,
        "l2_regularization": 0.0,
        "min_samples_leaf": 30,
    },
    "RandomForest": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": -1,
    },
    "GradientBoosting": {
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 3,
        "subsample": 0.8,
    },
    "XGBoost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    "LightGBM": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    },
    "CatBoost": {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
    },
}
def perform_hpo(X_train, y_train, method, model_name, use_timesplit=False, scaler_cls=None):
    """
    Seçilen model için, sadece MODEL_DEFAULT_PARAMS'te tanımlı hiperparametreler
    üzerinden Random Search veya Grid Search ile HPO yapan yardımcı fonksiyon.
    """
    # 1) Başlangıç modeli ve pipeline
    base_model = safe_model_factory(model_name)
    steps = []
    if scaler_cls is not None:
        steps.append(("scaler", scaler_cls()))
    steps.append(("model", base_model))
    pipeline = Pipeline(steps)

    # 2) CV stratejisi
    if use_timesplit:
        cv_strategy = TimeSeriesSplit(n_splits=3)
    else:
        cv_strategy = 3

    # 3) Model bazlı HPO uzayı (geniş uzay; sonra MODEL_DEFAULT_PARAMS ile filtrelenecek)
    random_space = {}
    grid_space = {}

    if model_name == "HistGradientBoosting":
        random_space = {
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__max_iter": stats.randint(100, 500),
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_leaf": stats.randint(20, 100),
            "model__l2_regularization": stats.loguniform(1e-9, 10),
        }
        grid_space = {
            "model__learning_rate": [0.01, 0.1],
            "model__max_iter": [100, 300],
            "model__max_depth": [None, 5, 10],
            "model__l2_regularization": [0, 0.1, 1.0],
        }

    elif model_name == "GradientBoosting":
        random_space = {
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__n_estimators": stats.randint(100, 300),
            "model__max_depth": stats.randint(3, 8),
            "model__min_samples_split": stats.randint(2, 20),
            "model__min_samples_leaf": stats.randint(1, 10),
            "model__subsample": stats.uniform(0.7, 0.3),
        }
        grid_space = {
            "model__learning_rate": [0.01, 0.1],
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
        }

    elif model_name == "RandomForest":
        random_space = {
            "model__n_estimators": stats.randint(100, 400),
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": stats.randint(2, 15),
            "model__min_samples_leaf": stats.randint(1, 10),
            "model__max_features": ["sqrt", "log2", None],
        }
        grid_space = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, None],
            "model__min_samples_split": [2, 5],
        }

    elif model_name == "XGBoost":
        random_space = {
            "model__n_estimators": stats.randint(100, 500),
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__max_depth": stats.randint(3, 10),
            "model__subsample": stats.uniform(0.6, 0.4),
            "model__colsample_bytree": stats.uniform(0.6, 0.4),
            "model__reg_alpha": stats.loguniform(1e-5, 10),
            "model__reg_lambda": stats.loguniform(1e-5, 10),
        }
        grid_space = {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 6],
        }

    elif model_name == "LightGBM":
        random_space = {
            "model__n_estimators": stats.randint(100, 500),
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__num_leaves": stats.randint(20, 150),
            "model__max_depth": stats.randint(-1, 15),
            "model__reg_alpha": stats.uniform(0, 1),
            "model__reg_lambda": stats.uniform(0, 1),
            "model__subsample": stats.uniform(0.6, 0.4),
        }
        grid_space = {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.1],
            "model__num_leaves": [31, 63],
        }

    elif model_name == "CatBoost":
        random_space = {
            "model__iterations": stats.randint(100, 500),
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__depth": stats.randint(4, 10),
            "model__l2_leaf_reg": stats.randint(1, 10),
            "model__subsample": stats.uniform(0.6, 0.4),
        }
        grid_space = {
            "model__iterations": [200],
            "model__learning_rate": [0.03, 0.1],
            "model__depth": [6, 8],
        }

    # 4) Sadece MODEL_DEFAULT_PARAMS'te bulunan parametreleri bırak
    if model_name in MODEL_DEFAULT_PARAMS:
        allowed = {f"model__{p}" for p in MODEL_DEFAULT_PARAMS[model_name].keys()}
        random_space = {k: v for k, v in random_space.items() if k in allowed}
        grid_space   = {k: v for k, v in grid_space.items()   if k in allowed}

    # 5) Seçilen yönteme göre uzayı belirle
    selected_space = random_space if method == "Random Search" else grid_space
    final_params = selected_space

    # HPO yapılacak parametre kalmadıysa, direkt pipeline döndür
    if not final_params:
        return pipeline

    st.write(
        f"⚙️ Tuning {model_name} with {method} "
        f"({'TimeSeries' if use_timesplit else 'KFold'})..."
    )

    # 6) Random / Grid Search
    if method == "Random Search":
        search_engine = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=final_params,
            n_iter=20,
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
    else:
        search_engine = GridSearchCV(
            estimator=pipeline,
            param_grid=final_params,
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

    search_engine.fit(X_train, y_train)
    return search_engine.best_estimator_


# -------------------------
# Training & evaluation
# -------------------------
def train_and_evaluate(X, y, test_size, model_name, hpo_method=None, use_timesplit=False, 
                       _progress_callback=None, active_metrics=None, 
                       outlier_methods=None, scaling_methods=None):
    """
    Eğitim, değerlendirme ve tahmin pipeline'ı.
    """
    
    # 1. SPLIT (Önce ayır, sonra işlem yap -> Leakage Fix)
    if use_timesplit:
        tss = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:tss].copy(), X.iloc[tss:].copy()
        y_train, y_test = y.iloc[:tss].copy(), y.iloc[tss:].copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False) # Shuffle False for safety in sequence data if not explicit TS
        X_train, X_test = X_train.copy(), X_test.copy()
        y_train, y_test = y_train.copy(), y_test.copy()

    # 2. OUTLIER HANDLING (Sadece Train verisine uygulanır)
    if outlier_methods:
        train_df = pd.concat([X_train, y_train], axis=1)
        target_col = y_train.name
        feat_cols = X_train.columns.tolist()
        
        df_clean = train_df.copy()
        
        if "IQR Capping" in outlier_methods:
            for col in feat_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])

        if "Z-Score Capping" in outlier_methods:
            for col in feat_cols:
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                upper = mean_val + 3 * std_val
                lower = mean_val - 3 * std_val
                df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
                df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])

        if "Isolation Forest (Drop)" in outlier_methods:
            iso = IsolationForest(contamination=0.05, random_state=42)
            yhat = iso.fit_predict(df_clean[feat_cols])
            df_clean = df_clean[yhat != -1]

        X_train = df_clean[feat_cols]
        y_train = df_clean[target_col]

    # 3. SCALER SELECTION (Pipeline için sınıf seçimi)
    scaler_cls = None
    if scaling_methods:
        if "Min-Max Scaling (0-1)" in scaling_methods:
            scaler_cls = MinMaxScaler
        elif "Standard Scaling (Z-Score)" in scaling_methods:
            scaler_cls = StandardScaler
        elif "Robust Scaling (IQR based)" in scaling_methods:
            scaler_cls = RobustScaler
        elif "MaxAbs Scaling (-1 to 1)" in scaling_methods:
            scaler_cls = MaxAbsScaler
    
    if scaling_methods and "Log Transformation (np.log1p)" in scaling_methods:
        if (X_train.values >= 0).all():
            X_train = np.log1p(X_train)
            X_test = np.log1p(X_test)

    if _progress_callback: _progress_callback(10)

    # 4. MODEL TRAINING (Pipeline + HPO)
    if hpo_method:
        model = perform_hpo(
            X_train, y_train, 
            hpo_method, 
            model_name, 
            use_timesplit=use_timesplit,
            scaler_cls=scaler_cls
        )
    else:
        base_model = safe_model_factory(model_name)
        steps = []
        if scaler_cls:
            steps.append(('scaler', scaler_cls()))
        steps.append(('model', base_model))
        
        model = Pipeline(steps)
        model.fit(X_train, y_train)

    if _progress_callback: _progress_callback(90)

    # 5. PREDICTION
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # 6. METRICS
    metrics = {}
    if active_metrics is None: active_metrics = ["RMSE", "MAE", "R2"]

    if "MSE" in active_metrics: metrics['MSE'] = mean_squared_error(y_test, y_pred)
    if "RMSE" in active_metrics: metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
    if "MAE" in active_metrics: metrics['MAE'] = mean_absolute_error(y_test, y_pred)
    if "R2" in active_metrics: metrics['R2'] = r2_score(y_test, y_pred)
    if "MAPE" in active_metrics: metrics['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)
    if "MedAE" in active_metrics: metrics['MedAE'] = median_absolute_error(y_test, y_pred)
    
    if _progress_callback: _progress_callback(100)

    return metrics, model, X_train, X_test, y_train, y_test, y_train_pred, y_pred

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

def _compute_influence(y_test_arr, y_pred_arr):
    try:
        X = sm.add_constant(pd.Series(y_pred_arr))
        model_ols = sm.OLS(pd.Series(y_test_arr), X).fit()
        influence = model_ols.get_influence()
        leverage = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]
    except Exception:
        leverage = np.zeros_like(y_pred_arr, dtype=float)
        cooks_d = np.zeros_like(y_pred_arr, dtype=float)
    return leverage, cooks_d

def _ensemble_anomaly_detector(resid, leverage, cooks_d):
    """
    Returns anomaly scores (0..1) and labels
    """
    n = len(resid)
    X_feat = np.vstack([resid, leverage, cooks_d]).T
    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.nan_to_num(X_feat))

    scores = np.zeros((n, 4))  # IF, LOF (neg), DBSCAN (outlier flag), 3sigma
    # IsolationForest
    try:
        if_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        if_model.fit(Xs)
        if_score = -if_model.score_samples(Xs)
        scores[:,0] = (if_score - if_score.min()) / (if_score.ptp() + 1e-9)
    except Exception:
        scores[:,0] = 0

    # LOF
    try:
        lof = LocalOutlierFactor(n_neighbors=max(5, min(50, int(n/5))), contamination=0.05, novelty=False)
        lof_pred = lof.fit_predict(Xs)
        lof_score = -lof.negative_outlier_factor_
        scores[:,1] = (lof_score - lof_score.min()) / (lof_score.ptp() + 1e-9)
    except Exception:
        scores[:,1] = 0

    # DBSCAN
    try:
        db = DBSCAN(eps=0.8, min_samples=5)
        db_labels = db.fit_predict(Xs)
        db_flag = (db_labels == -1).astype(float)
        scores[:,2] = db_flag
    except Exception:
        scores[:,2] = 0

    # 3-sigma on residuals
    sigma = np.std(resid)
    three_sigma_flag = (np.abs(resid) > 3 * sigma).astype(float)
    scores[:,3] = three_sigma_flag

    # aggregate: weighted sum
    weights = np.array([0.4, 0.3, 0.15, 0.15])
    agg = scores.dot(weights)
    agg = (agg - agg.min()) / (agg.ptp() + 1e-9)

    labels = np.zeros_like(agg, dtype=int)
    labels[agg >= 0.75] = 2  # strong
    labels[(agg >= 0.4) & (agg < 0.75)] = 1  # medium
    labels[agg < 0.4] = 0

    return agg, labels

def display_diagnostic_plots(
    y_test, y_pred, y_train=None, y_train_pred=None, 
    model=None, X_train=None, 
    key_prefix="diag_pro"
):
    """
    Full professional diagnostic suite (Merged and Fixed).
    """
    st.markdown("## 🔍 Model Diagnostic Suite — PRO")

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    resid = y_test_arr - y_pred_arr
    std_resid = (resid - resid.mean()) / (resid.std() + 1e-9)

    leverage, cooks_d = _compute_influence(y_test_arr, y_pred_arr)

    # anomaly ensemble
    anomaly_score, anomaly_label = _ensemble_anomaly_detector(resid, leverage, cooks_d)

    # --- TAB YAPISINI OLUŞTURMA ---
    tab_list = ["Adv. Scatter", "Residuals", "Distribution", "QQ", "Std Resid", "Influence", "3D Views", "Anomalies"]
    
    # Train verisi varsa Overfitting ve Learning Curve ekle
    has_train = (y_train is not None) and (y_train_pred is not None)
    if has_train:
        tab_list.insert(1, "Overfitting Check")
        
    # Model ve X_train varsa Learning Curve ekle
    has_learning_curve = (model is not None) and (X_train is not None) and (has_train)
    if has_learning_curve:
        tab_list.insert(2, "Learning Curve")

    tabs = st.tabs(tab_list)
    
    # manage selection via session_state
    if "diag_selected_idx" not in st.session_state:
        st.session_state["diag_selected_idx"] = []
        st.session_state["active_tab"] = 0

    hover_texts = [
        f"Idx: {i}<br>Pred: {y_pred_arr[i]:.4f}<br>Obs: {y_test_arr[i]:.4f}<br>Resid: {resid[i]:.4f}<br>Cook's: {cooks_d[i]:.6f}"
        for i in range(len(y_pred_arr))
    ]
    
    current_tab = 0

    # 1. ADVANCED SCATTER
    with tabs[current_tab]:
        st.markdown("**Actual vs Predicted with Distributions**")
        df_chart = pd.DataFrame({'Actual': y_test_arr, 'Predicted': y_pred_arr})
        
        fig_adv = px.scatter(
            df_chart, x='Actual', y='Predicted', 
            trendline="ols", 
            trendline_color_override="red",
            marginal_x="histogram", 
            marginal_y="histogram",
            opacity=0.6,
            height=500
        )
        min_val = min(y_test_arr.min(), y_pred_arr.min())
        max_val = max(y_test_arr.max(), y_pred_arr.max())
        fig_adv.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="black", dash="dash"))
        st.plotly_chart(fig_adv, use_container_width=True, key=f"{key_prefix}_adv_scatter")
    
    current_tab += 1

    # 2. OVERFITTING
    if has_train:
        with tabs[current_tab]:
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
            fig = go.Figure(go.Bar(
                x=["Train", "Test"],
                y=[train_rmse, test_rmse],
                marker=dict(color=[train_rmse, test_rmse], colorscale="Bluered"),
                text=[f"{train_rmse:.4f}", f"{test_rmse:.4f}"],
                textposition="inside",
                hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>",
            ))
            fig.update_layout(title="RMSE Comparison (Train vs Test)", template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_overfit")
        current_tab += 1

    # 3. LEARNING CURVE
    if has_learning_curve:
        with tabs[current_tab]:
            st.markdown("**Learning Curve Analysis** (Checks for Bias vs Variance)")
            with st.spinner("Calculating learning curve (this may take a moment)..."):
                try:
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, X_train, y_train, cv=3, scoring='neg_mean_squared_error', 
                        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
                    )
                    train_rmse_lc = np.sqrt(-train_scores.mean(axis=1))
                    test_rmse_lc = np.sqrt(-test_scores.mean(axis=1))
                    
                    fig_lc = go.Figure()
                    fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_rmse_lc, mode='lines+markers', name='Training Error', line=dict(color='blue')))
                    fig_lc.add_trace(go.Scatter(x=train_sizes, y=test_rmse_lc, mode='lines+markers', name='Validation Error', line=dict(color='green')))
                    fig_lc.update_layout(xaxis_title="Training Set Size", yaxis_title="RMSE", template="plotly_white", height=450)
                    st.plotly_chart(fig_lc, use_container_width=True, key=f"{key_prefix}_lc")
                except Exception as e:
                    st.warning(f"Could not compute learning curve: {e}")
        current_tab += 1

    # 4. RESIDUALS
    with tabs[current_tab]:
        st.markdown("**Residuals vs Predicted** — click/brush to highlight across charts")
        lowess = sm.nonparametric.lowess(resid, y_pred_arr, frac=0.25)
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred_arr, y=resid, mode="markers",
            marker=dict(size=8, color=anomaly_score, colorscale="Turbo", showscale=True),
            text=hover_texts,
            customdata=np.arange(len(y_pred_arr)),
            name="Residuals"
        ))
        fig_res.add_trace(go.Scatter(x=lowess[:,0], y=lowess[:,1], mode='lines', line=dict(width=3, color='black'), name='LOWESS'))
        fig_res.update_layout(title="Residuals vs Predicted", template="plotly_white", height=450,
                              xaxis=dict(rangeslider=dict(visible=True)))
        
        selected_points = []
        if PLOTLY_EVENTS_AVAILABLE:
            selected = plotly_events(fig_res, click_event=True, select_event=True, key=f"{key_prefix}_resid_events")
            if selected:
                for s in selected:
                    if "customdata" in s:
                        selected_points.append(int(s["customdata"]))
            st.session_state["diag_selected_idx"] = selected_points
            st.plotly_chart(fig_res, use_container_width=True, key=f"{key_prefix}_resid_chart")
        else:
            st.info("Tip: Install `streamlit-plotly-events` for interaction.")
            st.plotly_chart(fig_res, use_container_width=True, key=f"{key_prefix}_resid")
    current_tab += 1

    # 5. DISTRIBUTION
    with tabs[current_tab]:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=resid, nbinsx=50, name='Histogram', opacity=0.7))
        try:
            kde = gaussian_kde(resid)
            x_vals = np.linspace(resid.min(), resid.max(), 300)
            fig_dist.add_trace(go.Scatter(x=x_vals, y=kde(x_vals)*len(resid)*(resid.max()-resid.min())/50, mode='lines', name='KDE'))
        except Exception:
            pass
        fig_dist.add_trace(go.Scatter(x=resid, y=[0]*len(resid), mode='markers', marker=dict(symbol='line-ns-open', size=10), name='Rug'))
        fig_dist.update_layout(title="Residual Distribution", template="plotly_white", height=400)
        st.plotly_chart(fig_dist, use_container_width=True, key=f"{key_prefix}_dist")
    current_tab += 1

    # 6. QQ PLOT
    with tabs[current_tab]:
        qq = probplot(resid, dist='norm')
        theo, obs = qq[0][0], qq[0][1]
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theo, y=obs, mode='markers', name='Data', marker=dict(size=7)))
        fig_qq.add_trace(go.Scatter(x=theo, y=theo, mode='lines', name='Reference', line=dict(color='black')))
        fig_qq.update_layout(title="Q-Q Plot", template="plotly_white", height=400)
        st.plotly_chart(fig_qq, use_container_width=True, key=f"{key_prefix}_qq")
    current_tab += 1

    # 7. STANDARDIZED RESIDUALS
    with tabs[current_tab]:
        fig_std = px.scatter(x=y_pred_arr, y=std_resid, labels={'x':'Predicted','y':'Std Residuals'})
        fig_std.add_hline(y=0, line_dash='dash')
        fig_std.add_hline(y=3, line_dash='dot', line_color='red')
        fig_std.add_hline(y=-3, line_dash='dot', line_color='red')
        fig_std.update_layout(title="Standardized Residuals", template="plotly_white", height=400)
        if st.session_state.get("diag_selected_idx"):
            sel = st.session_state["diag_selected_idx"]
            fig_std.add_trace(go.Scatter(x=y_pred_arr[sel], y=std_resid[sel], mode='markers', marker=dict(size=12, color='red', symbol='x'), name='Selected'))
        st.plotly_chart(fig_std, use_container_width=True, key=f"{key_prefix}_std")
    current_tab += 1

    # 8. INFLUENCE
    with tabs[current_tab]:
        fig_inf = go.Figure()
        fig_inf.add_trace(go.Scatter(
            x=leverage, y=cooks_d, mode='markers',
            marker=dict(size=9, color=anomaly_score, colorscale='Inferno', showscale=True),
            text=hover_texts, customdata=np.arange(len(leverage)), name="Influence"
        ))
        thresh = 4 / max(1, len(y_pred_arr))
        fig_inf.add_hline(y=thresh, line_dash='dash', line_color='red', annotation_text="Cook's threshold")
        fig_inf.update_layout(title="Influence Plot", template="plotly_white", height=450)
        st.plotly_chart(fig_inf, use_container_width=True, key=f"{key_prefix}_inf")
    current_tab += 1

    # 9. 3D VIEWS
    with tabs[current_tab]:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter3d(
            x=y_pred_arr, y=leverage, z=resid, mode='markers',
            marker=dict(size=4, color=anomaly_score, colorscale='Turbo', showscale=True),
            text=hover_texts
        ))
        fig3.update_layout(scene=dict(xaxis_title='Pred', yaxis_title='Leverage', zaxis_title='Residual'), height=500, title="3D Residual Explorer")
        st.plotly_chart(fig3, use_container_width=True, key=f"{key_prefix}_3d_1")
    current_tab += 1

    # 10. ANOMALIES
    with tabs[current_tab]:
        df = pd.DataFrame({
            "index": np.arange(len(y_pred_arr)), "pred": y_pred_arr, "obs": y_test_arr, "resid": resid,
            "anomaly_score": anomaly_score, "anomaly_label": anomaly_label
        })
        label_map = {0: "Normal", 1: "Medium", 2: "Strong"}
        df["anomaly_text"] = df["anomaly_label"].map(label_map)

        st.markdown("**Top 15 Anomalies**")
        st.dataframe(df.sort_values("anomaly_score", ascending=False).head(15))

        fig_a = px.scatter(df, x="pred", y="resid", color="anomaly_text", title="Anomaly Highlighted Residuals")
        st.plotly_chart(fig_a, use_container_width=True, key=f"{key_prefix}_anom_scatter")

    # Final linked selection info
    if st.session_state.get("diag_selected_idx"):
        sel = st.session_state["diag_selected_idx"]
        st.markdown(f"**Selected indices:** {sel}")
        st.dataframe(df[df["index"].isin(sel)])

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

def plot_correlation_heatmap(df, features, target):
    st.markdown("### 📊 Data Correlation Analysis")
    # Sadece seçili featurelar ve target
    cols = features + [target]
    corr = df[cols].corr()
    
    fig = px.imshow(corr, 
                    text_auto=".2f", 
                    aspect="auto", 
                    color_continuous_scale="RdBu_r",
                    origin='lower',
                    title="Feature & Target Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

def plot_radar_chart(results):
    st.markdown("#### 🕸️ Model Comparison Radar")
    
    # Veriyi hazırla
    categories = ['R2', 'RMSE', 'MAE'] 
    
    fig = go.Figure()

    for scenario_name, res in results.items():
        metrics = res['metrics']
        # Basit normalizasyon (örnek amaçlı)
        values = [metrics.get('R2', 0), metrics.get('RMSE', 0), metrics.get('MAE', 0)]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=scenario_name
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Radar (Normalized view recommended)"
    )
    st.plotly_chart(fig, use_container_width=True)

# XAI (SHAP/PFI/LIME)
def run_xai_analysis(model, X_train, X_test, y_test, y_pred, methods, key_suffix):
    """
    XAI analizlerini Pipeline uyumlu hale getirilmiş şekilde çalıştırır.
    """
    if not methods:
        st.info("No XAI methods selected.")
        return

    xai_tabs = st.tabs(methods)
    
    # Pipeline ise içindeki ana modeli ve varsa scaler'ı ayrıştır
    is_pipeline = isinstance(model, Pipeline)
    estimator = model.named_steps['model'] if is_pipeline else model
    
    # SHAP için veriyi hazırlama (Eğer scaler varsa transform etmeliyiz)
    X_test_transformed = X_test.copy()
    if is_pipeline and 'scaler' in model.named_steps:
        scaler = model.named_steps['scaler']
        X_test_transformed = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
    
    # ----- SHAP -----
    if "SHAP" in methods:
        with xai_tabs[methods.index("SHAP")]:
            st.info("Computing SHAP values...")
            
            # Hız ve uyumluluk için örneklem al
            sample_size = min(200, len(X_test))
            X_shap = X_test_transformed.iloc[:sample_size]
            
            try:
                try:
                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(X_shap)
                except Exception:
                    # Fallback: Genel Explainer (Auto-detect)
                    explainer = shap.Explainer(estimator, X_shap)
                    shap_values = explainer(X_shap)
                
                # SHAP value formatını kontrol et
                if isinstance(shap_values, list):
                    vals = shap_values[0]
                elif hasattr(shap_values, 'values'):
                    vals = shap_values.values
                else:
                    vals = shap_values

                # Görselleştirme
                st.markdown("#### SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(vals, X_shap, show=False, feature_names=X_test.columns)
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.warning(f"SHAP calculation encountered an issue: {str(e)}")

    # ----- PFI (Permutation Feature Importance) -----
    if "PFI" in methods:
        with xai_tabs[methods.index("PFI")]:
            st.markdown("#### Permutation Feature Importance")
            try:
                # PFI'ı Pipeline'ın tamamı üzerinde çalıştırıyoruz (Raw Data ile)
                r_pfi = permutation_importance(
                    model, X_test, y_test, 
                    n_repeats=10, 
                    random_state=42, 
                    n_jobs=-1
                )
                
                # Sıralama ve DataFrame
                idx = r_pfi.importances_mean.argsort()[::-1]
                pfi_df = pd.DataFrame({
                    "Feature": X_test.columns[idx],
                    "Importance": r_pfi.importances_mean[idx],
                    "Std Dev": r_pfi.importances_std[idx]
                })
                
                # Grafik
                fig = px.bar(pfi_df.head(15), x='Importance', y='Feature', orientation='h', 
                             error_x="Std Dev", title="Top 15 Feature Importance",
                             color='Importance', color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key=f'pfi_{key_suffix}')
                
                with st.expander("View PFI Data"):
                    st.dataframe(pfi_df)
                    
            except Exception as e:
                st.error(f"PFI Error: {e}")

    # ----- LIME -----
    if "LIME" in methods:
        with xai_tabs[methods.index("LIME")]:
            if lime_tabular is None:
                st.warning("LIME library not found. Please install `lime`.")
            else:
                st.info("Generating LIME explanation for a specific instance.")
                
                # Kullanıcıdan index seçimi
                idx = st.number_input(
                    "Select Instance Index (Test Set)", 
                    min_value=0, 
                    max_value=len(X_test)-1, 
                    value=0, 
                    key=f'lime_idx_{key_suffix}'
                )
                
                try:
                    # LIME Explainer
                    explainer_lime = lime_tabular.LimeTabularExplainer(
                        training_data=np.array(X_train),
                        feature_names=X_train.columns.tolist(),
                        mode='regression',
                        verbose=False
                    )
                    
                    # Explain Instance
                    exp = explainer_lime.explain_instance(
                        data_row=X_test.iloc[idx],
                        predict_fn=model.predict,
                        num_features=10
                    )
                    
                    # HTML Gösterimi
                    components.html(exp.as_html(), height=500, scrolling=True)
                    
                except Exception as e:
                    st.error(f"LIME Error: {e}")

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

            # XAI Page
            fig_xai = plt.figure(figsize=(8.27, 11.69))
            fig_xai.suptitle(f"XAI Analysis: {scenario_name}", fontsize=14, fontweight='bold')
            
            gs = fig_xai.add_gridspec(2, 1, hspace=0.3)
            ax_imp = fig_xai.add_subplot(gs[0])
            
            try:
                # Try built-in feature importance first
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[-10:] 
                    ax_imp.barh(range(len(indices)), importances[indices], align='center', color='teal')
                    ax_imp.set_yticks(range(len(indices)))
                    ax_imp.set_yticklabels([X_test.columns[i] for i in indices])
                    ax_imp.set_title("Top 10 Feature Importances (Built-in)")
                elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'feature_importances_'):
                     importances = model.named_steps['model'].feature_importances_
                     indices = np.argsort(importances)[-10:] 
                     ax_imp.barh(range(len(indices)), importances[indices], align='center', color='teal')
                     ax_imp.set_yticks(range(len(indices)))
                     ax_imp.set_yticklabels([X_test.columns[i] for i in indices])
                     ax_imp.set_title("Top 10 Feature Importances (Pipeline)")
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
            ax_shap.set_title("SHAP Summary Plot")
            
            try:
                sample_size = min(100, len(X_test))
                X_sample = X_test.iloc[:sample_size]
                
                explainer = None
                shap_values = None
                
                # Check if pipeline
                estimator = model.named_steps['model'] if isinstance(model, Pipeline) else model
                
                try:
                    explainer = shap.TreeExplainer(estimator)
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

st.title("🚀 Model Diagnostic Dashboard – Full Edition")
st.markdown("All-in-one: UI polish, PDF export, extra models, TimeSeriesSplit, progress bars, SHAP/LIME/PFI.")

# Sidebar
with st.sidebar.expander("### Upload & Settings", expanded=False):
    uploaded_file = st.file_uploader("Upload CSV (must include target column)", type=['csv'])
if not uploaded_file:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

try:
        df_original = load_data(uploaded_file)
        st.success("Loaded CSV")
except Exception as e:
        st.error(f"Failed reading CSV: {e}")
        st.stop()


# Date handling
cols = df_original.columns.tolist()
date_guess = [c for c in cols if 'date' in c.lower() or 'time' in c.lower()]

with st.sidebar.expander("### Selected Date Column / Target / Feature ", expanded=False):

    date_col = st.selectbox('Date column (optional)', [None] + cols, index=0)
    if date_col:
        try:
            df_original[date_col] = pd.to_datetime(df_original[date_col])
            df_original.set_index(date_col, inplace=True)
            df_original = df_original.sort_index()
        except Exception:
            st.warning('Could not parse date column - leaving as-is')

    # Target & features
    target_default = 'Power' if 'Power' in cols else cols[-1]
    target_col = st.selectbox('Target column', cols, index=cols.index(target_default))
    feature_cols = st.multiselect('Feature columns', [c for c in cols if c!=target_col], default=[c for c in cols if c!=target_col][:6])
if not feature_cols:
    st.error('Select at least one feature')
    st.stop()

# Preproc
with st.sidebar.expander("### Data Processing ", expanded=False):
    outlier_methods = st.multiselect('Outlier handling (Training Data Only)', ["IQR Capping","Z-Score Capping","Isolation Forest (Drop)"])
    scaling_methods = st.multiselect('Scaling (Fit on Train, Transform Test)', ["Min-Max Scaling (0-1)","Standard Scaling (Z-Score)","Robust Scaling (IQR based)","MaxAbs Scaling (-1 to 1)","Log Transformation (np.log1p)"])

# Models
with st.sidebar.expander("### Selected Models ", expanded=False):
    available = ["HistGradientBoosting","RandomForest","GradientBoosting","CatBoost","XGBoost","LightGBM"]
    avail_filtered = [m for m in available if not (m=="XGBoost" and XGBRegressor is None) and not (m=="LightGBM" and LGBMRegressor is None) and not (m=="CatBoost" and CatBoostRegressor is None)]
    selected_models = st.multiselect('Models', avail_filtered, default=[avail_filtered[0]])

# Metrics
with st.sidebar.expander("### Selected Metrics ", expanded=False):
    metric_options = ["RMSE", "MAE", "R2", "MSE", "MAPE", "MedAE"]
    selected_metrics = st.multiselect('Evaluation Metrics', metric_options, default=["MSE", "RMSE", "MAE", "R2"])
    if not selected_metrics:
        st.error("Please select at least one metric.")
        st.stop()

# HPO and XAI
hpo_methods = st.sidebar.multiselect('HPO Methods', ['Random Search','Grid Search'])
xai_methods = st.sidebar.multiselect('XAI Methods', ['SHAP','PFI','LIME'])

with st.sidebar.expander("### Visualization ", expanded=False):
    use_timesplit = st.checkbox('Use Time-series split (train on earliest, test on latest)', value=False)
    show_combined = st.checkbox('Show combined forecast', value=False)
    show_forecasts = st.checkbox('Show scenario forecasts', value=False)
    show_diags = st.checkbox('Show diagnostics', value=False)
    # show_cor_heatmap = st.checkbox('Show Corelation Heatmap', value=False)
    # show_learning_curve = st.checkbox('Show Learning Curves', value=False)
    # show_actual_vs_pred_advanced = st.checkbox('Show Regression Plot with Marginals', value=False)
    # show_radar_chart = st.checkbox('Show Spider Chart', value=False)


train_btn = st.sidebar.button('Train & Run All')

# --- EĞİTİM ALGORİTMASI ---
if 'training_complete' not in st.session_state:
    st.session_state['training_complete'] = False

if train_btn and not st.session_state['training_complete']:
    st.info('Starting training — this may take some time depending on models and HPO.')
    
    # Tarih sıralaması kontrolü (Time Series Split için)
    if use_timesplit and not df_original.index.is_monotonic_increasing:
        st.warning("⚠️ Time Series Split selected but index is not sorted. Sorting now...")
        df_original = df_original.sort_index()

    results = {}
    progress = st.progress(0)
    
    # Senaryo Hazırlığı
    scenarios = []
    
    # 1. Default
    scenarios.append({
        "suffix": "Default",
        "hpo": None,
        "use_prep": False
    })

    # 2. Preprocessed
    if outlier_methods or scaling_methods:
        scenarios.append({
            "suffix": "Preprocessed",
            "hpo": None,
            "use_prep": True
        })
        
    # 3. HPO
    for hpo in hpo_methods:
        scenarios.append({
            "suffix": f"HPO ({hpo})",
            "hpo": hpo,
            "use_prep": True if (outlier_methods or scaling_methods) else False
        })

    total_tasks = len(selected_models) * len(scenarios)
    task_idx = 0

    # Ana Döngü
    for model_name in selected_models:
        for scen in scenarios:
            label = f"{model_name} - {scen['suffix']}"
            
            current_outlier = outlier_methods if scen['use_prep'] else None
            current_scaling = scaling_methods if scen['use_prep'] else None
            
            X_raw = df_original[feature_cols]
            y_raw = df_original[target_col]
            
            def prog_cb(p): 
                val = min(100, int((task_idx/total_tasks*100) + p/total_tasks))
                progress.progress(val)

            try:
                metrics, model, Xt, Xte, ytr, yte, ytr_pr, ypr = train_and_evaluate(
                    X_raw, y_raw, 
                    test_size=0.2, 
                    model_name=model_name, 
                    hpo_method=scen['hpo'], 
                    use_timesplit=use_timesplit, 
                    _progress_callback=prog_cb, 
                    active_metrics=selected_metrics,
                    outlier_methods=current_outlier,
                    scaling_methods=current_scaling
                )
                
                results[label] = {
                    'metrics': metrics,
                    'model': model,
                    'Xt': Xt,
                    'Xte': Xte,
                    'ytr': ytr,
                    'yte': yte,
                    'ytr_pr': ytr_pr,
                    'ypr': ypr
                }

            except Exception as e:
                st.error(f"Error in {label}: {e}")
            
            task_idx += 1
            progress.progress(min(100, int(task_idx/total_tasks*100)))

    st.success('Training complete')
    st.session_state['results_all'] = results
    st.session_state['training_complete'] = True
    st.rerun() 

# --- SONUÇLARI GÖSTERME ---
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
                        
                        display_diagnostic_plots(
                            y_test=sel['yte'], 
                            y_pred=sel['ypr'], 
                            y_train=sel['ytr'],        
                            y_train_pred=sel['ytr_pr'],
                            model=sel['model'],    
                            X_train=sel['Xt'], 
                            key_prefix=f"diag_{i}"
                        )
                        st.divider()
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
    with st.expander("### 📥 Export Full Report", expanded=False):        
    
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
