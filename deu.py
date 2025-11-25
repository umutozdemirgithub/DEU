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
from lime.lime_tabular import LimeTabularExplainer

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

# Opsiyonel Kütüphaneler (Hata almamak için try-except blokları)
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

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

# Uyarıları bastır
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Model Diagnostic Dashboard (Optimized)", layout="wide")

# --- SIDEBAR: STYLING & LOGO ---
st.sidebar.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        color: white;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-weight: 600;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
st.sidebar.title("AutoML Dashboard")

# -------------------------------------------------------------------------
# 1. CACHED UTILITIES (PERFORMANS İÇİN KRİTİK BÖLÜM)
# -------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """CSV dosyasını okur ve cache'ler."""
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            return None
    return None

@st.cache_data(show_spinner=False)
def preprocess_dataframe(df, date_col_name=None):
    """
    Veri seti ön işlemesini (Tarih formatlama, index atama) önbelleğe alır.
    Her buton tıklamasında veri setini baştan aşağı tekrar işlemez.
    """
    df_proc = df.copy()
    if date_col_name:
        try:
            df_proc[date_col_name] = pd.to_datetime(df_proc[date_col_name])
            df_proc.set_index(date_col_name, inplace=True)
            df_proc = df_proc.sort_index()
        except Exception:
            pass 
    return df_proc

def st_shap(plot, height=None):
    shap_html = f"{shap.getjs()}{plot.html()}"
    components.html(shap_html, height=height)

# -------------------------------------------------------------------------
# 2. MODEL FACTORY & CONFIGURATION
# -------------------------------------------------------------------------
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
        "n_jobs": 2,
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

# Farklı kütüphane parametrelerini ortak isimlere eşleyen sözlük
UNIFIED_PARAM_NAMES = {
    # --- Ağaç / İterasyon Sayısı ---
    "n_estimators": "Estimators / Iter",
    "max_iter": "Estimators / Iter",
    "iterations": "Estimators / Iter",
    
    # --- Öğrenme Hızı ---
    "learning_rate": "Learning Rate",
    
    # --- Ağaç Derinliği ---
    "max_depth": "Max Depth",
    "depth": "Max Depth",
    
    # --- Regularization (L2 & L1) ---
    "l2_regularization": "L2 Regularization",
    "reg_lambda": "L2 Regularization",
    "l2_leaf_reg": "L2 Regularization",
    "reg_alpha": "L1 Regularization",
    
    # --- Veri ve Özellik Örnekleme ---
    "subsample": "Subsample Ratio",
    "max_features": "Col Sample / Max Feat",
    "colsample_bytree": "Col Sample / Max Feat",
    
    # --- Yaprak Kısıtlamaları ---
    "min_samples_leaf": "Min Samples Leaf",
    "min_samples_split": "Min Samples Split",
    "num_leaves": "Max Leaves (LGBM)",
}

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

def perform_hpo(X_train, y_train, method, model_name, use_timesplit=False, scaler_cls=None):
    # 1) Base Model & Pipeline
    base_model = safe_model_factory(model_name)
    steps = []
    if scaler_cls is not None:
        steps.append(("scaler", scaler_cls()))
    steps.append(("model", base_model))
    pipeline = Pipeline(steps)

    # 2) CV Strategy
    if use_timesplit:
        cv_strategy = TimeSeriesSplit(n_splits=3)
    else:
        cv_strategy = 3

    # 3) HPO Spaces
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

    # 4) Filter params that exist in MODEL_DEFAULT_PARAMS
    if model_name in MODEL_DEFAULT_PARAMS:
        allowed = {f"model__{p}" for p in MODEL_DEFAULT_PARAMS[model_name].keys()}
        random_space = {k: v for k, v in random_space.items() if k in allowed}
        grid_space   = {k: v for k, v in grid_space.items()   if k in allowed}

    selected_space = random_space if method == "Random Search" else grid_space
    
    if not selected_space:
        return pipeline

    # 5) Search
    if method == "Random Search":
        search_engine = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=selected_space,
            n_iter=15, # Performance tweak
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=2,
            verbose=0,
        )
    else:
        search_engine = GridSearchCV(
            estimator=pipeline,
            param_grid=selected_space,
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
            n_jobs=2,
            verbose=0,
        )

    search_engine.fit(X_train, y_train)
    return search_engine.best_estimator_

# -------------------------------------------------------------------------
# 3. TRAINING & EVALUATION
# -------------------------------------------------------------------------
def train_and_evaluate(X, y, test_size, model_name, hpo_method=None, use_timesplit=False, 
                       _progress_callback=None, active_metrics=None, 
                       outlier_methods=None, scaling_methods=None):
    
    # 1. SPLIT
    if use_timesplit:
        tss = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:tss].copy(), X.iloc[tss:].copy()
        y_train, y_test = y.iloc[:tss].copy(), y.iloc[tss:].copy()
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        X_train, X_test = X_train.copy(), X_test.copy()
        y_train, y_test = y_train.copy(), y_test.copy()

    # 2. OUTLIER HANDLING (Train only)
    if outlier_methods:
        train_df = pd.concat([X_train, y_train], axis=1)
        target_col = y_train.name
        feat_cols = X_train.columns.tolist()
        num_cols = train_df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
        df_clean = train_df.copy()

        if "IQR Capping" in outlier_methods and num_cols:
            for col in num_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                df_clean[col] = df_clean[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

        if "Z-Score Capping" in outlier_methods and num_cols:
            for col in num_cols:
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                df_clean[col] = df_clean[col].clip(lower=mean_val - 3 * std_val, upper=mean_val + 3 * std_val)

        if "Isolation Forest Drop" in outlier_methods and num_cols:
            try:
                iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=1)
                yhat = iso.fit_predict(df_clean[num_cols])
                df_clean = df_clean[yhat != -1]
            except Exception:
                pass

        X_train = df_clean[feat_cols]
        y_train = df_clean[target_col]

    # 3. SCALER SELECTION
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
        # Basit pozitif kontrol
        if (X_train.values >= 0).all():
            X_train = np.log1p(X_train)
            X_test = np.log1p(X_test)

    if _progress_callback: _progress_callback(10)

    # 4. TRAIN
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

    # 5. PREDICT
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # 6. METRICS
    metrics = {}
    if active_metrics is None: active_metrics = ["MSE","RMSE", "MAE", "R2"]

    if "MSE" in active_metrics: metrics['MSE'] = mean_squared_error(y_test, y_pred)
    if "RMSE" in active_metrics: metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
    if "MAE" in active_metrics: metrics['MAE'] = mean_absolute_error(y_test, y_pred)
    if "R2" in active_metrics: metrics['R2'] = r2_score(y_test, y_pred)
    if "MAPE" in active_metrics: metrics['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)
    if "MedAE" in active_metrics: metrics['MedAE'] = median_absolute_error(y_test, y_pred)
    
    if _progress_callback: _progress_callback(100)

    return metrics, model, X_train, X_test, y_train, y_test, y_train_pred, y_pred

# -------------------------------------------------------------------------
# 4. DIAGNOSTICS - CACHED CALCULATIONS
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calculate_diagnostic_metrics(y_test_arr, y_pred_arr):
    """
    Ağır matematiksel işlemleri (Cook's D, Isolation Forest vb.) önbelleğe alır.
    Grafik çiziminden ayrıştırılmıştır.
    """
    resid = y_test_arr - y_pred_arr
    resid_mean = np.mean(resid)
    resid_std = np.std(resid)
    std_resid = (resid - resid_mean) / (resid_std + 1e-9)

    # Influence
    try:
        X = sm.add_constant(pd.Series(y_pred_arr))
        model_ols = sm.OLS(pd.Series(y_test_arr), X).fit()
        influence = model_ols.get_influence()
        leverage = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]
    except Exception:
        leverage = np.zeros_like(y_pred_arr, dtype=float)
        cooks_d = np.zeros_like(y_pred_arr, dtype=float)

    # Anomaly Ensemble Logic
    scores = np.zeros((len(resid), 4))
    
    # 1. Isolation Forest
    try:
        if_model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42, n_jobs=1)
        if_model.fit(resid.reshape(-1, 1))
        if_score = -if_model.score_samples(resid.reshape(-1, 1))
        scores[:, 0] = (if_score - if_score.min()) / (np.ptp(if_score) + 1e-9)
    except: pass

    # 2. LOF
    try:
        lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
        lof_pred = lof.fit_predict(resid.reshape(-1, 1))
        lof_score = -lof.negative_outlier_factor_
        scores[:, 1] = (lof_score - lof_score.min()) / (np.ptp(lof_score) + 1e-9)
    except: pass
    
    # 3. 3-Sigma
    scores[:, 3] = (np.abs(resid) > 3 * resid_std).astype(float)

    # Combine
    agg = scores.mean(axis=1)
    norm_agg = (agg - agg.min()) / (np.ptp(agg) + 1e-9)
    
    labels = np.zeros_like(norm_agg, dtype=int)
    labels[norm_agg >= 0.75] = 2 # Strong
    labels[(norm_agg >= 0.4) & (norm_agg < 0.75)] = 1 # Medium
    
    return resid, std_resid, leverage, cooks_d, norm_agg, labels

def display_diagnostic_plots(
    y_test, y_pred, y_train=None, y_train_pred=None, 
    model=None, X_train=None, 
    key_prefix="diag",
    active_plots=None
):
    """
    Hızlı ve optimize edilmiş diagnostic plots. 
    Sadece seçilen grafikleri çizer. Overfitting Check sonradan seçilse bile donmaz.
    """

    if not active_plots:
        st.info("No diagnostic plots selected in the sidebar.")
        return

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    
    # --- Cached metrics ---
    resid, std_resid, leverage, cooks_d, anomaly_score, anomaly_label = calculate_diagnostic_metrics(y_test_arr, y_pred_arr)
    
    hover_texts = [f"Resid: {r:.2f}<br>Cook's D: {c:.4f}" for r, c in zip(resid, cooks_d)]
    has_train = (y_train is not None) and (y_train_pred is not None)

    # --- Overfitting metrics (cached/precomputed) ---
    if has_train:
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
    else:
        train_rmse = test_rmse = None

    # --- Dynamic tabs, remove overfitting if no train ---
    final_tabs = [p for p in active_plots]
    if "Overfitting Check" in final_tabs and not has_train:
        final_tabs.remove("Overfitting Check")
        
    if not final_tabs:
        st.warning("No valid diagnostic plots available (train/test mismatch).")
        return

    # --- Create tabs ---
    tabs = st.tabs(final_tabs)

    layout_opts = dict(template='plotly_white', height=500)

    for i, tab_name in enumerate(final_tabs):
        with tabs[i]:

            # --- 1. Advanced Scatter ---
            if tab_name == "Adv. Scatter":
                df_chart = pd.DataFrame({'Actual': y_test_arr, 'Predicted': y_pred_arr})
                fig_adv = px.scatter(
                    df_chart, x='Actual', y='Predicted', 
                    trendline="ols", trendline_color_override="red",
                    marginal_x="histogram", marginal_y="histogram",
                    opacity=0.6, height=layout_opts['height'],
                    title="Actual vs Predicted"
                )
                st.plotly_chart(fig_adv, use_container_width=True, key=f"{key_prefix}_adv")

            # --- 2. Overfitting Check ---
            elif tab_name == "Overfitting Check" and has_train:
                # RMSE Hesapla
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))

                # Grafik
                fig = go.Figure(go.Bar(
                    x=["Train", "Test"], y=[train_rmse, test_rmse],
                    marker_color=["#1f77b4", "#ff7f0e"],
                    text=[f"{train_rmse:.4f}", f"{test_rmse:.4f}"], textposition="auto"
                ))
                fig.update_layout(title="RMSE Comparison (Train vs Test)", **layout_opts)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_over")

                # --- Overfitting Değerlendirmesi ---
                # Dinamik eşik (%20 fark varsayılan)
                overfit_threshold = 0.2
                rmse_diff = test_rmse - train_rmse
                ratio = test_rmse / train_rmse if train_rmse > 0 else np.inf

                if ratio > 1 + overfit_threshold:
                    st.warning(
                        f"⚠️ Potential overfitting detected!\n"
                        f"Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}, Δ = {rmse_diff:.4f} ({ratio:.2f}×)"
                    )
                else:
                    st.success(
                        f"✅ No significant overfitting detected.\n"
                        f"Train RMSE = {train_rmse:.4f}, Test RMSE = {test_rmse:.4f}, Δ = {rmse_diff:.4f}"
                    )

            # --- 3. Residuals ---
            elif tab_name == "Residuals":
                fig_res = px.scatter(
                    x=y_pred_arr, y=resid,
                    color=anomaly_score, 
                    color_continuous_scale="Turbo",
                    labels={'x': 'Predicted', 'y': 'Residual'},
                    hover_data={'Residual': resid, "Cook's D": cooks_d},
                    title="Residuals vs Predicted",
                    height=layout_opts['height']
                )
                fig_res.add_hline(y=0, line_dash='dash', line_color='black')
                st.plotly_chart(fig_res, use_container_width=True, key=f"{key_prefix}_res")

            # --- 4. Residual Distribution ---
            elif tab_name == "Distribution":
                fig_dist = px.histogram(
                    resid, nbins=50, marginal="box", 
                    labels={'value': 'Residual'}, 
                    title="Residual Distribution",
                    height=layout_opts['height']
                )
                st.plotly_chart(fig_dist, use_container_width=True, key=f"{key_prefix}_dist")

            # --- 5. QQ Plot ---
            elif tab_name == "QQ Plot":
                qq = probplot(resid, dist='norm')
                fig_qq = px.scatter(
                    x=qq[0][0], y=qq[0][1], 
                    labels={'x': 'Theoretical', 'y': 'Observed'}, 
                    title="Q-Q Plot", height=layout_opts['height']
                )
                fig_qq.add_shape(
                    type="line", x0=min(qq[0][0]), y0=min(qq[0][0]),
                    x1=max(qq[0][0]), y1=max(qq[0][0]), line=dict(color="red")
                )
                st.plotly_chart(fig_qq, use_container_width=True, key=f"{key_prefix}_qq")

            # --- 6. Influence ---
            elif tab_name == "Influence":
                fig_inf = px.scatter(
                    x=leverage, y=cooks_d, color=anomaly_score,
                    color_continuous_scale="Viridis",
                    labels={'x': 'Leverage', 'y': "Cook's D"},
                    title="Influence Plot",
                    height=layout_opts['height']
                )
                fig_inf.add_hline(y=4/len(resid), line_dash="dash", line_color="red")
                st.plotly_chart(fig_inf, use_container_width=True, key=f"{key_prefix}_inf")

            # --- 7. Anomalies ---
            elif tab_name == "Anomalies":
                df_anom = pd.DataFrame({
                    "Actual": y_test_arr, "Predicted": y_pred_arr, 
                    "Residual": resid, "AnomalyScore": anomaly_score, 
                    "Label": anomaly_label
                }).sort_values("AnomalyScore", ascending=False)
                st.dataframe(df_anom.head(50), use_container_width=True, height=400)

# -------------------------------------------------------------------------
# 5. XAI (SHAP/LIME/PFI) - CACHED
# -------------------------------------------------------------------------
# -------------------------
# SHAP Calculation - Cached
# -------------------------
@st.cache_data(show_spinner=True)
def calculate_shap_values(_model, X_test_sample):
    estimator = _model
    if isinstance(_model, Pipeline) and 'model' in _model.named_steps:
        estimator = _model.named_steps['model']

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test_sample)
    except:
        explainer = shap.Explainer(estimator, X_test_sample)
        shap_values = explainer(X_test_sample)

    vals = shap_values
    if isinstance(vals, list):
        vals = vals[0]
    elif hasattr(vals, 'values'):
        vals = vals.values
    return vals

# -------------------------
# LIME Aggregated Feature Importance
# -------------------------
def compute_lime_feature_importance(model, X_train, X_test, sample_size=50, num_features=12):
    if lime_tabular is None:
        raise RuntimeError("LIME is not installed")

    n = min(sample_size, len(X_test))
    X_sample = X_test.iloc[:n]

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        mode="regression",
        discretize_continuous=True,
        verbose=False
    )

    feature_names = X_train.columns.tolist()
    acc = {f: [] for f in feature_names}

    for i in range(n):
        try:
            exp = explainer.explain_instance(
                data_row=X_sample.iloc[i].values,
                predict_fn=lambda x: model.predict(x),
                num_features=num_features
            )
            for feat, weight in exp.as_list():
                base = feat.split(' ')[0]
                if base in acc:
                    acc[base].append(abs(weight))
                else:
                    for col in feature_names:
                        if col in feat:
                            acc[col].append(abs(weight))
                            break
        except:
            continue

    rows = [(f, float(np.mean(vals)) if vals else 0.0, len(vals)) for f, vals in acc.items()]
    imp_df = pd.DataFrame(rows, columns=["Feature", "Mean Abs Weight", "Count"]).sort_values("Mean Abs Weight", ascending=False)
    return imp_df

# -------------------------
# SHAP Importance DataFrame
# -------------------------
def shap_importance_df(shap_values, X_shap):
    vals = shap_values
    if isinstance(vals, list):
        vals = np.array(vals[0])
    elif hasattr(vals, 'values'):
        vals = np.array(vals.values)

    if vals.ndim == 3:
        vals = vals[0]

    abs_vals = np.mean(np.abs(vals), axis=0)
    imp_df = pd.DataFrame({'Feature': X_shap.columns.tolist(), 'Importance': abs_vals}).sort_values('Importance', ascending=False)
    return imp_df

# -------------------------
# Combined Radar Chart (PFI + SHAP)
# -------------------------
def create_combined_radar(pfi_df, shap_df=None, top_k=10, title="PFI + SHAP Radar"):
    pfi_top = pfi_df.groupby('Feature').agg({'Importance':'mean'}).reset_index().sort_values('Importance', ascending=False).head(top_k)
    features = list(pfi_top['Feature'])

    shap_dict = {}
    if shap_df is not None:
        shap_top = shap_df.groupby('Feature').agg({'Importance':'mean'}).reset_index().sort_values('Importance', ascending=False).head(top_k)
        features = list(pd.Index(features).union(shap_top['Feature']))
        shap_dict = {r['Feature']: r['Importance'] for _, r in shap_top.iterrows()}

    pfi_dict = {r['Feature']: r['Importance'] for _, r in pfi_top.iterrows()}
    features = features[:top_k]

    def norm(x):
        x = np.array(x, dtype=float)
        return list(x/x.max()) if x.max() != 0 else list(x)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm([pfi_dict.get(f,0) for f in features]+[pfi_dict.get(features[0],0)]), theta=features+[features[0]], fill='toself', name='PFI'))
    fig.add_trace(go.Scatterpolar(r=norm([shap_dict.get(f,0) for f in features]+[shap_dict.get(features[0],0)]), theta=features+[features[0]], fill='toself', name='SHAP'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, title=title)
    return fig

# -------------------------
# Main XAI Runner
# -------------------------
def run_xai_analysis(model, X_train, X_test, y_test, methods, key_suffix, lime_sample=50, lime_num_features=12, radar_top_k=12):
    if not methods:
        st.info("No XAI methods selected.")
        return

    xai_tabs = st.tabs(methods)
    is_pipeline = isinstance(model, Pipeline)
    estimator = model.named_steps['model'] if is_pipeline else model

    shap_imp_df = None
    if 'SHAP' in methods and shap is not None:
        with xai_tabs[methods.index('SHAP')]:
            st.markdown('## 🌈 SHAP Global & Local Explanations')
            try:
                X_shap = X_test.iloc[:min(300,len(X_test))].copy()
                if is_pipeline and 'scaler' in model.named_steps:
                    try: X_shap = pd.DataFrame(model.named_steps['scaler'].transform(X_shap), columns=X_shap.columns, index=X_shap.index)
                    except: pass
                try: explainer = shap.TreeExplainer(estimator); shap_values = explainer.shap_values(X_shap)
                except: explainer = shap.Explainer(estimator, X_shap); shap_values = explainer(X_shap)
                shap_imp_df = shap_importance_df(shap_values, X_shap)

                shap_tabs = st.tabs(['🎯 Summary Plot', '📊 Feature Importance', '🔎 Instance Waterfall'])
                with shap_tabs[0]:
                    fig, ax = plt.subplots(figsize=(7,5))
                    shap.summary_plot(shap_values, X_shap, show=False)
                    st.pyplot(fig)

                with shap_tabs[1]:
                    st.dataframe(shap_imp_df)
                    fig_imp = px.bar(shap_imp_df.head(20), x='Importance', y='Feature', orientation='h', title='Top SHAP Feature Importance')
                    fig_imp.update_layout(template='plotly_white', height=450)
                    # FIX: Unique key added
                    st.plotly_chart(fig_imp, use_container_width=True, key=f"shap_imp_{key_suffix}")

                with shap_tabs[2]:
                    idx_w = st.number_input('Select Instance Index', 0, len(X_shap)-1, 0, key=f'shap_w_input_{key_suffix}')
                    vals = shap_values.values if hasattr(shap_values,'values') else shap_values
                    if isinstance(vals,list): vals=np.array(vals[0])
                    if vals.ndim==3: vals=vals[0]
                    df_w = pd.DataFrame({'Feature': X_shap.columns.tolist(), 'SHAP': vals[idx_w]})
                    df_w = df_w.sort_values('SHAP', key=np.abs, ascending=False).head(12)
                    df_w['Contribution'] = df_w['SHAP'].cumsum() + float(explainer.expected_value if not isinstance(explainer.expected_value,(list,np.ndarray)) else explainer.expected_value[0])
                    fig_w = px.bar(df_w, x='SHAP', y='Feature', orientation='h', title=f'SHAP Waterfall – Instance {idx_w}', color='SHAP')
                    fig_w.update_layout(template='plotly_white', height=450)
                    # FIX: Unique key added
                    st.plotly_chart(fig_w, use_container_width=True, key=f"shap_waterfall_{key_suffix}")
            except Exception as e: st.error(f'SHAP Error: {e}')

    pfi_imp_df = None
    if 'PFI' in methods:
        with xai_tabs[methods.index('PFI')]:
            st.markdown('## 🚀 Permutation Feature Importance (PFI)')
            try:
                r = permutation_importance(estimator, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
                idx = r.importances_mean.argsort()[::-1]
                pfi_imp_df = pd.DataFrame({'Feature': X_test.columns[idx], 'Importance': r.importances_mean[idx], 'Std Dev': r.importances_std[idx]})
                st.dataframe(pfi_imp_df)
                fig_pfi = px.bar(pfi_imp_df.head(20), x='Importance', y='Feature', orientation='h', error_x='Std Dev', title='Top 20 PFI Features')
                fig_pfi.update_layout(template='plotly_white', height=450)
                # FIX: Unique key added
                st.plotly_chart(fig_pfi, use_container_width=True, key=f"pfi_{key_suffix}")
            except Exception as e: st.error(f'PFI Error: {e}')

    if 'LIME' in methods:
        with xai_tabs[methods.index('LIME')]:
            st.markdown('## 💡 LIME Local Explanations (with aggregated table)')
            if lime_tabular is None: st.warning('LIME is not installed.')
            else:
                try:
                    explainer_lime = lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), mode='regression', discretize_continuous=True, verbose=False)
                    idx = st.number_input('Select Test Instance', 0, len(X_test)-1, 0, key=f'lime_idx_{key_suffix}')
                    exp = explainer_lime.explain_instance(X_test.iloc[idx].values, lambda x:model.predict(x), num_features=12)
                    if components is not None: components.html(exp.as_html(), height=450, scrolling=True)
                    else: st.write(exp.as_list())
                    limedf = compute_lime_feature_importance(model, X_train, X_test, sample_size=lime_sample, num_features=lime_num_features)
                    st.dataframe(limedf)
                    fig_lime = px.bar(limedf.head(20), x='Mean Abs Weight', y='Feature', orientation='h', title='Top LIME Features (mean abs weight)')
                    fig_lime.update_layout(template='plotly_white', height=450)
                    # FIX: Unique key added
                    st.plotly_chart(fig_lime, use_container_width=True, key=f"lime_{key_suffix}")
                except Exception as e: st.error(f'LIME Error: {e}')

    if ('PFI' in methods or 'SHAP' in methods):
        try:
            with st.expander('📡 PFI + SHAP Combined Radar'):
                shap_df = shap_imp_df if shap_imp_df is not None else None
                fig_radar = create_combined_radar(pfi_imp_df if pfi_imp_df is not None else pd.DataFrame({'Feature':[], 'Importance':[]}), shap_df, top_k=radar_top_k)
                # FIX: Unique key added
                st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{key_suffix}")
        except Exception as e: st.error(f'Radar chart generation failed: {e}')

    try:
        with st.expander('🗂️ Automatic XAI Dashboard (Summary)'):
            if pfi_imp_df is not None: st.table(pfi_imp_df.head(10))
            if shap_imp_df is not None: st.table(shap_imp_df.head(10))
            if 'LIME' in methods:
                limedf_small = compute_lime_feature_importance(model, X_train, X_test, sample_size=min(30,len(X_test)), num_features=lime_num_features)
                st.table(limedf_small.head(10))
            if pfi_imp_df is not None and shap_imp_df is not None:
                merged = pfi_imp_df[['Feature','Importance']].merge(shap_imp_df[['Feature','Importance']], on='Feature', how='outer', suffixes=('_PFI','_SHAP')).fillna(0)
                merged['PFI_rank'] = merged['Importance_PFI'].rank(ascending=False)
                merged['SHAP_rank'] = merged['Importance_SHAP'].rank(ascending=False)
                merged['rank_diff'] = (merged['PFI_rank']-merged['SHAP_rank']).abs()
                st.dataframe(merged.sort_values('rank_diff').head(20))
    except: pass

    return {'pfi': pfi_imp_df, 'shap': shap_imp_df}

# -------------------------
# Multi-Model XAI Runner
# -------------------------
def run_models_xai(models: dict, X_train, X_test, y_test, methods=['SHAP','PFI','LIME'], key_prefix='model'):
    results = {}
    for i, (name, m) in enumerate(models.items()):
        with st.expander(f"Model: {name}"):
            st.markdown(f"## Model: {name}")
            results[name] = run_xai_analysis(m, X_train, X_test, y_test, methods, key_suffix=f"{key_prefix}_{i}")
    return results

def run_dashboard(results, selected_diag_plots, xai_ops):
    if not results:
        st.info("👈 Please upload data and click 'Start Training' in the sidebar.")
        return
    
    # Leaderboard
    with st.expander("### 🏆 Leaderboard", expanded=True):
        lb_tabs = st.tabs(["📋 Table", "📈 Chart", "⚙️ Parameters"])

        # --- Tab 1: Table  ---
        with lb_tabs[0]:
            df_res = pd.DataFrame([{**{'Model': k}, **v['metrics']} for k, v in results.items()])
            
            valid_cols = df_res.columns.tolist()
            
            min_cols = [c for c in ["MSE", "RMSE", "MAE", "MAPE", "MedAE"] if c in valid_cols]
            max_cols = [c for c in ["R2"] if c in valid_cols]

            styler = df_res.style.format(precision=4)

            if min_cols:
                styler = styler.highlight_min(subset=min_cols, color='lightblue')
            if max_cols:
                styler = styler.highlight_max(subset=max_cols, color='red')

            st.dataframe(styler, use_container_width=True)

        # --- Tab 2: Chart  ---
        with lb_tabs[1]:
            df_res = pd.DataFrame([{**{'Model': k}, **v['metrics']} for k, v in results.items()])
            
            fig = go.Figure()
            defined_colors = {
                "MSE": "blue",
                "RMSE": "red",
                "MAE": "orange",
                "MAPE": "purple",
                "MedAE": "brown",
                "R2": "green"
            }
            
            metrics_to_plot = [col for col in df_res.columns if col != 'Model' and col != 'R2']

            if not metrics_to_plot and "R2" not in df_res.columns:
                st.warning("Grafik çizmek için hesaplanmış metrik bulunamadı.")
            else:
                for metric in metrics_to_plot:
                    fig.add_trace(go.Bar(
                        x=df_res['Model'],
                        y=df_res[metric],
                        name=metric,
                        marker_color=defined_colors.get(metric, "#1f77b4"),
                        text=[f"{v:.4f}" for v in df_res[metric]],
                        textposition="inside",
                        insidetextanchor="middle",
                        hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.6f}}<extra></extra>"
                    ))
                
                if "R2" in df_res.columns:
                    min_size = 12
                    max_size = 30
                    r2_min = df_res["R2"].min()
                    r2_max = df_res["R2"].max()
                    if r2_max != r2_min:
                        sizes = min_size + (df_res["R2"] - r2_min) / (r2_max - r2_min) * (max_size - min_size)
                    else:
                        sizes = [ (min_size + max_size)/2 ] * len(df_res)
                    
                    fig.add_trace(go.Scatter(
                        x=df_res['Model'],
                        y=df_res["R2"],
                        mode='markers+lines',
                        marker=dict(
                            symbol='star',
                            size=sizes,
                            color=defined_colors.get("R2", "green"),
                            line=dict(width=1, color='black')
                        ),
                        line=dict(color=defined_colors.get("R2", "green"), dash='dash', width=2),
                        name="R2 Trend",
                        hovertemplate="<b>%{x}</b><br>R2: %{y:.6f}<extra></extra>"
                    ))

            fig.update_layout(
                barmode='group',
                title="Leaderboard Metrics Comparison",
                xaxis_title="Model",
                yaxis_title="Metric Value",
                template="plotly_white",
                height=500,
                legend=dict(title="Metrics"),
                hovermode="x unified"
            )

            # FIX: Added unique key here to solve DuplicateElementId error
            st.plotly_chart(fig, use_container_width=True, key="leaderboard_chart")

        # --- Tab 3: Parameters  ---
        with lb_tabs[2]:
            param_list = []
            
            for model_name_full, res_data in results.items():
                model_obj = res_data['model']
                if isinstance(model_obj, Pipeline):
                    if 'model' in model_obj.named_steps:
                        final_model = model_obj.named_steps['model']
                    else:
                        final_model = model_obj
                else:
                    final_model = model_obj
                
                current_params = final_model.get_params()
                
                base_name = model_name_full.split(' (')[0].strip()
                
                unified_row = {"Model": model_name_full}
                
                if base_name in MODEL_DEFAULT_PARAMS:
                    target_keys = MODEL_DEFAULT_PARAMS[base_name].keys()
                    
                    for key in target_keys:
                        if key in current_params:
                            val = current_params[key]
                            
                            if val is None: val = "None"
                            if isinstance(val, (float)) and val == 0.0: val = 0 
                            
                            display_name = UNIFIED_PARAM_NAMES.get(key, key)
                            
                            unified_row[display_name] = val
                
                param_list.append(unified_row)

            if param_list:
                df_params = pd.DataFrame(param_list)
                
                logical_order = [
                    "Model", 
                    "Estimators / Iter", 
                    "Learning Rate", 
                    "Max Depth", 
                    "Subsample Ratio", 
                    "Col Sample / Max Feat",
                    "L2 Regularization",
                    "Min Samples Leaf"
                ]
                
                existing_cols = df_params.columns.tolist()
                final_cols = [c for c in logical_order if c in existing_cols] + \
                             [c for c in existing_cols if c not in logical_order]
                
                df_params = df_params[final_cols]
                
                df_params = df_params.fillna("-")
                
                st.markdown("###### 🛠️ Model Hyperparameters Comparison")
                st.dataframe(df_params, use_container_width=True)
            else:
                st.info("No parameter data available.")

    # Model Tabs
    with st.expander("### 📊 Visualization"):
        model_tabs = st.tabs(list(results.keys()))
        for i, model_name in enumerate(results.keys()):
            res = results[model_name]
            with model_tabs[i]:
                # Forecast Tab
                forecast_tab, diag_tab, xai_tab = st.tabs(["📉 Forecast", "🔍 Diagnostics", "🧠 XAI"])
                # Forecast
                with forecast_tab:
                    limit = min(200, len(res['yte']))
                    
                    # Alt tablar
                    forecast_sub_tabs = st.tabs(["📊 Single Model", "🔀 Comparison"])
                    
                    # 1️⃣ Single Model
                    with forecast_sub_tabs[0]:
                        df_viz = pd.DataFrame({'Actual': res['yte'].iloc[:limit], 'Predicted': res['ypr'][:limit]})
                        fig = px.line(df_viz, markers=True, title=f"{model_name} Forecast (First {limit} samples)")
                        st.plotly_chart(fig, use_container_width=True, key=f"{model_name}_single_forecast")
                    
                    # 2️⃣ Comparison
                    with forecast_sub_tabs[1]:
                        fig_comb = go.Figure()
                        fig_comb.add_trace(go.Scatter(y=res['yte'].iloc[:limit], name="Actual", line=dict(color='black', width=3)))
                        for n, r in results.items():
                            fig_comb.add_trace(go.Scatter(y=r['ypr'][:limit], name=n, line=dict(dash='dot')))
                        fig_comb.update_layout(title="Combined Model Forecast Comparison", height=500)
                        st.plotly_chart(fig_comb, use_container_width=True, key=f"{model_name}_combined_forecast")


                # Diagnostics
                with diag_tab:
                    display_diagnostic_plots(
                        res['yte'], res['ypr'],
                        res.get('ytr'), res.get('ytr_pr'),
                        key_prefix=model_name,
                        active_plots=selected_diag_plots
                    )

                # XAI
                with xai_tab:
                    run_xai_analysis(
                        model=res['model'], X_train=res['Xt'], X_test=res['Xte'], y_test=res['yte'],
                        methods=xai_ops, key_suffix=model_name
                    )

    # Export Section
    with st.expander("### 📥 Exports"):
        c1, c2 = st.columns(2)
        with c1:
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("Download Metrics (CSV)", csv, "metrics.csv", "text/csv")
        with c2:
            if st.button("Generate PDF Report"):
                with st.spinner("Rendering PDF..."):
                    try:
                        pdf_data = generate_pdf_report(results)
                        st.download_button("Download PDF Report", pdf_data, "report.pdf", "application/pdf")
                    except Exception as e:
                        st.error(f"PDF Error: {e}")


# -------------------------------------------------------------------------
# 6. PDF REPORT GENERATOR
# -------------------------------------------------------------------------

def generate_pdf_report(results):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # Cover Page
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.6, "Automated Model Report", ha='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.5, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ha='center', fontsize=14)
        pdf.savefig(fig)
        plt.close()

        # Leaderboard
        dfm = pd.DataFrame([{**{'Model':k}, **v['metrics']} for k,v in results.items()])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=dfm.round(4).values, colLabels=dfm.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax.set_title("Model Leaderboard", fontweight='bold')
        pdf.savefig(fig)
        plt.close()

        # Individual Model Plots (Top 3)
        for i, (name, res) in enumerate(results.items()):
            if i >= 3: break # Limit PDF size
            
            fig, axs = plt.subplots(2, 1, figsize=(8.5, 11))
            
            # Forecast
            limit = min(100, len(res['yte']))
            axs[0].plot(res['yte'].values[:limit], label='Actual')
            axs[0].plot(res['ypr'][:limit], label='Predicted', linestyle='--')
            axs[0].set_title(f"Forecast: {name}")
            axs[0].legend()
            
            # Residuals
            resid = res['yte'] - res['ypr']
            axs[1].hist(resid, bins=30, color='skyblue', edgecolor='black')
            axs[1].set_title(f"Residual Distribution: {name}")
            
            pdf.savefig(fig)
            plt.close()
            
    buf.seek(0)
    return buf.read()

# -------------------------------------------------------------------------
# 7. MAIN APP UI
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# SIDEBAR: MODERN UI & CONFIGURATION
# -------------------------------------------------------------------------

# 1. Custom CSS for better button and layout
st.sidebar.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B; 
        color: white; 
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        border-color: #FF2B2B;
        color: white;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.05rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- SECTION 1: DATA PIPELINE ---
st.sidebar.header("1. Veri ve Hazırlık")

with st.sidebar.expander("📂 Veri Yükleme & Sütunlar", expanded=True):
    uploaded_file = st.file_uploader("CSV Dosyası Yükle", type=['csv'], help="Analiz edilecek ham veriyi buraya sürükleyin.")
    
    # Initialize variables to prevent NameError if no file is uploaded
    df_raw = None
    feature_cols, target_col, date_col = [], None, None
    outlier_ops, scaler_ops = [], []
    
    if uploaded_file:
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            st.success(f"✅ Yüklendi: {df_raw.shape[0]} satır, {df_raw.shape[1]} sütun")
            cols = df_raw.columns.tolist()
            
            date_col = st.selectbox("📅 Tarih Sütunu (Opsiyonel)", [None] + cols, help="Zaman serisi grafikleri için gereklidir.")
            target_col = st.selectbox("🎯 Hedef Sütun (Target)", cols, index=len(cols)-1, help="Modelin tahmin etmeye çalışacağı değer.")
            
            default_feats = [c for c in cols if c != target_col][:5]
            feature_cols = st.multiselect("Lütfen Özellikleri Seçin", [c for c in cols if c != target_col], default=default_feats)
            
            if not feature_cols:
                st.error("⚠️ En az bir özellik seçmelisiniz.")
        else:
            st.error("Dosya okunamadı.")

if df_raw is not None and feature_cols:
    # Process Data
    df = preprocess_dataframe(df_raw, date_col)
    st.sidebar.header("2. Ön İşleme")
    with st.sidebar.expander("⚙️ Ön İşleme (Preprocessing)", expanded=False):
        st.markdown("**Veri Temizliği**")
        outlier_ops = st.multiselect(
            "Aykırı Değer Yönetimi", 
            ["IQR Capping", "Z-Score Capping", "Isolation Forest Drop"],
            help="Eğitim verisindeki uç değerleri baskılar veya siler."
        )
        
        st.markdown("**Ölçeklendirme**")
        scaler_ops = st.multiselect(
            "Feature Scaling", 
            ["Min-Max Scaling (0-1)", "Standard Scaling (Z-Score)", "Log Transformation (np.log1p)"],
            help="Verileri belirli bir aralığa sıkıştırarak model performansını artırır."
        )

    # --- SECTION 2: MODELING ---
    st.sidebar.header("3. Modelleme")
    
    with st.sidebar.expander("🧠 Model ve Parametreler", expanded=False):
        avail_models = ["HistGradientBoosting", "RandomForest", "GradientBoosting"]
        if XGBRegressor: avail_models.append("XGBoost")
        if LGBMRegressor: avail_models.append("LightGBM")
        if CatBoostRegressor: avail_models.append("CatBoost")
        
        selected_models = st.multiselect("Modelleri Seçin", avail_models, default=["HistGradientBoosting"])
        
        st.caption("Gelişmiş Ayarlar")
        hpo_ops = st.multiselect("Hiperparametre Optimizasyonu", ["Random Search"], help="En iyi parametreleri otomatik bulur (Süreyi uzatır).")
        metric_ops = st.multiselect("Metrikler", ["MSE", "RMSE", "MAE", "R2", "MAPE"], default=["MSE","RMSE", "R2"])

    # --- SECTION 3: ANALYSIS ---
    st.sidebar.header("4. Analiz Çıktıları")
    
    with st.sidebar.expander("📊 Grafikler ve XAI", expanded=False):
        available_plots = ["Adv. Scatter", "Residuals", "Distribution", "QQ Plot", "Influence", "Anomalies", "Overfitting Check"]
        selected_diag_plots = st.multiselect("Tanısal Grafikler", available_plots, default=["Adv. Scatter", "Overfitting Check"])
        
        xai_ops = st.multiselect("Açıklanabilirlik (XAI)", ["SHAP", "PFI", "LIME"], help="Modelin 'neden' bu kararı verdiğini açıklar.")

    # --- ACTION BUTTON ---
    train_btn = st.sidebar.button("🚀 Analizi Başlat", type="primary")

else:
    # Veri yüklenmediyse kullanıcıya bilgi ver ve butonu pasif/gizli tut
    if not uploaded_file:
        st.info("👈 Başlamak için lütfen sol menüden bir CSV dosyası yükleyin.")
    train_btn = False # Logic flow'u bozmamak için

# -------------------------------------------------------------------------
# EXECUTION LOGIC
# -------------------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state["results"] = None

if train_btn and df_raw is not None:
    st.session_state["results"] = {} # Reset
    
    # Progress UI
    col_prog1, col_prog2 = st.columns([3, 1])
    with col_prog1:
        prog_bar = st.progress(0)
    with col_prog2:
        status_text = st.empty()
    
    # Prepare Data
    X = df[feature_cols]
    y = df[target_col]
    
    # Define Scenarios
    scenarios = [("Raw", None, None)]
    if outlier_ops or scaler_ops:
        scenarios.append(("Processed", outlier_ops, scaler_ops))
    
    # Calculate Total Tasks
    total_tasks = len(selected_models) * len(scenarios) * (1 + len(hpo_ops))
    current_task = 0
    temp_results = {}
    
    start_time = time.time()

    for model_name in selected_models:
        for scen_label, scen_outliers, scen_scalers in scenarios:
            
            # --- Default Run ---
            display_name = f"{model_name} ({scen_label})"
            status_text.markdown(f"**Training:** `{display_name}`...")
            
            try:
                metrics, model, Xt, Xte, ytr, yte, ytr_pr, ypr = train_and_evaluate(
                    X, y, 0.2, model_name, 
                    outlier_methods=scen_outliers,
                    scaling_methods=scen_scalers,
                    active_metrics=metric_ops
                )
                
                temp_results[display_name] = {
                    'metrics': metrics, 'model': model, 
                    'Xt': Xt, 'Xte': Xte, 'yte': yte, 'ypr': ypr, 'ytr': ytr, 'ytr_pr': ytr_pr
                }
            except Exception as e:
                st.error(f"Error in {display_name}: {e}")
                
            current_task += 1
            prog_bar.progress(min(current_task / total_tasks, 1.0))
            
            # --- HPO Runs ---
            for hpo in hpo_ops:
                hpo_display_name = f"{model_name} ({hpo} - {scen_label})"
                status_text.markdown(f"**Tuning:** `{hpo_display_name}`...")
                
                try:
                    metrics_h, model_h, Xt_h, Xte_h, ytr_h, yte_h, ytr_pr_h, ypr_h = train_and_evaluate(
                        X, y, 0.2, model_name, hpo_method=hpo,
                        outlier_methods=scen_outliers,
                        scaling_methods=scen_scalers,
                        active_metrics=metric_ops
                    )
                    temp_results[hpo_display_name] = {
                        'metrics': metrics_h, 'model': model_h, 
                        'Xt': Xt_h, 'Xte': Xte_h, 'yte': yte_h, 'ypr': ypr_h, 'ytr': ytr_h, 'ytr_pr': ytr_h
                    }
                except Exception as e:
                    st.error(f"Error in HPO {hpo_display_name}: {e}")
                
                current_task += 1
                prog_bar.progress(min(current_task / total_tasks, 1.0))
            
    st.session_state["results"] = temp_results
    
    elapsed = time.time() - start_time
    status_text.success(f"Tamamlandı! ({elapsed:.1f}s)")
    time.sleep(1)
    status_text.empty()
    st.rerun()

# --- DISPLAY LOGIC ---
if st.session_state["results"]:
    results = st.session_state["results"]
    run_dashboard(results, selected_diag_plots, xai_ops)

