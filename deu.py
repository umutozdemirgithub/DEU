import streamlit as st
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# --- SKLEARN MODEL SELECTION ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit

# --- SKLEARN MODELS (ENSEMBLE) ---
from sklearn.ensemble import (
    HistGradientBoostingRegressor, 
    GradientBoostingRegressor, 
    RandomForestRegressor, 
    AdaBoostRegressor, 
    ExtraTreesRegressor,
    IsolationForest
)

# --- SKLEARN MODELS (LINEAR, BAYESIAN & ROBUST) ---
from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
    Lasso, 
    ElasticNet, 
    HuberRegressor,
    SGDRegressor, 
    BayesianRidge, 
    ARDRegression, 
    TheilSenRegressor, 
    RANSACRegressor,
    PassiveAggressiveRegressor
)

# --- SKLEARN MODELS (SVM) ---
from sklearn.svm import SVR, LinearSVR, NuSVR

# --- SKLEARN MODELS (OTHERS: TREE, NEIGHBORS, NN, GAUSSIAN) ---
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

# --- PREPROCESSING & METRICS ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score, max_error, mean_squared_log_error
)

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
# --- UTILS & STATS ---
from scipy import stats
import statsmodels.api as sm
from scipy.stats import gaussian_kde, probplot
import warnings
import time
from io import BytesIO
import plotly.io as pio
import math
from matplotlib.backends.backend_pdf import PdfPages
import plotly.figure_factory as ff
import random
from sklearn.base import clone

import io
import matplotlib.image as mpimg

import shap
import dice_ml
try:
    from alibi.explainers import AnchorTabular
except ImportError:
    AnchorTabular = None

import streamlit.components.v1 as components

import config

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from lime import lime_tabular
except ImportError:
    lime_tabular = None

try:
    import optuna
    from optuna.samplers import TPESampler,GPSampler
    from optuna.pruners import HyperbandPruner
except ImportError:
    optuna = None

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title=config.APP_CONFIG["page_title"], 
    layout=config.APP_CONFIG["page_layout"]
)

st.sidebar.markdown(
    f"""
    <style>
    /* Sidebar Butonları */
    .stButton>button {{
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .stButton>button:hover {{
        background-color: #FF2B2B;
        transform: scale(1.05);
        box-shadow: 0 6px 10px rgba(0,0,0,0.2);
    }}

    /* Expander Başlıkları */
    div[data-testid="stExpander"] div[role="button"] p {{
        font-weight: 600;
        font-size: 1rem;
        background: linear-gradient(90deg, #FF6B6B, #5F27CD, #FFD93D, #1DD1A1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: all 0.3s ease;
    }}
    div[data-testid="stExpander"] div[role="button"]:hover p {{
        transform: scale(1.05);
        text-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }}

    /* Sidebar Başlığı */
    .sidebar-title {{
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #5F27CD, #FFD93D, #1DD1A1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite;
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.image(config.APP_CONFIG["sidebar_icon"], width=400)
st.sidebar.markdown(f"<p class='sidebar-title'>{config.APP_CONFIG['sidebar_title']}</p>", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    @keyframes fadeInSlide {{
        0% {{ opacity: 0; transform: translateY(-30px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    @keyframes shine {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    @keyframes glowPulse {{
        0%, 100% {{ box-shadow: 0 0 10px rgba(255, 255, 255, 0.2); }}
        50% {{ box-shadow: 0 0 30px rgba(255, 255, 255, 0.5); }}
    }}

    @keyframes wave {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-5px); }}
    }}

    .fancy-header {{
        display: flex;
        flex-direction: column;
        align-items: center;
        animation: glowPulse 3s infinite ease-in-out, wave 4s infinite ease-in-out;
    }}

    .fancy-title {{
        position: relative;
        text-align: center; 
        background: linear-gradient(90deg, #FF6B6B, #5F27CD, #FFD93D, #1DD1A1);
        background-size: 300% 300%;
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        animation: fadeInSlide 1.2s ease-out forwards, gradientShift 8s ease infinite;
        transition: transform 0.3s ease, text-shadow 0.3s ease;
        cursor: default;
        overflow: hidden;
    }}

    .fancy-title::after {{
        content: '';
        position: absolute;
        top: 0; left: -75%;
        width: 50%;
        height: 100%;
        background: linear-gradient(
            120deg, 
            rgba(255,255,255,0) 0%, 
            rgba(255,255,255,0.5) 50%, 
            rgba(255,255,255,0) 100%
        );
        transform: skewX(-25deg);
        animation: shine 2s infinite linear;
    }}

    .fancy-title:hover {{
        transform: scale(1.05);
        text-shadow: 4px 4px 8px rgba(0,0,0,0.5);
        animation: fadeInSlide 1.2s ease-out forwards, gradientShift 3s ease infinite;
    }}

    .fancy-title:hover::after {{
        animation: shine 1s infinite linear;
    }}

    .fancy-hr {{
        border: none;
        height: 4px;
        background: linear-gradient(270deg, #FF6B6B, #5F27CD, #FFD93D, #1DD1A1);
        background-size: 400% 400%;
        border-radius: 5px;
        width: 60%;
        margin: 10px auto 40px auto;
        animation: gradientShift 6s ease infinite;
    }}

    .fancy-hr:hover {{
        animation: gradientShift 2s ease infinite;
    }}

    /* Responsive Tasarım */
    @media (max-width: 768px) {{
        .fancy-title {{
            font-size: 2.2em;
        }}
        .fancy-hr {{
            width: 80%;
        }}
    }}

    @media (max-width: 480px) {{
        .fancy-title {{
            font-size: 1.6em;
        }}
        .fancy-hr {{
            width: 90%;
        }}
    }}
    </style>

    <div class="fancy-header">
        <h1 class="fancy-title">{config.APP_CONFIG["page_title"]}</h1>
        <hr class="fancy-hr">
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# 📖 USER MANUAL
# =============================================================================
with st.expander("📖 Application User Manual and Workflow (Click to Expand)", expanded=False):
    st.markdown("""
    This panel is designed for you to manage end-to-end machine learning processes (AutoML), compare models,
    and interpret results using **Explainable AI (XAI)** tools.
    """)
    
    # Organize the guide into tabs
    guide_tab1, guide_tab2, guide_tab3, guide_tab4 = st.tabs([
        "📂 1. Data Loading", 
        "⚙️ 2. Preprocessing", 
        "🧠 3. Modeling", 
        "📊 4. Analysis & Outputs"
    ])

    # --- TAB 1: DATA ---
    with guide_tab1:
        st.info("Start by introducing your dataset to the system.")
        st.markdown("""
        1.  **File Upload:** Drag and drop your file in the **'Data & Preparation'** section on the left panel.
            * *Supported Formats:* `.csv`, `.xlsx` (Excel), `.mat` (Matlab).
        2.  **Column Selection:**
            * **Date Column:** (Optional) Select the date column if you are performing time series analysis.
            * **Target Column:** Select the target variable (Label/Output) you want to predict.
            * **Features:** Select the columns to be used as inputs for the model.
        3.  **Preview:** Check the data in the **'Preview'** area on the main screen after making your selections.
        """)

    # --- TAB 2: PREPROCESSING ---
    with guide_tab2:
        st.warning("Apply preprocessing steps to improve data quality.")
        st.markdown("""
        Use the **'Preprocessing'** menu on the left panel:
        * **Missing Value Imputation:** Method to fill missing data (Mean, Median, Mode, etc.).
        * **Outlier Handling:** Suppress (IQR/Z-Score) or remove (Isolation Forest) outliers.
        * **Scaling:** Scale data to a specific range (Min-Max, Standard Scaler, etc.).
        
        👉 **Important:** After selecting settings, click the **'Apply'** button to save changes. 
        You can use the **'Reset'** button to revert the data to its original state.
        """)

    # --- TAB 3: MODELING ---
    with guide_tab3:
        st.success("Determine which algorithms will compete.")
        st.markdown("""
        Define your strategy from the **'Modeling'** menu:
        1.  **Select Models:** Choose the algorithms you want to use (e.g., *XGBoost, RandomForest, CatBoost*).
        2.  **Hyperparameter Optimization (HPO):**
            * *Random Search:* Fast but random parameter scanning.
            * *Grid Search:* Comprehensive but slower scanning.
            * *(If no selection is made, models are trained with default settings).*
        3.  **Evaluation Metrics:** Select which success criteria (RMSE, R2, MAE, etc.) to calculate when comparing models.
        """)

    # --- TAB 4: ANALYSIS ---
    with guide_tab4:
        st.error("Visualize results and explain model decisions.")
        st.markdown("""
        After clicking the **🚀 Start Analysis** button, results are presented in 3 main sections:
        
        1.  **🏆 Leaderboard:** Compares performance metrics of all models via tables and charts. Allows identifying the best model (based on R2 or RMSE).
        2.  **📊 Visualization (For Each Model):**
            * *Forecast:* Actual vs. Predicted graph.
            * *Diagnostics:* Error analysis (Residuals, QQ Plot, Overfitting Check).
            * *XAI (Explainability):* Explains why the model made that decision (SHAP, LIME, Feature Importance).
        3.  **📥 Exports:** You can download results as Excel or generate a PDF report.
        """)

# -------------------------------------------------------------------------
# 1. CACHED UTILITIES 
# -------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """
    Reads the file uploaded by the user based on its extension defined in CONFIG.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    df = None
    
    if file_extension not in config.DATA_CONFIG["supported_extensions"]:
        st.error(f"Unsupported file type: .{file_extension}. Supported: {config.DATA_CONFIG['supported_extensions']}")
        return None

    try:
        if file_extension in ['csv', 'txt']:
            uploaded_file.seek(0) 
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        
        elif file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file)
        
        elif file_extension == 'mat':
            mat_contents = loadmat(uploaded_file)
            for key in mat_contents:
                if isinstance(mat_contents[key], np.ndarray) and mat_contents[key].ndim >= 2:
                    df = pd.DataFrame(mat_contents[key])
                    potential_col_key = key + "_colnames"
                    if potential_col_key in mat_contents and mat_contents[potential_col_key].shape[1] == df.shape[1]:
                        df.columns = [str(c[0][0]) for c in mat_contents[potential_col_key]]
                    else:
                        df.columns = [f"col_{i}" for i in range(df.shape[1])]
                    break
            if df is None:
                st.error("Could not find a suitable data matrix to process in the MAT file.")
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    if df is not None:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        original_cols = df.shape[1]
        df.dropna(axis=1, how='all', inplace=True)
        if df.shape[1] < original_cols:
             st.warning(f"Dropped {original_cols - df.shape[1]} columns that were entirely empty.")
        
        inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
        if inf_count > 0:
             st.warning(f"Replacing {inf_count} infinite values with NaN.")
             df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

@st.cache_data(show_spinner=False)
def preprocess_dataframe(df, date_col_name=None):
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
# 1.1 DATA PROCESSİNG
# -------------------------------------------------------------------------
# ==========================================
# 🔧 Missing Value Filling Function
# ==========================================
def impute_missing(df, method):
    df_copy = df.copy()

    if method == "Mean":
        for col in df_copy.columns:
            if df_copy[col].dtype != "object":
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)

    elif method == "Median":
        for col in df_copy.columns:
            if df_copy[col].dtype != "object":
                df_copy[col].fillna(df_copy[col].median(), inplace=True)

    elif method == "Mode":
        for col in df_copy.columns:
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)

    elif method == "Zero":
        df_copy.fillna(0, inplace=True)

    return df_copy

# ==========================================
# 🔧 OUTLIER HANDLING FUNCTION
# ==========================================
def apply_outlier_handling(X_train, y_train, methods):
    if not methods:
        return X_train, y_train

    train_df = pd.concat([X_train, y_train], axis=1)
    target_col = y_train.name
    feat_cols = X_train.columns.tolist()
    num_cols = train_df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()

    df_clean = train_df.copy()

    # --- IQR Capping ---
    if "IQR Capping" in methods and num_cols:
        for col in num_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            df_clean[col] = df_clean[col].clip(
                lower=Q1 - 1.5 * IQR,
                upper=Q3 + 1.5 * IQR
            )

    # --- Z-Score Capping ---
    if "Z-Score Capping" in methods and num_cols:
        for col in num_cols:
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            df_clean[col] = df_clean[col].clip(
                lower=mean_val - 3 * std_val,
                upper=mean_val + 3 * std_val
            )

    # --- Isolation Forest Drop ---
    if "Isolation Forest Drop" in methods and num_cols:
        try:
            iso = IsolationForest(
                contamination=0.05,
                random_state=42,
                n_jobs=1
            )
            yhat = iso.fit_predict(df_clean[num_cols])
            df_clean = df_clean[yhat != -1]
        except:
            pass

    # Return cleaned data
    X_train_clean = df_clean[feat_cols]
    y_train_clean = df_clean[target_col]

    return X_train_clean, y_train_clean

# ==========================================
# 🔧 SCALING FUNCTION (choosing scaler)
# ==========================================
def get_scaler(scaling_methods):
    if not scaling_methods:
        return None

    if "Min-Max Scaling (0-1)" in scaling_methods:
        return MinMaxScaler
    elif "Standard Scaling (Z-Score)" in scaling_methods:
        return StandardScaler
    elif "Robust Scaling (IQR based)" in scaling_methods:
        return RobustScaler
    elif "MaxAbs Scaling (-1 to 1)" in scaling_methods:
        return MaxAbsScaler
    return None
# ==========================================
# 🔧 LOG TRANSFORM FUNCTION
# ==========================================
def apply_log_transform(X_train, X_test, scaling_methods):
    if scaling_methods and "Log Transformation (np.log1p)" in scaling_methods:
        # Only safe if all values non-negative
        if (X_train.values >= 0).all():
            return np.log1p(X_train), np.log1p(X_test)
    return X_train, X_test

# -------------------------------------------------------------------------
# 2. MODEL FACTORY & CONFIGURATION
# -------------------------------------------------------------------------
def safe_model_factory(name, random_state=config.DATA_CONFIG['random_state']):
    defaults = config.MODEL_DEFAULT_PARAMS.get(name, {}).copy()
    
    if 'random_state' in defaults:
        defaults.pop('random_state')
    
    # --- TREE & ENSEMBLE ---
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=random_state, **defaults)
    elif name == "RandomForest":
        return RandomForestRegressor(random_state=random_state, **defaults)
    elif name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=random_state, **defaults)
    elif name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=random_state, **defaults)
    elif name == "AdaBoost":
        return AdaBoostRegressor(random_state=random_state, **defaults)
    
    elif name == "XGBoost":
        if XGBRegressor is None: return None  
        return XGBRegressor(random_state=random_state, **defaults)
    elif name == "LightGBM":
        if LGBMRegressor is None: return None
        return LGBMRegressor(random_state=random_state, **defaults)
    elif name == "CatBoost":
        if CatBoostRegressor is None: return None
        return CatBoostRegressor(random_state=random_state, **defaults)
    
    # --- LINEAR & REGULARIZED ---
    elif name == "LinearRegression":
        return LinearRegression(**defaults)
    elif name == "Ridge":
        return Ridge(random_state=random_state, **defaults)
    elif name == "Lasso":
        return Lasso(random_state=random_state, **defaults)
    elif name == "ElasticNet":
        return ElasticNet(random_state=random_state, **defaults)
    elif name == "SGDRegressor":
        return SGDRegressor(random_state=random_state, **defaults)
    elif name == "HuberRegressor":
        return HuberRegressor(**defaults) 
        
    # --- BAYESIAN & ROBUST ---
    elif name == "BayesianRidge":
        return BayesianRidge(**defaults)
    elif name == "ARDRegression":
        return ARDRegression(**defaults)
    elif name == "TheilSenRegressor":
        return TheilSenRegressor(random_state=random_state, **defaults)
    elif name == "RANSACRegressor":
        return RANSACRegressor(random_state=random_state, **defaults)

    # --- SVM ---
    elif name == "SVR":
        return SVR(**defaults)
    elif name == "LinearSVR":
        return LinearSVR(random_state=random_state, **defaults)
    elif name == "NuSVR":
        return NuSVR(**defaults)

    # --- OTHERS ---
    elif name == "GaussianProcessRegressor":
        return GaussianProcessRegressor(random_state=random_state, **defaults)
    elif name == "DecisionTree":
        return DecisionTreeRegressor(random_state=random_state, **defaults)
    elif name == "KNeighborsRegressor":
        return KNeighborsRegressor(**defaults)
    elif name == "MLPRegressor":
        return MLPRegressor(random_state=random_state, **defaults)
    
    else:
        st.warning(f"Model '{name}' is not defined in the factory. Using default (HistGradient).")
        return HistGradientBoostingRegressor(random_state=random_state)

def get_optuna_params(trial, model_name):
    """
        Dynamic, model-specific parameter search space for Optuna.
        The logic of the config file has been preserved.
    """
    params = {}
    
    if model_name == "HistGradientBoosting":
        params["model__learning_rate"] = trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True)
        params["model__max_iter"] = trial.suggest_int("model__max_iter", 100, 500)
        params["model__max_depth"] = trial.suggest_categorical("model__max_depth", [None, 3, 5, 10])
        params["model__l2_regularization"] = trial.suggest_float("model__l2_regularization", 1e-6, 1.0, log=True)

    elif model_name == "RandomForest":
        params["model__n_estimators"] = trial.suggest_int("model__n_estimators", 50, 400)
        params["model__max_depth"] = trial.suggest_categorical("model__max_depth", [None, 10, 20, 30])
        params["model__min_samples_split"] = trial.suggest_int("model__min_samples_split", 2, 15)
        
    elif model_name == "XGBoost":
        params["model__n_estimators"] = trial.suggest_int("model__n_estimators", 100, 500)
        params["model__learning_rate"] = trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True)
        params["model__max_depth"] = trial.suggest_int("model__max_depth", 3, 10)
        params["model__subsample"] = trial.suggest_float("model__subsample", 0.5, 1.0)
        params["model__colsample_bytree"] = trial.suggest_float("model__colsample_bytree", 0.5, 1.0)
        
    elif model_name == "LightGBM":
        params["model__n_estimators"] = trial.suggest_int("model__n_estimators", 100, 500)
        params["model__learning_rate"] = trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True)
        params["model__num_leaves"] = trial.suggest_int("model__num_leaves", 20, 100)
        
    elif model_name == "CatBoost":
        params["model__iterations"] = trial.suggest_int("model__iterations", 100, 500)
        params["model__learning_rate"] = trial.suggest_float("model__learning_rate", 0.01, 0.3, log=True)
        params["model__depth"] = trial.suggest_int("model__depth", 4, 10)
        params["model__l2_leaf_reg"] = trial.suggest_float("model__l2_leaf_reg", 1, 10)

    elif model_name == "SVR":
        params["model__C"] = trial.suggest_float("model__C", 0.1, 100, log=True)
        params["model__epsilon"] = trial.suggest_float("model__epsilon", 0.01, 1.0, log=True)
        
    elif model_name == "Ridge":
        params["model__alpha"] = trial.suggest_float("model__alpha", 0.1, 100.0, log=True)
        
    elif model_name == "Lasso":
        params["model__alpha"] = trial.suggest_float("model__alpha", 0.001, 10.0, log=True)
        
    elif model_name == "ElasticNet":
        params["model__alpha"] = trial.suggest_float("model__alpha", 0.001, 10.0, log=True)
        params["model__l1_ratio"] = trial.suggest_float("model__l1_ratio", 0.1, 0.9)

    elif model_name == "DecisionTree":
        params["model__max_depth"] = trial.suggest_categorical("model__max_depth", [None, 5, 10, 20])
        params["model__min_samples_split"] = trial.suggest_int("model__min_samples_split", 2, 20)

    elif model_name == "MLPRegressor":
        params["model__alpha"] = trial.suggest_float("model__alpha", 1e-5, 1e-1, log=True)
        params["model__learning_rate_init"] = trial.suggest_float("model__learning_rate_init", 1e-4, 1e-1, log=True)
        
    return params

class ABCHyperparameterOptimizer:
    """
    Artificial Bee Colony (ABC) based hyperparameter optimization class 
    for Scikit-learn models.
    (Final Reinforced Version: Error-protected and Guaranteed Fit feature)
    """
    def __init__(self, estimator, param_distributions, cv, scoring="neg_mean_squared_error", 
                 n_population=10, max_iter=10):
        self.estimator = estimator
        self.base_estimator_ = clone(estimator) 
        self.param_dist = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.n_pop = n_population
        self.max_iter = max_iter
        self.limit = max_iter // 2
        
    def _get_random_params(self):
        """Draws a random set of parameters from Scipy distributions."""
        params = {}
        for k, v in self.param_dist.items():
            if hasattr(v, "rvs"):
                val = v.rvs()
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    val = int(val)
                params[k] = val
            elif isinstance(v, list):
                params[k] = random.choice(v)
            else:
                params[k] = v
        return params

    def _mutate_params(self, current_params, neighbor_params):
        new_params = current_params.copy()
        for k, v in current_params.items():
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                phi = random.uniform(-1, 1)
                neighbor_val = neighbor_params[k]
                
                if neighbor_val is None:
                    new_params[k] = v
                    continue

                new_val = v + phi * (v - neighbor_val)
                
                if isinstance(v, (int, np.integer)):
                    new_val = int(round(abs(new_val)))
                    if new_val < 1: new_val = 1
                else:
                    new_val = abs(new_val) 
                
                new_params[k] = new_val
            
            else:
                if random.random() < 0.5:
                    new_params[k] = neighbor_params[k]
                else:
                    if hasattr(self.param_dist[k], "rvs"):
                        val = self.param_dist[k].rvs()
                        if isinstance(val, (np.integer, np.int64)): val = int(val)
                        new_params[k] = val
                    elif isinstance(self.param_dist[k], list):
                        new_params[k] = random.choice(self.param_dist[k])
                        
        return new_params

    def fit(self, X, y):
        population = []
        
        for _ in range(self.n_pop):
            params = self._get_random_params()
            score = self._evaluate(params, X, y)
            population.append({'params': params, 'score': score, 'trial': 0})
            
        self.best_solution_ = max(population, key=lambda x: x['score'])
        
        for i in range(self.max_iter):
            for j in range(self.n_pop):
                current = population[j]
                idxs = list(range(self.n_pop))
                idxs.remove(j)
                neighbor = population[random.choice(idxs)]
                
                new_params = self._mutate_params(current['params'], neighbor['params'])
                new_score = self._evaluate(new_params, X, y)

                if new_score > current['score']:
                    population[j] = {'params': new_params, 'score': new_score, 'trial': 0}
                else:
                    population[j]['trial'] += 1

            scores = [p['score'] for p in population]
            valid_scores = [s for s in scores if s != -float('inf')]
            if not valid_scores:
                min_s, max_s = -1.0, 0.0
            else:
                min_s, max_s = min(valid_scores), max(valid_scores)
            
            probs = []
            if max_s == min_s: 
                 probs = [1.0/self.n_pop] * self.n_pop
            else:
                for s in scores:
                    if s == -float('inf'): probs.append(0.0)
                    else: probs.append((s - min_s) / (max_s - min_s + 1e-9))

            if sum(probs) == 0: probs = [1.0/self.n_pop] * self.n_pop

            for _ in range(self.n_pop):
                try:
                    j = random.choices(range(self.n_pop), weights=probs, k=1)[0]
                except:
                    j = random.choice(range(self.n_pop))

                current = population[j]
                idxs = list(range(self.n_pop))
                idxs.remove(j)
                neighbor = population[random.choice(idxs)]
                
                new_params = self._mutate_params(current['params'], neighbor['params'])
                new_score = self._evaluate(new_params, X, y)
                    
                if new_score > current['score']:
                    population[j] = {'params': new_params, 'score': new_score, 'trial': 0}
                else:
                    population[j]['trial'] += 1

            for j in range(self.n_pop):
                if population[j]['trial'] > self.limit:
                    params = self._get_random_params()
                    score = self._evaluate(params, X, y)
                    population[j] = {'params': params, 'score': score, 'trial': 0}

            current_best = max(population, key=lambda x: x['score'])
            if current_best['score'] > self.best_solution_['score']:
                self.best_solution_ = current_best

        try:
            if self.best_solution_['score'] != -float('inf'):
                self.estimator.set_params(**self.best_solution_['params'])
                self.best_estimator_ = self.estimator
                self.best_estimator_.fit(X, y)
            else:
                st.warning("⚠️ ABC Optimization could not find valid parameters. Using default model.")
                self.best_estimator_ = clone(self.base_estimator_)
                self.best_estimator_.fit(X, y)
                
        except Exception as e:
            st.warning(f"⚠️ ABC could not be trained with best parameters ({str(e)}). Reverting to default settings.")
            try:
                self.best_estimator_ = clone(self.base_estimator_)
                self.best_estimator_.fit(X, y)
            except Exception as e2:
                st.error(f"❌ Critical Error: Model could not be trained even with default settings: {e2}")
                self.best_estimator_ = self.estimator 

        return self

    def _evaluate(self, params, X, y):
        try:
            est = clone(self.estimator)
            est.set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
            return scores.mean()
        except Exception:
            return -float('inf')

    def predict(self, X):
        return self.best_estimator_.predict(X)

class GeneticHyperparameterOptimizer:
    """
    Scikit-learn modelleri için Genetik Algoritma (GA) tabanlı 
    hiperparametre optimizasyon sınıfı.
    (Güçlendirilmiş Versiyon: Hata korumalı, Elitizm ve Garanti Fit özellikli)
    """
    def __init__(self, estimator, param_distributions, cv, scoring="neg_mean_squared_error", 
                 n_population=20, max_iter=10, mutation_rate=0.1, crossover_rate=0.8):
        self.estimator = estimator
        self.base_estimator_ = clone(estimator)
        self.param_dist = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.n_pop = n_population
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def _get_random_params(self):
        """Rastgele bir birey (parametre seti) oluşturur."""
        params = {}
        for k, v in self.param_dist.items():
            if hasattr(v, "rvs"):
                val = v.rvs()
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    val = int(val)
                params[k] = val
            elif isinstance(v, list):
                params[k] = random.choice(v)
            else:
                params[k] = v
        return params

    def _crossover(self, parent1, parent2):
        """İki ebeveynden yeni bir çocuk (parametre seti) üretir."""
        child = parent1.copy()
        for k in parent1.keys():
            if random.random() < 0.5:
                child[k] = parent2[k]
        return child

    def _mutate(self, params):
        """Bireyin genlerinde rastgele değişiklik yapar."""
        mutated = params.copy()
        for k, v in params.items():
            if random.random() < self.mutation_rate:
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    change = v * 0.2 * random.uniform(-1, 1)
                    if change == 0: change = random.uniform(-0.1, 0.1) 
                    new_val = v + change
                    
                    if isinstance(v, (int, np.integer)):
                        new_val = int(round(abs(new_val)))
                        if new_val < 1: new_val = 1
                    else:
                        new_val = abs(new_val)
                    mutated[k] = new_val
                
                else:
                    if hasattr(self.param_dist[k], "rvs"):
                        val = self.param_dist[k].rvs()
                        if isinstance(val, (np.integer, np.int64)): val = int(val)
                        mutated[k] = val
                    elif isinstance(self.param_dist[k], list):
                        mutated[k] = random.choice(self.param_dist[k])
        return mutated

    def fit(self, X, y):
        population = []
        for _ in range(self.n_pop):
            params = self._get_random_params()
            score = self._evaluate(params, X, y)
            population.append({'params': params, 'score': score})

        self.best_solution_ = max(population, key=lambda x: x['score'])

        for generation in range(self.max_iter):
            population.sort(key=lambda x: x['score'], reverse=True)
            
            elite_count = int(self.n_pop * 0.2)
            new_population = population[:elite_count]
            
            while len(new_population) < self.n_pop:
                candidates = random.sample(population, 3)
                parent1 = max(candidates, key=lambda x: x['score'])['params']
                candidates = random.sample(population, 3)
                parent2 = max(candidates, key=lambda x: x['score'])['params']
                
                if random.random() < self.crossover_rate:
                    child_params = self._crossover(parent1, parent2)
                else:
                    child_params = parent1
                
                child_params = self._mutate(child_params)
                
                score = self._evaluate(child_params, X, y)
                new_population.append({'params': child_params, 'score': score})
            
            population = new_population
            
            current_best = max(population, key=lambda x: x['score'])
            if current_best['score'] > self.best_solution_['score']:
                self.best_solution_ = current_best

        try:
            if self.best_solution_['score'] != -float('inf'):
                self.estimator.set_params(**self.best_solution_['params'])
                self.best_estimator_ = self.estimator
                self.best_estimator_.fit(X, y)
            else:
                st.warning("⚠️ GA geçerli parametre bulamadı. Varsayılan model kullanılıyor.")
                self.best_estimator_ = clone(self.base_estimator_)
                self.best_estimator_.fit(X, y)
        except Exception as e:
            st.warning(f"⚠️ GA en iyi parametrelerle eğitilemedi. Varsayılana dönülüyor. Hata: {e}")
            self.best_estimator_ = self.estimator 
            
        return self

    def _evaluate(self, params, X, y):
        try:
            est = clone(self.estimator)
            est.set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
            return scores.mean()
        except:
            return -float('inf')

    def predict(self, X):
        return self.best_estimator_.predict(X)

class PSOHyperparameterOptimizer:
    """
    Parçacık Sürü Optimizasyonu (PSO) tabanlı hiperparametre optimizasyon sınıfı.
    (Güçlendirilmiş Versiyon: Hata korumalı)
    """
    def __init__(self, estimator, param_distributions, cv, scoring="neg_mean_squared_error", 
                 n_particles=10, max_iter=10, c1=1.5, c2=1.5, w=0.7):
        self.estimator = estimator
        self.base_estimator_ = clone(estimator)
        self.param_dist = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.c1 = c1 
        self.c2 = c2 
        self.w = w  

    def _get_random_params(self):
        """Rastgele bir parçacık konumu oluşturur."""
        params = {}
        for k, v in self.param_dist.items():
            if hasattr(v, "rvs"):
                val = v.rvs()
                if isinstance(val, (np.integer, np.int64, np.int32)): val = int(val)
                params[k] = val
            elif isinstance(v, list):
                params[k] = random.choice(v)
            else:
                params[k] = v
        return params

    def _update_particle(self, current_params, p_best_params, g_best_params):
        """
        PSO Hız ve Konum Güncelleme Mantığı:
        Sayısal değerler için standart PSO formülü uygulanır.
        Kategorik değerler için 'olasılıksal' geçiş yapılır.
        """
        new_params = current_params.copy()
        
        for k, v in current_params.items():
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                p_best_val = p_best_params[k]
                g_best_val = g_best_params[k]
                
                r1, r2 = random.random(), random.random()
                
                cognitive_velocity = self.c1 * r1 * (p_best_val - v)
                social_velocity = self.c2 * r2 * (g_best_val - v)
                inertia = self.w * (random.uniform(-1, 1) * v * 0.1) 
                
                new_val = v + cognitive_velocity + social_velocity + inertia
                
                if isinstance(v, (int, np.integer)):
                    new_val = int(round(abs(new_val)))
                    if new_val < 1: new_val = 1
                else:
                    new_val = abs(new_val)
                
                new_params[k] = new_val

            else:
                roll = random.random()
                if roll < 0.3:
                    new_params[k] = g_best_params[k]
                elif roll < 0.6:
                    new_params[k] = p_best_params[k]
                else:
                    new_params[k] = v 
                    
        return new_params

    def fit(self, X, y):
        particles = []
        
        for _ in range(self.n_particles):
            params = self._get_random_params()
            score = self._evaluate(params, X, y)
            particles.append({
                'current_params': params,
                'current_score': score,
                'p_best_params': params,
                'p_best_score': score
            })
            
        g_best = max(particles, key=lambda x: x['p_best_score'])
        g_best_params = g_best['p_best_params']
        g_best_score = g_best['p_best_score']

        for i in range(self.max_iter):
            for p in particles:
                new_params = self._update_particle(p['current_params'], p['p_best_params'], g_best_params)
                new_score = self._evaluate(new_params, X, y)
                
                p['current_params'] = new_params
                p['current_score'] = new_score
                
                if new_score > p['p_best_score']:
                    p['p_best_params'] = new_params
                    p['p_best_score'] = new_score
                    
                    if new_score > g_best_score:
                        g_best_score = new_score
                        g_best_params = new_params

        self.best_solution_ = {'params': g_best_params, 'score': g_best_score}

        try:
            if self.best_solution_['score'] != -float('inf'):
                self.estimator.set_params(**self.best_solution_['params'])
                self.best_estimator_ = self.estimator
                self.best_estimator_.fit(X, y)
            else:
                st.warning("⚠️ PSO geçerli parametre bulamadı. Varsayılan kullanılıyor.")
                self.best_estimator_ = clone(self.base_estimator_)
                self.best_estimator_.fit(X, y)
        except Exception as e:
            st.warning(f"⚠️ PSO hatası: {e}. Varsayılan modele dönülüyor.")
            self.best_estimator_ = self.estimator
            
        return self

    def _evaluate(self, params, X, y):
        try:
            est = clone(self.estimator)
            est.set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
            return scores.mean()
        except:
            return -float('inf')
    
    def predict(self, X):
        return self.best_estimator_.predict(X)

class SimulatedAnnealingOptimizer:
    """
    Benzetilmiş Tavlama (Simulated Annealing - SA) tabanlı 
    hiperparametre optimizasyon sınıfı.
    """
    def __init__(self, estimator, param_distributions, cv, scoring="neg_mean_squared_error", 
                 max_iter=50, initial_temp=10.0, cooling_rate=0.9):
        self.estimator = estimator
        self.base_estimator_ = clone(estimator)
        self.param_dist = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.max_iter = max_iter
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def _get_random_params(self):
        params = {}
        for k, v in self.param_dist.items():
            if hasattr(v, "rvs"):
                val = v.rvs()
                if isinstance(val, (np.integer, np.int64, np.int32)): val = int(val)
                params[k] = val
            elif isinstance(v, list):
                params[k] = random.choice(v)
            else:
                params[k] = v
        return params

    def _get_neighbor(self, params):
        """Mevcut çözümün yakınında rastgele bir komşu türetir (Mutasyon benzeri)."""
        neighbor = params.copy()
        keys = list(params.keys())
        k = random.choice(keys)
        v = params[k]
        
        if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
            change = v * 0.3 * random.uniform(-1, 1)
            if change == 0: change = random.uniform(-0.1, 0.1)
            new_val = v + change
            
            if isinstance(v, (int, np.integer)):
                new_val = int(round(abs(new_val)))
                if new_val < 1: new_val = 1
            else:
                new_val = abs(new_val)
            neighbor[k] = new_val
            
        else:
            if hasattr(self.param_dist[k], "rvs"):
                val = self.param_dist[k].rvs()
                if isinstance(val, (np.integer, np.int64)): val = int(val)
                neighbor[k] = val
            elif isinstance(self.param_dist[k], list):
                neighbor[k] = random.choice(self.param_dist[k])
                
        return neighbor

    def fit(self, X, y):
        current_params = self._get_random_params()
        current_score = self._evaluate(current_params, X, y)
        
        best_params = current_params
        best_score = current_score
        
        temperature = self.initial_temp
        
        for i in range(self.max_iter):
            neighbor_params = self._get_neighbor(current_params)
            neighbor_score = self._evaluate(neighbor_params, X, y)
            
            delta = neighbor_score - current_score
            
            if delta > 0:
                accept = True
            else:
                try:
                    prob = math.exp(delta / temperature)
                except OverflowError:
                    prob = 0
                accept = random.random() < prob
            
            if accept:
                current_params = neighbor_params
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_params = current_params
            
            temperature *= self.cooling_rate
            
        self.best_solution_ = {'params': best_params, 'score': best_score}

        try:
            if self.best_solution_['score'] != -float('inf'):
                self.estimator.set_params(**self.best_solution_['params'])
                self.best_estimator_ = self.estimator
                self.best_estimator_.fit(X, y)
            else:
                st.warning("⚠️ SA geçerli parametre bulamadı. Varsayılan kullanılıyor.")
                self.best_estimator_ = clone(self.base_estimator_)
                self.best_estimator_.fit(X, y)
        except Exception as e:
            st.warning(f"⚠️ SA hatası: {e}. Varsayılana dönülüyor.")
            self.best_estimator_ = self.estimator
            
        return self

    def _evaluate(self, params, X, y):
        try:
            est = clone(self.estimator)
            est.set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
            return scores.mean()
        except:
            return -float('inf')

    def predict(self, X):
        return self.best_estimator_.predict(X)

def perform_hpo(X_train, y_train, method, model_name, use_timesplit=False, scaler_cls=None):
    # 1) Base Model & Pipeline Setup
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

    # =========================================================
    # ARTIFICIAL BEE COLONY BLOĞU
    # =========================================================
    if method == "Artificial Bee Colony":
            model_spaces = config.HPO_SPACES.get(model_name, {})
            selected_space = model_spaces.get("random", {})
            
            if not selected_space:
                pipeline.fit(X_train, y_train)
                return pipeline

            st.toast(f"🐝 ABC Optimizing: {model_name}...", icon="🐝")
            
            abc_opt = ABCHyperparameterOptimizer(
                estimator=pipeline,
                param_distributions=selected_space,
                cv=cv_strategy,
                n_population=6,  
                max_iter=5       
            )
            
            abc_opt.fit(X_train, y_train)
            return abc_opt.best_estimator_

    elif method == "Genetic Algorithm":
            model_spaces = config.HPO_SPACES.get(model_name, {})
            selected_space = model_spaces.get("random", {}) 
            
            if not selected_space:
                pipeline.fit(X_train, y_train)
                return pipeline
                
            st.toast(f"🧬 GA Evolving: {model_name}...", icon="🧬")
            
            ga_opt = GeneticHyperparameterOptimizer(
                estimator=pipeline,
                param_distributions=selected_space,
                cv=cv_strategy,
                n_population=10, 
                max_iter=5
            )
            ga_opt.fit(X_train, y_train)
            return ga_opt.best_estimator_

    elif method == "Particle Swarm Optimization":
        model_spaces = config.HPO_SPACES.get(model_name, {})
        selected_space = model_spaces.get("random", {})
        
        if not selected_space:
            pipeline.fit(X_train, y_train)
            return pipeline
            
        st.toast(f"🕊️ PSO Swarming: {model_name}...", icon="🕊️")
        
        pso_opt = PSOHyperparameterOptimizer(
            estimator=pipeline,
            param_distributions=selected_space,
            cv=cv_strategy,
            n_particles=10, 
            max_iter=5
        )
        pso_opt.fit(X_train, y_train)
        return pso_opt.best_estimator_

    elif method == "Simulated Annealing":
        model_spaces = config.HPO_SPACES.get(model_name, {})
        selected_space = model_spaces.get("random", {})
        
        if not selected_space:
            pipeline.fit(X_train, y_train)
            return pipeline
            
        st.toast(f"🔥 SA Annealing: {model_name}...", icon="🔥")
        
        sa_opt = SimulatedAnnealingOptimizer(
            estimator=pipeline,
            param_distributions=selected_space,
            cv=cv_strategy,
            max_iter=20,     
            initial_temp=10, 
            cooling_rate=0.85 
        )
        sa_opt.fit(X_train, y_train)
        return sa_opt.best_estimator_

    # =========================================================
    # OPTUNA and HYPERBAND INTEGRATION
    # =========================================================
    if method in ["Optuna", "Hyperband","Bayesian Optimization"]:
        if optuna is None:
            st.error("Optuna library is not installed! Run 'pip install optuna'.")
            return pipeline

        def objective(trial):
            params = get_optuna_params(trial, model_name)
            if not params: 
                return float('inf')

            try:
                pipeline.set_params(**params)
            except Exception:
                pass 
            
            scores = cross_val_score(
                pipeline, X_train, y_train, 
                cv=cv_strategy, 
                scoring="neg_mean_squared_error", 
                n_jobs=-1
            )
            mse_score = -scores.mean()
            return mse_score

        pruner = HyperbandPruner(min_resource=1, max_resource="auto", reduction_factor=3) if method == "Hyperband" else None
        
        if method == "Bayesian Optimization":
            sampler = TPESampler(seed=config.DATA_CONFIG["random_state"])
        else:
            sampler = TPESampler(seed=config.DATA_CONFIG["random_state"])

        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        
        n_trials_count = 50 if method == "Hyperband" else 20
        
        study.optimize(objective, n_trials=n_trials_count, show_progress_bar=False)
        
        best_params = study.best_params
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        
        return pipeline

    # =========================================================
    # EXISTING RANDOM / GRID SEARCH (Legacy Code Block)
    # =========================================================
    # 3) HPO Spaces from CONFIG
    model_spaces = config.HPO_SPACES.get(model_name, {})
    
    if method == "Random Search":
        selected_space = model_spaces.get("random", {})
    else: 
        selected_space = model_spaces.get("grid", {})

    if not selected_space:
        pipeline.fit(X_train, y_train)
        return pipeline

    if model_name in config.MODEL_DEFAULT_PARAMS:
        allowed_keys = config.MODEL_DEFAULT_PARAMS[model_name].keys()
        pass 
    
    if not selected_space:
         pipeline.fit(X_train, y_train)
         return pipeline

    if method == "Random Search":
        search_engine = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=selected_space,
            n_iter=15, 
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
            random_state=config.DATA_CONFIG["random_state"],
            n_jobs=-1,
            verbose=0,
        )
    else:
        search_engine = GridSearchCV(
            estimator=pipeline,
            param_grid=selected_space,
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False,
            random_state=config.DATA_CONFIG["random_state"]
        )
        X_train, X_test = X_train.copy(), X_test.copy()
        y_train, y_test = y_train.copy(), y_test.copy()

        # AFTER SPLIT inside train_and_evaluate, insert:
        # 2. OUTLIER HANDLING
        X_train, y_train = apply_outlier_handling(X_train, y_train, outlier_methods)

        # 3. LOG TRANSFORM
        X_train, X_test = apply_log_transform(X_train, X_test, scaling_methods)

        # 4. SCALER SELECTION
        scaler_cls = get_scaler(scaling_methods)


    if _progress_callback: _progress_callback(10)

    # 5. TRAIN
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
            steps.append(("scaler", scaler_cls()))

        steps.append(("model", base_model))
        model = Pipeline(steps)
        model.fit(X_train, y_train)

    if _progress_callback: _progress_callback(90)

    # 6. PREDICT
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # 7. METRICS
# 7. METRICS (English Version)
    metrics = {}
    if active_metrics is None:
        active_metrics = config.METRICS_CONFIG["defaults"]

    # --- Basic Metrics ---
    if "MSE" in active_metrics: 
        metrics['MSE'] = mean_squared_error(y_test, y_pred)
    
    if "RMSE" in active_metrics: 
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
    
    if "MAE" in active_metrics: 
        metrics['MAE'] = mean_absolute_error(y_test, y_pred)
    
    # --- R2 and Derivatives ---
    r2_val = r2_score(y_test, y_pred)
    if "R2" in active_metrics: 
        metrics['R2'] = r2_val
    
    if "Adj_R2" in active_metrics:
        # Adjusted R2 Formula: 1 - (1-R2) * (n-1)/(n-p-1)
        n = len(y_test)
        p = X_test.shape[1]
        if n > p + 1:
            metrics['Adj_R2'] = 1 - (1 - r2_val) * (n - 1) / (n - p - 1)
        else:
            metrics['Adj_R2'] = r2_val # Return normal R2 if data is insufficient for adjustment

    # --- Error Distribution Metrics ---
    if "ExpVar" in active_metrics:
        metrics['ExpVar'] = explained_variance_score(y_test, y_pred)

    if "MAPE" in active_metrics: 
        metrics['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)

    if "MedAE" in active_metrics: 
        metrics['MedAE'] = median_absolute_error(y_test, y_pred)
        
    if "MaxErr" in active_metrics:
        metrics['MaxErr'] = max_error(y_test, y_pred)

    # --- Logarithmic Errors (Safe Calculation) ---
    if "MSLE" in active_metrics or "RMSLE" in active_metrics:
        # Rule 1: Actual values (y_test) must NEVER be negative for log metrics.
        if (y_test < 0).any():
            # If actual data contains negatives, log error is mathematically impossible.
            if "MSLE" in active_metrics: metrics['MSLE'] = None
            if "RMSLE" in active_metrics: metrics['RMSLE'] = None
        else:
            # Rule 2: If predictions (y_pred) are negative, clip them to 0.
            # This prevents crashes if the model predicts slightly negative values (e.g. -0.01).
            y_pred_safe = np.maximum(y_pred, 0)
            
            try:
                msle_val = mean_squared_log_error(y_test, y_pred_safe)
                
                if "MSLE" in active_metrics: 
                    metrics['MSLE'] = msle_val
                if "RMSLE" in active_metrics: 
                    metrics['RMSLE'] = np.sqrt(msle_val)
            except ValueError:
                # Return None in case of unexpected errors
                if "MSLE" in active_metrics: metrics['MSLE'] = None
                if "RMSLE" in active_metrics: metrics['RMSLE'] = None

    if _progress_callback: _progress_callback(100)

    return metrics, model, X_train, X_test, y_train, y_test, y_train_pred, y_pred

# -------------------------------------------------------------------------
# 4. DIAGNOSTICS - CACHED CALCULATIONS
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calculate_diagnostic_metrics(y_test_arr, y_pred_arr):
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
    
    try:
        if_model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42, n_jobs=1)
        if_model.fit(resid.reshape(-1, 1))
        if_score = -if_model.score_samples(resid.reshape(-1, 1))
        scores[:, 0] = (if_score - if_score.min()) / (np.ptp(if_score) + 1e-9)
    except: pass

    try:
        lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
        lof_pred = lof.fit_predict(resid.reshape(-1, 1))
        lof_score = -lof.negative_outlier_factor_
        scores[:, 1] = (lof_score - lof_score.min()) / (np.ptp(lof_score) + 1e-9)
    except: pass
    
    scores[:, 3] = (np.abs(resid) > 3 * resid_std).astype(float)

    agg = scores.mean(axis=1)
    norm_agg = (agg - agg.min()) / (np.ptp(agg) + 1e-9)
    
    labels = np.zeros_like(norm_agg, dtype=int)
    labels[norm_agg >= 0.75] = 2 
    labels[(norm_agg >= 0.4) & (norm_agg < 0.75)] = 1 
    
    return resid, std_resid, leverage, cooks_d, norm_agg, labels

def display_diagnostic_plots(
    y_test, y_pred, y_train=None, y_train_pred=None, 
    model=None, X_train=None, 
    key_prefix="diag",
    active_plots=None,
    cache_for_pdf=None
):
    if not active_plots:
        st.info("No diagnostic plots selected in the sidebar.")
        return

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    
    resid, std_resid, leverage, cooks_d, anomaly_score, anomaly_label = calculate_diagnostic_metrics(y_test_arr, y_pred_arr)
    
    has_train = (y_train is not None) and (y_train_pred is not None)
    if has_train:
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
    else:
        train_rmse = test_rmse = None

    final_tabs = [p for p in active_plots]
    if "Overfitting Check" in final_tabs and not has_train:
        final_tabs.remove("Overfitting Check")
        
    if not final_tabs:
        st.warning("No valid diagnostic plots available (train/test mismatch).")
        return

    tabs = st.tabs(final_tabs)
    layout_opts = dict(template='plotly_white', height=500)

    if cache_for_pdf is not None:
        if key_prefix not in cache_for_pdf:
            cache_for_pdf[key_prefix] = {"figs": {}}
        
        if "figs" not in cache_for_pdf[key_prefix]:
            cache_for_pdf[key_prefix]["figs"] = {}

        cache_for_pdf[key_prefix]["diag_metrics"] = (resid, leverage, cooks_d, anomaly_score, anomaly_label)

    for i, tab_name in enumerate(final_tabs):
        with tabs[i]:
            fig_to_cache = None
            
            plot_title = f"{key_prefix} - {tab_name}"

            # ---------- Diagnostic Plots ----------
            if tab_name == "Advanced Scatter": 
                df_chart = pd.DataFrame({'Actual': y_test_arr, 'Predicted': y_pred_arr})
                fig_to_cache = px.scatter(
                    df_chart, x='Actual', y='Predicted', 
                    trendline="ols", trendline_color_override="red",
                    marginal_x="histogram", marginal_y="histogram",
                    opacity=0.6, height=layout_opts['height'],
                    title=plot_title 
                )

            elif tab_name == "Overfitting Check" and has_train:
                fig_to_cache = go.Figure(go.Bar(
                    x=["Train", "Test"], y=[train_rmse, test_rmse],
                    marker_color=["#1f77b4", "#ff7f0e"],
                    text=[f"{train_rmse:.4f}", f"{test_rmse:.4f}"], textposition="auto"
                ))
                fig_to_cache.update_layout(title=f"{key_prefix} - RMSE Comparison (Train vs Test)", **layout_opts) 

            elif tab_name == "Residuals":
                fig_to_cache = px.scatter(
                    x=y_pred_arr, y=resid,
                    color=anomaly_score, color_continuous_scale="Turbo",
                    labels={'x': 'Predicted', 'y': 'Residual'},
                    title=f"{key_prefix} - Residuals vs Predicted", height=layout_opts['height'] 
                )
                fig_to_cache.add_hline(y=0, line_dash='dash', line_color='black')

            elif tab_name == "Distribution":
                fig_to_cache = px.histogram(resid, nbins=50, marginal="box", 
                                          title=f"{key_prefix} - Residual Distribution", height=layout_opts['height']) 

            elif tab_name == "QQ Plot":
                qq = probplot(resid, dist='norm')
                fig_to_cache = px.scatter(x=qq[0][0], y=qq[0][1],
                                          labels={'x': 'Theoretical', 'y': 'Observed'},
                                          title=f"{key_prefix} - Q-Q Plot", height=layout_opts['height']) 
                fig_to_cache.add_shape(type="line", x0=min(qq[0][0]), y0=min(qq[0][0]),
                                       x1=max(qq[0][0]), y1=max(qq[0][0]), line=dict(color="red"))

            elif tab_name == "Influence":
                fig_to_cache = px.scatter(
                    x=leverage, y=cooks_d, color=anomaly_score, color_continuous_scale="Viridis",
                    labels={'x': 'Leverage', 'y': "Cook's D"}, 
                    title=f"{key_prefix} - Influence Plot", height=layout_opts['height'] 
                )
                fig_to_cache.add_hline(y=4/len(resid), line_dash="dash", line_color="red")

            elif tab_name == "Residual vs Actual":
                df_rva = pd.DataFrame({"Actual": y_test_arr, "Residual": resid})
                fig_to_cache = px.scatter(
                    df_rva, x="Actual", y="Residual",
                    color=np.abs(resid),
                    color_continuous_scale="Plasma",
                    title=f"{key_prefix} - Residuals vs Actual Values", 
                    labels={"color": "|Residual|"},
                    opacity=0.7,
                    height=layout_opts['height']
                )
                fig_to_cache.add_hline(y=0, line_dash="dash", line_color="black")

            elif tab_name == "Error Bands":
                df_err = pd.DataFrame({
                    "Index": np.arange(len(y_pred_arr)),
                    "Predicted": y_pred_arr,
                    "Actual": y_test_arr,
                    "Residual": resid
                })
                sigma = np.std(resid)
                fig_to_cache = go.Figure()
                fig_to_cache.add_trace(go.Scatter(x=df_err["Index"], y=df_err["Predicted"],
                                                  mode="lines", name="Predicted", line=dict(color="blue")))
                fig_to_cache.add_trace(go.Scatter(x=df_err["Index"], y=df_err["Predicted"] + sigma,
                                                  mode="lines", name="+1σ", line=dict(color="green", dash="dash")))
                fig_to_cache.add_trace(go.Scatter(x=df_err["Index"], y=df_err["Predicted"] - sigma,
                                                  mode="lines", name="-1σ", line=dict(color="green", dash="dash")))
                fig_to_cache.add_trace(go.Scatter(x=df_err["Index"], y=df_err["Actual"],
                                                  mode="markers", name="Actual", marker=dict(size=6, color="orange")))
                fig_to_cache.update_layout(title=f"{key_prefix} - Predicted with Error Bands (±1σ)", **layout_opts) 

            elif tab_name == "KDE Distribution":
                fig_to_cache = ff.create_distplot([resid], group_labels=["Residuals"], show_hist=True, show_curve=True)
                fig_to_cache.update_layout(title=f"{key_prefix} - Residuals Distribution + KDE Curve", **layout_opts) 

            elif tab_name == "Bubble Influence":
                df_bubble = pd.DataFrame({"Leverage": leverage, "Residual": resid, "Cook": cooks_d})
                fig_to_cache = px.scatter(
                    df_bubble,
                    x="Leverage", y="Residual",
                    size="Cook", color="Cook",
                    color_continuous_scale="Inferno",
                    title=f"{key_prefix} - Leverage vs Residuals (Bubble Size = Cook's D)", 
                    height=layout_opts["height"],
                    labels={"Cook": "Cook's Distance"}
                )
                fig_to_cache.add_hline(y=0, line_dash="dash", line_color="black")

            # ---------- Streamlit ve PDF Cache ----------
            if fig_to_cache:
                st.plotly_chart(fig_to_cache, use_container_width=True, key=f"{key_prefix}_{tab_name.replace(' ', '_')}")
                if cache_for_pdf is not None:
                    cache_for_pdf[key_prefix]["figs"][tab_name] = fig_to_cache

# -------------------------------------------------------------------------
# 5. XAI (EXPLAINABLE AI) SETTINGS
# -------------------------------------------------------------------------
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
    if isinstance(vals, list): vals = vals[0]
    elif hasattr(vals, 'values'): vals = vals.values
    return vals

def shap_importance_df(shap_values, X_shap):
    vals = shap_values
    if isinstance(vals, list): vals = np.array(vals[0])
    elif hasattr(vals, 'values'): vals = np.array(vals.values)
    if vals.ndim == 3: vals = vals[0]
    abs_vals = np.mean(np.abs(vals), axis=0)
    imp_df = pd.DataFrame({'Feature': X_shap.columns.tolist(), 'Importance': abs_vals}).sort_values('Importance', ascending=False)
    return imp_df

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

# --------------------------------------------------------------------------
# 1. SHAP ANALİZ FONKSİYONU
# --------------------------------------------------------------------------
def analyze_shap(model, estimator, X_test, key_suffix, cache_for_pdf=None):
    """
    SHAP analizlerini gerçekleştirir ve SHAP önem tablosunu döndürür.
    """
    shap_imp_df = None
    st.markdown(f'## 🌈 SHAP Global & Local Explanations ({key_suffix})')
    
    try:
        # Veri setini hazırla (Pipeline ise scale et)
        X_shap = X_test.iloc[:min(300, len(X_test))].copy()
        is_pipeline = isinstance(model, Pipeline)
        
        if is_pipeline and 'scaler' in model.named_steps:
            try:
                X_shap = pd.DataFrame(
                    model.named_steps['scaler'].transform(X_shap),
                    columns=X_shap.columns, 
                    index=X_shap.index
                )
            except: 
                pass

        # Explainer oluştur
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_shap)
        except:
            explainer = shap.Explainer(estimator, X_shap)
            shap_values = explainer(X_shap)

        # shap_importance_df fonksiyonunun tanımlı olduğu varsayılmaktadır
        shap_imp_df = shap_importance_df(shap_values, X_shap)

        shap_tabs = st.tabs(['🎯 Summary Plot', '📊 Feature Importance', '🔎 Instance Waterfall', '⚡ Force Plot'])

        # --- Tab 1: Summary Plot ---
        with shap_tabs[0]:
            fig, ax = plt.subplots(figsize=(7,5))
            plt.title(f"{key_suffix} - SHAP Summary") 
            shap.summary_plot(shap_values, X_shap, show=False)
            st.pyplot(fig)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["SHAP Summary"] = fig

        # --- Tab 2: Feature Importance ---
        with shap_tabs[1]:
            if shap_imp_df is not None and not shap_imp_df.empty:
                sub1, sub2 = st.tabs(["📋 Table", "📊 Chart"])
                with sub1:
                    st.dataframe(shap_imp_df)
                    if cache_for_pdf is not None:
                        cache_for_pdf[key_suffix]["metrics"]["SHAP Table"] = shap_imp_df
                with sub2:
                    fig_imp = px.bar(
                        shap_imp_df.head(20),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f'{key_suffix} - Top SHAP Feature Importance' 
                    )
                    fig_imp.update_layout(template='plotly_white', height=450)
                    st.plotly_chart(fig_imp, use_container_width=True)
                    if cache_for_pdf is not None:
                        cache_for_pdf[key_suffix]["figs"]["SHAP Feature Importance"] = fig_imp

        # --- Tab 3: Waterfall Plot ---
        with shap_tabs[2]:
            idx_w = st.number_input('Select Instance Index', 0, len(X_shap)-1, 0,
                                    key=f'shap_w_input_{key_suffix}')
            vals = shap_values.values if hasattr(shap_values,'values') else shap_values
            if isinstance(vals, list):
                vals = np.array(vals[0])
            if vals.ndim == 3:
                vals = vals[0]
                
            df_w = pd.DataFrame({'Feature': X_shap.columns.tolist(), 'SHAP': vals[idx_w]})
            df_w = df_w.sort_values('SHAP', key=np.abs, ascending=False).head(12)
            
            # Base value kontrolü
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[0]

            df_w['Contribution'] = df_w['SHAP'].cumsum() + float(base_val)
            
            fig_w = px.bar(df_w, x='SHAP', y='Feature', orientation='h',
                           title=f'{key_suffix} - SHAP Waterfall (Instance {idx_w})', color='SHAP') 
            fig_w.update_layout(template='plotly_white', height=450)
            st.plotly_chart(fig_w, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"][f"SHAP Waterfall {idx_w}"] = fig_w

        # --- Tab 4: Force Plot ---
        with shap_tabs[3]:
            idx_f = st.number_input('Select Instance for Force Plot', 0, len(X_shap)-1, 0,
                                    key=f'shap_force_idx_{key_suffix}')
            vals = shap_values.values if hasattr(shap_values, 'values') else shap_values
            if isinstance(vals, list):
                vals = np.array(vals[0])
            if vals.ndim == 3:
                vals = vals[0]
            instance_shap = vals[idx_f]
            
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(base_val[0])
                
            force_plot = shap.force_plot(
                base_val,
                instance_shap,
                X_shap.iloc[idx_f, :],
                matplotlib=False
            )
            st.write(f"**Model:** {key_suffix}") 
            # st_shap fonksiyonunun streamlit-shap kütüphanesinden geldiği varsayılmaktadır
            st_shap(force_plot, height=300)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"][f"SHAP Force Plot {idx_f}"] = force_plot

    except Exception as e:
        st.error(f'SHAP Error: {e}')
        
    return shap_imp_df

# --------------------------------------------------------------------------
# 2. PFI (Permutation Feature Importance) ANALİZ FONKSİYONU
# --------------------------------------------------------------------------
def analyze_pfi(estimator, X_test, y_test, key_suffix, cache_for_pdf=None):
    """
    PFI analizlerini gerçekleştirir ve önem tablosunu döndürür.
    """
    pfi_imp_df = None
    st.markdown(f'## 🚀 PFI - {key_suffix}')
    
    try:
        r = permutation_importance(
            estimator, X_test, y_test,
            n_repeats=15,
            random_state=42,
            n_jobs=-1
        )
        idx = r.importances_mean.argsort()[::-1]
        pfi_imp_df = pd.DataFrame({
            'Feature': X_test.columns[idx],
            'Importance': r.importances_mean[idx],
            'Std Dev': r.importances_std[idx]
        })

        tab_table, tab_bar, tab_box, tab_errorbar = st.tabs([
            "📋 Table", "📊 Bar Chart", "📉 Boxplot", "📈 Error Bars"
        ])

        # --- Tab 1: Table ---
        with tab_table:
            st.dataframe(pfi_imp_df)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["metrics"]["PFI Table"] = pfi_imp_df

        # --- Tab 2: Bar Chart ---
        with tab_bar:
            fig_pfi = px.bar(
                pfi_imp_df.head(20),
                x='Importance',
                y='Feature',
                orientation='h',
                error_x='Std Dev',
                title=f'{key_suffix} - Top 20 PFI Features' 
            )
            fig_pfi.update_layout(template='plotly_white', height=450)
            st.plotly_chart(fig_pfi, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["PFI Bar"] = fig_pfi

        # --- Tab 3: Boxplot ---
        with tab_box:
            fig_box = px.box(
                r.importances.T[:, idx][:, :20],
                labels={'variable': 'Feature', 'value': 'Importance'},
                title=f'{key_suffix} - PFI Distribution' 
            )
            fig_box.update_layout(
                template='plotly_white',
                height=500,
                xaxis=dict(
                    tickvals=list(range(20)),
                    ticktext=list(pfi_imp_df['Feature'].head(20))
                )
            )
            st.plotly_chart(fig_box, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["PFI Boxplot"] = fig_box

        # --- Tab 4: Error Bars ---
        with tab_errorbar:
            fig_err = px.scatter(
                pfi_imp_df.head(20),
                x='Importance',
                y='Feature',
                error_x='Std Dev',
                title=f'{key_suffix} - PFI Importance ± Std Dev' 
            )
            fig_err.update_traces(mode='markers')
            fig_err.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig_err, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["PFI Error Bars"] = fig_err

    except Exception as e:
        st.error(f'PFI Error: {e}')
        
    return pfi_imp_df

# --------------------------------------------------------------------------
# 3. LIME ANALİZ FONKSİYONU
# --------------------------------------------------------------------------
def analyze_lime(model, X_train, X_test, key_suffix, lime_num_features, cache_for_pdf=None):
    """
    LIME analizlerini gerçekleştirir.
    """
    st.markdown(f'## 💡 LIME - {key_suffix}')
    
    try:
        # Import lime_tabular here or at top
        from lime import lime_tabular 
        
        explainer_lime = lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns.tolist(),
            mode='regression',
            discretize_continuous=True,
            verbose=False
        )

        # --- Single Instance Selection ---
        idx = st.number_input(
            'Select Test Instance',
            0, len(X_test)-1, 0,
            key=f'lime_idx_{key_suffix}'
        )
        exp = explainer_lime.explain_instance(
            X_test.iloc[idx].values,
            lambda x: model.predict(x),
            num_features=lime_num_features
        )
        lime_df = pd.DataFrame(exp.as_list(), columns=['feature', 'LIME Score'])
        lime_df['Percentage Importance LIME'] = (
            lime_df['LIME Score'].abs() / lime_df['LIME Score'].abs().sum()
        ) * 100

        lime_tabs = st.tabs([
            "🔍 Instance Explanation", 
            "📄 Contribution Table", 
            "📊 Bar Chart", 
            "🐝 Multi-Instance Summary"
        ])

        # --- Tab 1: Instance Explanation ---
        with lime_tabs[0]:
            st.caption(f"LIME Explanation for Model: {key_suffix}")
            # components'in streamlit.components.v1 olduğu varsayılır
            if 'components' in globals() and components is not None:
                components.html(exp.as_html(), height=450, scrolling=True)
            else:
                # Fallback if components is not imported
                st.components.v1.html(exp.as_html(), height=450, scrolling=True)

        # --- Tab 2: Contribution Table ---
        with lime_tabs[1]:
            fig_table = go.Figure(data=[go.Table(
                header=dict(
                    values=["Feature", "LIME Score", "% Importance"],
                    fill_color='darkslategray',
                    font=dict(color='white', size=14),
                    align='left'
                ),
                cells=dict(
                    values=[
                        lime_df['feature'],
                        np.round(lime_df['LIME Score'], 4),
                        np.round(lime_df['Percentage Importance LIME'], 2)
                    ],
                    fill_color='lavender',
                    align='left',
                    font=dict(color='black', size=12)
                )
            )])
            st.plotly_chart(fig_table, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["metrics"]["LIME Table"] = lime_df

        # --- Tab 3: Enhanced Bar Chart ---
        with lime_tabs[2]:
            fig_bar = px.bar(
                lime_df.sort_values("LIME Score", ascending=True),
                x="LIME Score",
                y="feature",
                orientation="h",
                color="LIME Score",
                color_continuous_scale=px.colors.diverging.RdBu,
                color_continuous_midpoint=0,
                title=f"{key_suffix} - LIME Feature Contributions",
                labels={"feature": "Feature", "LIME Score": "Score"},
                height=500
            )
            fig_bar.update_layout(
                title_font=dict(size=18, family="Arial, bold"),
                yaxis=dict(tickfont=dict(size=12)),
                xaxis=dict(tickfont=dict(size=12))
            )
            fig_bar.update_traces(
                hovertemplate="<b>%{y}</b><br>LIME Score: %{x:.4f}<br>% Importance: %{customdata[0]:.2f}%",
                customdata=np.stack((lime_df['Percentage Importance LIME'],), axis=-1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["LIME Bar"] = fig_bar

        # --- Tab 4: Multi-instance Summary ---
        with lime_tabs[3]:
            st.caption(f"Multi-instance LIME Summary for {key_suffix}")
            num_instances = min(50, len(X_test)) 
            lime_summary = []

            for i in range(num_instances):
                exp_i = explainer_lime.explain_instance(
                    X_test.iloc[i].values,
                    lambda x: model.predict(x),
                    num_features=len(X_train.columns)
                )
                temp_df = pd.DataFrame(exp_i.as_list(), columns=['feature', 'LIME Score'])
                temp_df['Instance'] = i
                lime_summary.append(temp_df)

            lime_summary_df = pd.concat(lime_summary, ignore_index=True)

            summary_plot_type = st.radio(
                "Select Multi-instance Summary Plot Type",
                options=["Heatmap", "Beeswarm"],
                index=0,
                key=f"multi_lime_plot_{key_suffix}"
            )

            if summary_plot_type == "Heatmap":
                heatmap_data = lime_summary_df.pivot(index='feature', columns='Instance', values='LIME Score')
                fig_heatmap = px.imshow(
                    heatmap_data,
                    color_continuous_scale='RdBu',
                    aspect='auto',
                    origin='lower',
                    labels=dict(x="Instance", y="Feature", color="LIME Score"),
                    title=f"{key_suffix} - Multi-instance LIME Heatmap"
                )
                fig_heatmap.update_layout(height=600, width=900)
                st.plotly_chart(fig_heatmap, use_container_width=True)

            elif summary_plot_type == "Beeswarm":
                fig_beeswarm = px.strip(
                    lime_summary_df,
                    x="LIME Score",
                    y="feature",
                    color="LIME Score",
                    color_continuous_scale='RdBu',
                    hover_data=["Instance"],
                    title=f"{key_suffix} - Multi-instance LIME Beeswarm",
                    orientation='h'
                )
                fig_beeswarm.update_layout(
                    height=600,
                    width=900,
                    yaxis=dict(categoryorder='total ascending')
                )
                st.plotly_chart(fig_beeswarm, use_container_width=True)

    except Exception as e:
        st.error(f"LIME Error: {e}")

# --------------------------------------------------------------------------
# 4. COUNTERFACTUAL (DICE) ANALİZ FONKSİYONU
# --------------------------------------------------------------------------
def analyze_counterfactual(estimator, X_train, X_test, key_suffix, cache_for_pdf=None):
    """
    Counterfactual (Dice) analizlerini gerçekleştirir.
    """
    st.markdown(f"## 🔄 Counterfactual - {key_suffix}")
    
    try:
        import dice_ml  # Fonksiyon içinde import
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📄 Tabular View",
            "📊 Difference Heatmap",
            "🧭 Radar Plot",
            "🔧 Minimal Feature Change",
            "📈 CF Prediction Plot"
        ])

        idx = st.number_input("Select Test Instance", 0, len(X_test)-1, 0,
                              key=f"cf_idx_{key_suffix}")
        x0 = X_test.iloc[[idx]].copy()
        y0 = estimator.predict(x0)[0]
        delta = abs(y0)*0.2 if y0 != 0 else 1

        df_dice = X_train.copy()
        outcome_col = "target"  
        df_dice[outcome_col] = estimator.predict(X_train)

        data_dice = dice_ml.Data(
            dataframe=df_dice,
            continuous_features=X_train.columns.tolist(),
            outcome_name=outcome_col
        )
        model_dice = dice_ml.Model(model=estimator, backend="sklearn", model_type="regressor")
        exp_cf = dice_ml.Dice(data_dice, model_dice, method="random")

        desired_range = [y0 - delta, y0 + delta]

        cf = exp_cf.generate_counterfactuals(
            x0,
            total_CFs=3,
            desired_range=desired_range
        )
        cf_df = cf.cf_examples_list[0].final_cfs_df.copy()

        if "type" not in cf_df.columns:
            cf_df["type"] = "CF"
        original_df = x0.copy()
        original_df["type"] = "Original"
        
        # --- Tab 1: Tabular View ---
        with tab1:
            st.dataframe(cf_df)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["metrics"]["Counterfactual Table"] = cf_df

        cols_to_drop = ['type', outcome_col]
        cf_features = cf_df.drop(columns=cols_to_drop, errors='ignore')

        # --- Tab 2: Difference Heatmap ---
        with tab2:
            diff = cf_features.subtract(x0.values[0])
            fig_diff = px.imshow(diff.T, text_auto=True, aspect="auto",
                                 labels=dict(x="CF Instance", y="Feature", color="Difference"),
                                 title=f"{key_suffix} - Feature Difference Heatmap") 
            st.plotly_chart(fig_diff, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["CF Heatmap"] = fig_diff

        # --- Tab 3: Radar Plot ---
        with tab3:
            fig_radar_cf = px.line_polar(
                cf_features.T,
                r=cf_features.T.values.flatten(),
                theta=cf_features.T.index.repeat(cf_features.shape[0]),
                line_close=True, title=f"{key_suffix} - CF Radar Plot"
            ) 
            st.plotly_chart(fig_radar_cf, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["CF Radar"] = fig_radar_cf

        # --- Tab 4: Minimal Feature Change ---
        with tab4:
            min_change = (cf_features - x0.values).abs().min(axis=1)
            fig_min = px.bar(
                x=[f"CF{i}" for i in range(len(min_change))],
                y=min_change,
                title=f"{key_suffix} - Minimal Feature Change per CF"
            ) 
            st.plotly_chart(fig_min, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["CF Min Change"] = fig_min

        # --- Tab 5: CF Prediction Plot ---
        with tab5:
            pred_vals = estimator.predict(cf_features)
            fig_pred = px.bar(
                x=[f"CF{i}" for i in range(len(pred_vals))],
                y=pred_vals,
                title=f"{key_suffix} - Counterfactual Predictions"
            ) 
            st.plotly_chart(fig_pred, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["CF Predictions"] = fig_pred

    except Exception as e:
        st.error(f"Counterfactual Error: {e}")

# --------------------------------------------------------------------------
# 5. ANCHOR ANALİZ FONKSİYONU
# --------------------------------------------------------------------------
def analyze_anchor(estimator, X_train, X_test, key_suffix, cache_for_pdf=None):
    """
    Anchor analizlerini gerçekleştirir.
    """
    st.markdown(f"## 🪝 Anchor - {key_suffix}")
    
    try:
        from alibi.explainers import AnchorTabular
        
        feature_names = X_train.columns.tolist()
        X_train_np = X_train.values
        X_test_np = X_test.values
        predict_fn = lambda x: estimator.predict(x)

        explainer_anchor = AnchorTabular(predict_fn, feature_names=feature_names)
        explainer_anchor.fit(X_train_np)

        idx = st.number_input(
            "Select Test Instance for Anchor",
            0, len(X_test_np)-1, 0,
            key=f"anchor_idx_{key_suffix}"
        )
        exp_anchor = explainer_anchor.explain(X_test_np[int(idx)])
        anchor_rules = list(exp_anchor.anchor)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📌 Anchor Rules", "🌊 Precision & Coverage", "📈 3D Heatmap",
            "🌳 Decision Tree", "📡 Neighborhood Stats"
        ])

        # --- Tab 1: Rules ---
        with tab1:
            st.write(anchor_rules)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["metrics"]["Anchor Rules"] = anchor_rules

        # --- Tab 2: Stats ---
        with tab2:
            st.metric("Precision", exp_anchor.precision)
            st.metric("Coverage", exp_anchor.coverage)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["metrics"]["Anchor Precision"] = exp_anchor.precision
                cache_for_pdf[key_suffix]["metrics"]["Anchor Coverage"] = exp_anchor.coverage

        # --- Tab 3: 3D Heatmap ---
        with tab3:
            y_pred = estimator.predict(X_test_np)
            fig_heat = px.scatter_3d(
                x=X_test_np[:,0], y=X_test_np[:,1], z=y_pred,
                color=y_pred, title=f"{key_suffix} - Local Predictions 3D Heatmap" 
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["Anchor Heatmap"] = fig_heat

        # --- Tab 4: Decision Tree ---
        with tab4:
            if hasattr(estimator, 'estimators_'):
                tree = estimator.estimators_[0].tree_
                fig_tree = px.treemap(
                    pd.DataFrame({
                        "Feature": [feature_names[i] if i >= 0 else "Leaf" for i in tree.feature],
                        "Samples": tree.n_node_samples
                    }),
                    path=["Feature"], values="Samples",
                    title=f"{key_suffix} - Decision Tree Treemap" 
                )
            elif hasattr(estimator, 'tree_'):
                tree = estimator.tree_
                fig_tree = px.treemap(
                    pd.DataFrame({
                        "Feature": [feature_names[i] if i >= 0 else "Leaf" for i in tree.feature],
                        "Samples": tree.n_node_samples
                    }),
                    path=["Feature"], values="Samples",
                    title=f"{key_suffix} - Decision Tree Treemap" 
                )
            else:
                st.info("Decision tree visualization not available for this model.")
                fig_tree = None

            if fig_tree is not None:
                st.plotly_chart(fig_tree, use_container_width=True)
                if cache_for_pdf is not None:
                    cache_for_pdf[key_suffix]["figs"]["Anchor Tree"] = fig_tree

        # --- Tab 5: Neighborhood Stats ---
        with tab5:
            mask = np.ones(len(X_train), dtype=bool)
            for rule in anchor_rules:
                parts = rule.split(" ")
                if len(parts) == 3 and parts[1] in ['<=','>']:
                    f, op, val = parts
                    val = float(val)
                    if op == "<=":
                        mask &= X_train[f] <= val
                    else:
                        mask &= X_train[f] > val
            neigh_df = X_train[mask]
            st.dataframe(neigh_df.head(20))

            fig_neigh = px.histogram(
                neigh_df, x=feature_names[0], nbins=20,
                title=f"{key_suffix} - Neighborhood of '{feature_names[0]}'"
            ) 
            st.plotly_chart(fig_neigh, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["Anchor Neighborhood"] = fig_neigh

    except Exception as e:
        st.error(f"Anchor Error: {e}")

# --------------------------------------------------------------------------
# 6. RADAR CHART ANALİZ FONKSİYONU
# --------------------------------------------------------------------------
def analyze_radar(pfi_imp_df, shap_imp_df, key_suffix, radar_top_k, cache_for_pdf=None):
    """
    PFI ve SHAP sonuçlarını birleştiren Radar grafiğini çizer.
    """
    try:
        with st.expander('📡 Combined Radar'):
            # create_combined_radar fonksiyonunun tanımlı olduğu varsayılmaktadır
            fig_radar = create_combined_radar(
                pfi_imp_df if pfi_imp_df is not None else pd.DataFrame({'Feature':[], 'Importance':[]}), 
                shap_imp_df, 
                top_k=radar_top_k,
                title=f"{key_suffix} - PFI + SHAP Radar" 
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            if cache_for_pdf is not None:
                cache_for_pdf[key_suffix]["figs"]["Combined Radar"] = fig_radar
    except Exception as e:
        st.error(f'Radar Error: {e}')

# --------------------------------------------------------------------------
# 7. ANA ORKESTRASYON FONKSİYONU
# --------------------------------------------------------------------------
def run_xai_analysis(model, X_train, X_test, y_test, methods, key_suffix,
                      cache_for_pdf=None,
                      lime_sample=config.XAI_CONFIG["lime_sample_size"], 
                      lime_num_features=config.XAI_CONFIG["lime_num_features"], 
                      radar_top_k=config.XAI_CONFIG["radar_top_k"]):
    
    if not methods:
        st.info("No XAI methods selected.")
        return

    # PDF Cache başlatma
    if cache_for_pdf is not None:
        if key_suffix not in cache_for_pdf:
            cache_for_pdf[key_suffix] = {"figs": {}, "metrics": {}}

    xai_tabs = st.tabs(methods)
    is_pipeline = isinstance(model, Pipeline)
    estimator = model.named_steps['model'] if is_pipeline else model

    shap_imp_df = None
    pfi_imp_df = None

    # SHAP
    if 'SHAP' in methods and shap is not None:
        with xai_tabs[methods.index('SHAP')]:
            shap_imp_df = analyze_shap(model, estimator, X_test, key_suffix, cache_for_pdf)

    # PFI
    if 'PFI' in methods:
        with xai_tabs[methods.index('PFI')]:
            pfi_imp_df = analyze_pfi(estimator, X_test, y_test, key_suffix, cache_for_pdf)

    # LIME
    if 'LIME' in methods:
        with xai_tabs[methods.index('LIME')]:
            analyze_lime(model, X_train, X_test, key_suffix, lime_num_features, cache_for_pdf)

    # Counterfactual
    if 'Counterfactual' in methods:
        with xai_tabs[methods.index('Counterfactual')]:
            analyze_counterfactual(estimator, X_train, X_test, key_suffix, cache_for_pdf)

    # Anchor
    if 'Anchor' in methods:
        with xai_tabs[methods.index('Anchor')]:
            analyze_anchor(estimator, X_train, X_test, key_suffix, cache_for_pdf)

    # Radar Chart (PFI veya SHAP varsa)
    if ('PFI' in methods or 'SHAP' in methods):
        analyze_radar(pfi_imp_df, shap_imp_df, key_suffix, radar_top_k, cache_for_pdf)

# ---------------------------
# RUN DASHBOARD
# ---------------------------
def run_dashboard(results, selected_diag_plots, xai_ops):
    if not results:
        st.info("👈 Upload data & Start Training.")
        return
    
    cached_diagnostics = {} 

    # ------------------ Leaderboard ------------------
    with st.expander("### 🏆 Leaderboard", expanded=False):
        lb_tabs = st.tabs(["📋 Table", "📈 Chart", "⚙️ Parameters"])
        
        # Prepare Table Data
        table_data = []
        for k, v in results.items():
            row = v['metrics'].copy()
            row['Model'] = v.get('base_model', k) 
            row['Processing'] = v.get('proc_info', '-')
            table_data.append(row)
        df_res = pd.DataFrame(table_data)
        
        # Reorder columns safely
        base_cols = ['Model', 'Processing']
        other_cols = [c for c in df_res.columns if c not in base_cols]
        df_res = df_res[base_cols + other_cols]

        # --- TAB 1: TABLE---
        with lb_tabs[0]:
            numeric_cols = df_res.select_dtypes(include=[np.number]).columns.tolist()
            
            hl_min_cols = [c for c in ["MSE", "RMSE", "MAE", "MAPE"] if c in df_res.columns]
            hl_max_cols = [c for c in ["R2", "Adj_R2"] if c in df_res.columns]

            st_style = df_res.style.format(subset=numeric_cols, precision=4)
            
            if hl_min_cols:
                st_style = st_style.highlight_min(subset=hl_min_cols, color='#d9f2d9') # Light Green
            if hl_max_cols:
                st_style = st_style.highlight_max(subset=hl_max_cols, color='#ffcccc') # Light Red

            st.dataframe(st_style, use_container_width=True)

        # --- TAB 2: CHART ---
        with lb_tabs[1]:
            df_chart = pd.DataFrame([{**{'Model': k}, **v['metrics']} for k, v in results.items()])
            fig = go.Figure()
            
            metrics_to_plot = [col for col in df_chart.columns if col not in ['Model','R2'] and pd.api.types.is_numeric_dtype(df_chart[col])]
            
            for metric in metrics_to_plot:
                text_labels = []
                for val in df_chart[metric]:
                    try:
                        text_labels.append(f"{val:.4f}")
                    except (ValueError, TypeError):
                        text_labels.append(str(val))

                fig.add_trace(go.Bar(
                    x=df_chart['Model'], 
                    y=df_chart[metric], 
                    name=metric,
                    text=text_labels, 
                    textposition="outside"
                ))
            
            if "R2" in df_chart.columns and pd.api.types.is_numeric_dtype(df_chart["R2"]):
                fig.add_trace(go.Scatter(x=df_chart['Model'], y=df_chart["R2"], mode='markers+lines', 
                                         name="R2", line=dict(color='green', dash='dash')))
            
            fig.update_layout(title="Leaderboard Metrics", barmode='group', template="plotly_white", height=550)
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 3: PARAMETERS ---
        with lb_tabs[2]:
            param_list = []
            for m_name, res in results.items():
                model_obj = res['model']
                final_model = model_obj.named_steps['model'] if isinstance(model_obj, Pipeline) else model_obj
                params = final_model.get_params()
                base_name = m_name.split(' (')[0].strip()
                row = {"Model": m_name}
                if base_name in config.MODEL_DEFAULT_PARAMS:
                    for k in config.MODEL_DEFAULT_PARAMS[base_name].keys():
                        if k in params:
                            disp_name = config.UNIFIED_PARAM_NAMES.get(k, k)
                            val = params[k]
                            row[disp_name] = str(val) if isinstance(val, (list, tuple)) else val
                param_list.append(row)
            if param_list:
                st.dataframe(pd.DataFrame(param_list).fillna("-").astype(str), use_container_width=True)

    # ------------------ Model Visualizations ------------------
    with st.expander("### 📊 Visualization"):
        model_tabs = st.tabs(list(results.keys()))
        for i, model_name in enumerate(results.keys()):
            res = results[model_name]
            # Initialize cache dict for this model
            cached_diagnostics[model_name] = {"metrics": res['metrics'], "figs": {}}
            
            with model_tabs[i]:
                ftab, dtab, xtab = st.tabs(["📉 Forecast", "🔍 Diagnostics", "🧠 XAI"])

                # ---------- Forecast ----------
                with ftab:
                    max_limit = len(res['yte'])
                    limit = st.slider(f"Select number of points to display for {model_name}",
                                      min_value=10, max_value=min(500, max_limit), value=min(200, max_limit))
                    df_viz = pd.DataFrame({'Actual': res['yte'].iloc[:limit], 'Predicted': res['ypr'][:limit]})
                    fig_fc = px.line(df_viz, markers=True, title=f"{model_name} Forecast",
                                     labels={'index': 'Index', 'value': 'Value', 'variable': 'Legend'}, template="plotly_white")
                    fig_fc.update_traces(line=dict(width=3), marker=dict(size=6))
                    fig_fc.update_layout(legend=dict(title="Series"), xaxis_title="Time/Index", yaxis_title="Value",
                                         hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40))
                    st.plotly_chart(fig_fc, use_container_width=True, key=f"{model_name}_fc")
                    cached_diagnostics[model_name]["figs"]["Forecast"] = fig_fc

                # ---------- Diagnostics ----------
                with dtab:
                    display_diagnostic_plots(
                        res['yte'], res['ypr'], res.get('ytr'), res.get('ytr_pr'),
                        key_prefix=model_name, active_plots=selected_diag_plots,
                        cache_for_pdf=cached_diagnostics
                    )

                # ---------- XAI  ----------
                with xtab:
                    run_xai_analysis(
                        res['model'], 
                        res['Xt'], 
                        res['Xte'], 
                        res['yte'], 
                        xai_ops, 
                        key_suffix=model_name,
                        cache_for_pdf=cached_diagnostics
                    )
    # ------------------ Exports ------------------
    with st.expander("### 📥 Exports"):
        c1, c2 = st.columns(2)
        c1.download_button("Download Metrics (CSV)", df_res.to_csv(index=False).encode('utf-8'), "metrics.csv", "text/csv")
        if c2.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf_data = generate_pdf_report(results, cached_diagnostics)
                    st.download_button("Download PDF", pdf_data, "AutoML_Report.pdf", "application/pdf")
                except Exception as e:
                    st.error(f"PDF Error: {e}")

# ---------------------------
# GENERATE PDF REPORT
# ---------------------------
def generate_pdf_report(results, cached_diagnostics):
    buf = BytesIO()
    
    with PdfPages(buf) as pdf:
        
        fig_cover = plt.figure(figsize=(11.69, 8.27))
        plt.axis('off')
        report_title = config.APP_CONFIG.get("pdf_title", "AutoML Professional Report")
        plt.text(0.5, 0.6, report_title, ha='center', va='center', fontsize=28, fontweight='bold', color='#2E4053')
        plt.text(0.5, 0.5, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ha='center', fontsize=14, color='gray')
        pdf.savefig(fig_cover)
        plt.close()

        # 2. LEADERBOARD
        if results:
            dfm = pd.DataFrame([{**{'Model': k}, **v['metrics']} for k, v in results.items()])
            cols = ['Model'] + [c for c in dfm.columns if c != 'Model']
            dfm = dfm[cols]
            numeric_cols = dfm.select_dtypes(include=[np.number]).columns
            dfm[numeric_cols] = dfm[numeric_cols].round(4)

            fig_lb = plt.figure(figsize=(11.69, 8.27))
            ax_table = fig_lb.add_subplot(111)
            ax_table.axis('off')
            
            table = ax_table.table(cellText=dfm.values, colLabels=dfm.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.2)
            
            ax_table.set_title("🏆 Leaderboard Metrics", fontsize=14, fontweight='bold')
            pdf.savefig(fig_lb)
            plt.close()

        # 3. MODEL DETAYLARI (Forecast, Diagnostics, XAI)
        for model_name, res in results.items():
            cached = cached_diagnostics.get(model_name, {})
            figs = cached.get("figs", {})
            metrics = cached.get("metrics", {})

            # --- A) FORECAST PAGE ---
            fig_fc_page = plt.figure(figsize=(11.69, 8.27))
            
            ax_head = fig_fc_page.add_subplot(211)
            ax_head.axis('off')
            ax_head.text(0.5, 0.5, f"Model: {model_name}\nForecast & Analysis", ha='center', va='center', fontsize=20, fontweight='bold', color='#1F618D')

            # Forecast Resmi
            ax_plot = fig_fc_page.add_subplot(212) 
            ax_plot.axis('off')
            
            if "Forecast" in figs:
                try:
                    img_bytes = pio.to_image(figs["Forecast"], format='png', scale=2)
                    img = mpimg.imread(BytesIO(img_bytes))
                    ax_plot.imshow(img)
                except Exception as e:
                    ax_plot.text(0.5, 0.5, f"Error plotting forecast: {e}", ha='center')
            
            pdf.savefig(fig_fc_page)
            plt.close()

            # --- B) DIAGNOSTIC & XAI PLOTS (Auto Detect Type) ---
            all_other_figs = [k for k in figs.keys() if k != "Forecast"]

            for fig_name in all_other_figs:
                fig_obj = figs[fig_name]
                
                fig_page = plt.figure(figsize=(11.69, 8.27))
                ax = fig_page.add_subplot(111)
                ax.axis('off')
                ax.set_title(f"{model_name} - {fig_name}", fontsize=14, fontweight='bold', pad=20)

                try:
                    if hasattr(fig_obj, 'savefig'):
                        buf_img = BytesIO()
                        fig_obj.savefig(buf_img, format='png', bbox_inches='tight')
                        buf_img.seek(0)
                        img = mpimg.imread(buf_img)
                        ax.imshow(img)
                    
                    elif hasattr(fig_obj, 'write_image') or (isinstance(fig_obj, dict) and 'data' in fig_obj):
                        img_bytes = pio.to_image(fig_obj, format='png', scale=2)
                        img = mpimg.imread(BytesIO(img_bytes))
                        ax.imshow(img)
                    
                    else:
                        ax.text(0.5, 0.5, "Bu grafik formatı (HTML/JS) PDF raporunda desteklenmemektedir.", ha='center', color='gray')

                except Exception as e:
                    ax.text(0.5, 0.5, f"Görsel oluşturma hatası:\n{str(e)}", ha='center', color='red')

                pdf.savefig(fig_page)
                plt.close(fig_page)

            # --- C) METRIC TABLES ---
            for metric_name, df_metric in metrics.items():
                if isinstance(df_metric, pd.DataFrame):
                    fig_table = plt.figure(figsize=(11.69, 8.27))
                    ax = fig_table.add_subplot(111)
                    ax.axis('off')
                    
                    row_count = len(df_metric)
                    font_size = 10 if row_count < 20 else 6
                    
                    cell_text = df_metric.astype(str).values
                    
                    table = ax.table(cellText=cell_text, colLabels=df_metric.columns, loc='center', cellLoc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(font_size)
                    
                    ax.set_title(f"{model_name} - {metric_name}", fontsize=14, fontweight='bold')
                    pdf.savefig(fig_table)
                    plt.close()

    buf.seek(0)
    return buf.read()

# -------------------------------------------------------------------------
# SIDEBAR: CONFIG BASED INTERFACE
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
st.sidebar.header("1️⃣ Data & Preparation")

with st.sidebar.expander("📂 Upload & Select Columns", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload Data File", 
        type=config.DATA_CONFIG["supported_extensions"],
        help="""
        **Supported File Formats:**
        
        * **📄 CSV / TXT:** Comma or tab-separated value files. Standard for tabular data.
        * **📊 XLSX:** Microsoft Excel files. The first sheet will be read.
        * **🧮 MAT:** MATLAB data files. Must contain 2D matrices/arrays.
        
        _Ensure your file contains clean, tabular data._
        """
    )

    df_raw = None
    feature_cols, target_col, date_col = [], None, None
    outlier_ops, scaler_ops = [], []

    if uploaded_file:
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            cols = df_raw.columns.tolist()
                
            date_col = st.selectbox(
                    "📅 Date Column (Optional)", 
                    [None] + cols,
                    help="Select the column containing date/time information. This is required for Time Series Cross-Validation and trend plotting. If data is not time-dependent, leave as 'None'."
            )
                
            target_col = st.selectbox(
                    "🎯 Target Column", 
                    cols, 
                    index=len(cols)-1,
                    help="The dependent variable (Label/Y) you want to predict. For regression tasks, ensure this column contains continuous numerical values."
            )
                
            feature_cols = st.multiselect(
                    "🧩 Select Features", 
                    [c for c in cols if c != target_col], 
                    default=None,
                    help="The independent variables (Features/X) used to train the model. Select all relevant columns that influence the target."
            )
        else:
            st.error("❌ Failed to read the file.")

# --- Data Preview ---
if df_raw is not None:
    file_size_kb = uploaded_file.size / 1024
    header_text = f"✅ {uploaded_file.name} ({file_size_kb:.1f} KB) | 📊 {df_raw.shape[0]} rows x {df_raw.shape[1]} cols | 👀 Preview"

    with st.expander(header_text, expanded=False):
        st.dataframe(df_raw.head(24), use_container_width=True)

    if feature_cols:
        with st.expander(f"🧬 Selected Features ({len(feature_cols)})", expanded=False):
            try:
                summary_df = pd.DataFrame({
                    "Type": df_raw[feature_cols].dtypes.astype(str),
                    "Unique": df_raw[feature_cols].nunique(),
                    "Missing": df_raw[feature_cols].isnull().sum()
                })
                st.dataframe(summary_df, use_container_width=True, height=220)
                st.caption(f"**Features:** {', '.join(feature_cols)}")
            except Exception as e:
                st.error(f"⚠️ Error creating feature summary: {e}")
    else:
        st.warning("⚠️ Please select at least one feature. Date column is optional.")
# -----------------------------
# --- SECTION 2: PREPROCESSING ---
train_btn = False
if df_raw is not None and feature_cols:
    df = preprocess_dataframe(df_raw, date_col)

    if "processing_log" not in st.session_state:
        st.session_state["processing_log"] = []

    st.sidebar.header("2️⃣ Preprocessing")

    with st.sidebar.expander("⚙️ Clean & Scale Data", expanded=False):

        st.markdown("### 🩹 Missing Value Imputation")
        # --- 1. FILLING METHOD ---
        filling_method = st.selectbox(
            "Select Filling Method",
            config.DATA_CONFIG.get("imputation_methods", []),
            index=None,                         
            placeholder="Select a filling method...",
            help="""
            **Imputation Strategy Guide:**
            * **Mean:** Fills missing values with the average. Best for normally distributed data without outliers. 
            [Image of normal vs skewed distribution for imputation]
            * **Median:** Fills with the middle value. More robust to outliers and skewed data than Mean.
            * **Mode:** Fills with the most frequent value. Ideal for categorical data or discrete numbers.
            * **Zero:** Fills with 0. Use only if 'missing' logically implies 'zero quantity' or 'absence'.
            """
        )
        # --- 2. OUTLIER HANDLING ---
        st.markdown("### 🧹 Outlier Handling")
        outlier_ops = st.multiselect(
            "Choose Outlier Treatment",
            config.DATA_CONFIG.get("outlier_methods", []),
            placeholder="Choose outlier treatments...", 
            help="""
            **Outlier Treatment Methods:**
            * **IQR Capping:** Caps values based on the Interquartile Range (25th-75th percentile). Best for skewed data; modifies extreme values to the nearest 'fence'. 
            [Image of boxplot IQR outlier detection]
            * **Z-Score Capping:** Caps values that are far from the mean (usually >3 std dev). Assumes data is normally distributed.
            * **Isolation Forest Drop:** Uses an anomaly detection algorithm to identify and **REMOVE** rows completely. Note: This reduces your dataset size.
            """
        )

        # --- 3. SCALING METHODS ---
        st.markdown("### 📏 Scaling Methods")
        scaling_ops = st.multiselect(
            "Select Scaling Method",
            config.DATA_CONFIG.get("scaling_methods", []),
            placeholder="Select scaling methods...", 
            help="""
            **Scaling Techniques:**
            * **Min-Max (0-1):** Scales data to a fixed range [0, 1]. Preserves distribution shape but sensitive to outliers. Good for Neural Networks.
            * **Standard (Z-Score):** Centers data around 0 with unit variance. Standard choice for SVM, Linear Regression, and KNN. 
            * **Robust:** Uses Median and IQR. The best choice if your data contains many outliers.
            * **MaxAbs:** Scales by dividing by the maximum absolute value. Preserves sparsity (0 remains 0).
            * **Log Transform:** Applies `log(1+x)` to compress large values. Essential for highly right-skewed data (e.g., income, prices).
            """
        )

        col_apply, col_reset = st.columns(2)

        with col_apply:
            apply_processing = st.button("Apply", type="primary", use_container_width=True)
        
        with col_reset:
            reset_processing = st.button("Reset", use_container_width=True)

        if apply_processing:
            old_missing = int(df.isna().sum().sum())
            old_shape = df.shape

            new_df = impute_missing(df, filling_method)
            # (Other processes can be added here)

            st.session_state["processed_df"] = new_df

            summary_entry = {
                "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                "Filling": filling_method,
                "Outlier": outlier_ops,
                "Scaling": scaling_ops,
                "Before Missing": int(old_missing),
                "After Missing": int(new_df.isna().sum().sum()),
                "Before Shape": old_shape,
                "After Shape": new_df.shape,
            }
            st.session_state["processing_log"].append(summary_entry)
            st.toast("Dataset processed!", icon="✅")
            time.sleep(0.5)
            st.rerun()

        if reset_processing:
            if "processed_df" in st.session_state:
                del st.session_state["processed_df"]
            st.session_state["processing_log"] = []
            st.session_state["results"] = None 
            st.toast("Data reset to original!", icon="🔙")
            time.sleep(0.5)
            st.rerun()

    # -----------------------------
    # --- SECTION 3: MODELING ---
    # -----------------------------
    st.sidebar.header("3️⃣ Modeling")

    with st.sidebar.expander("🧠 Models & Parameters", expanded=False):
        st.markdown("### 📚 Model Categories", help="""
        **Guide to Model Families:**
        
        * **🌲 Tree & Ensemble:** (e.g. XGBoost, RandomForest) Generally state-of-the-art for tabular data. Captures complex, non-linear patterns.
        * **📈 Linear & Regularized:** (e.g. Ridge, Lasso) Simple, fast, and interpretable. Great for establishing a baseline.
        * **🛡️ Bayesian & Robust:** (e.g. RANSAC, BayesianRidge) Excellent for small datasets or data containing many outliers.
        * **⭕ Support Vector Machines:** (e.g. SVR) Effective in high-dimensional spaces.
        * **🧠 Neural Networks & Others:** (e.g. MLP, KNN) Captures complex interactions (Neural Nets) or local patterns (Neighbors).
        """)
        
        st.info("Select models from different families below:")

        selected_models = []
        
        for group_name, models_in_group in config.MODEL_GROUPS.items():
            
            available_in_group = []
            for m in models_in_group:
                if m == "XGBoost" and (XGBRegressor is None): continue
                if m == "LightGBM" and (LGBMRegressor is None): continue
                if m == "CatBoost" and (CatBoostRegressor is None): continue
                available_in_group.append(m)
            
            if available_in_group:
                defaults = ["HistGradientBoosting"] if "HistGradientBoosting" in available_in_group else []
                
                st.markdown(f"**📌 {group_name}**")
                
                current_selection = st.multiselect(
                    f"Choose {group_name}", 
                    available_in_group, 
                    default=defaults,
                    key=f"multiselect_{group_name}", 
                    label_visibility="collapsed" 
                )
                selected_models.extend(current_selection)

        if not selected_models:
            st.warning("⚠️ Please select at least one model to proceed.")

        st.markdown("---")
        st.markdown("**⚙️ Advanced Settings**")
        hpo_ops = st.multiselect(
            "Hyperparameter Optimization", 
            config.AVAILABLE_HPO_METHODS,
            help="""
            **Method Comparison:**
            
            * **Random Search:** Selects random parameters from the space. Fast and generally provides good results.
            * **Grid Search:** Tries all possible combinations. The most comprehensive but slowest method.
            * **Optuna:** Uses the TPE (Tree-structured Parzen Estimator) algorithm. Performs smart search by learning from previous trials.
            * **Hyperband:** Uses Optuna infrastructure but stops poor-performing runs early (Pruning). Resource-friendly and fast.
            * **Bayesian Optimization:** Performs statistical optimization using Gaussian Process (GP).
            """
        )
        metric_ops = st.multiselect(
            "Evaluation Metrics", 
            config.METRICS_CONFIG["available"], 
            default=config.METRICS_CONFIG["defaults"],
            help="""
            **Metric Definitions:**
            
            * **MSE (Mean Squared Error):** Average squared difference between actual and predicted values. Heavily penalizes large errors.
            * **RMSE (Root Mean Squared Error):** Square root of MSE. In the same unit as the target variable; easy to interpret.
            * **MAE (Mean Absolute Error):** Average absolute difference. Less sensitive to outliers than MSE.
            * **R2 (R-Squared):** Proportion of variance explained by the model (0 to 1). Higher is better.
            * **Adj_R2 (Adjusted R2):** R2 adjusted for the number of predictors. Prevents overfitting illusion when adding features.
            * **MAPE (Mean Absolute Percentage Error):** Average percentage error. Easy to interpret for business stakeholders.
            * **MedAE (Median Absolute Error):** Median of all absolute errors. Very robust to outliers.
            * **MaxErr (Max Error):** The largest single error made by the model. Represents the worst-case prediction.
            * **ExpVar (Explained Variance):** Measures the proportion of variation in the target explained by the model.
            * **MSLE / RMSLE:** Logarithmic error metrics. Useful when target values span several orders of magnitude or you care about relative error.
            """
        )
# -------------------------------------------------------------------------
# --- SECTION 4: ANALYSIS & XAI ---
# -------------------------------------------------------------------------
    st.sidebar.header("4️⃣ Analysis & Outputs")
    with st.sidebar.expander("📊 Diagnostic Plots & XAI", expanded=False):
        # --- DIAGNOSTIC PLOTS ---
        selected_diag_plots = st.multiselect(
            "Diagnostic Plots", 
            config.DIAGNOSTIC_PLOTS, 
            default=config.DIAGNOSTIC_DEFAULTS,
            help="""
            **Diagnostic Guide:**
            * **📉 Overfitting Check:** Compares Train vs Test error. Large gap indicates memorization (Overfitting).
            * **🎯 Advanced Scatter:** Actual vs Predicted values. Points on the diagonal line are perfect predictions.
            * **〰️ Residuals:** Shows errors (Actual - Predicted). Random scatter is ideal; patterns indicate missed information.
            * **🔔 Distribution:** Histogram of residuals. Ideally, it should look like a bell curve (Normal Distribution). 
            [Image of normal vs skewed distribution for imputation]
            * **📈 QQ Plot:** Checks if errors follow a normal distribution. Points should hug the red line.
            * **💣 Influence:** Identifies data points that disproportionately 'pull' the model (High Leverage/Cook's D).
            * **🚨 Anomalies:** Uses Isolation Forest to highlight specific rows where the model failed significantly.
            """
        )
        # --- XAI METHODS ---
        xai_ops = st.multiselect(
            "Explainable AI (XAI)", 
            config.XAI_CONFIG["methods"], 
            help="""
            **XAI Method Guide:**
            * **🌈 SHAP:** The 'Gold Standard' based on Game Theory. Shows exactly how much each feature contributed to a specific prediction (Global & Local).
            * **🚀 PFI (Permutation Importance):** Shuffles a column and checks how much the error increases. Best for finding the most important features overall.
            * **💡 LIME:** Creates a simple, interpretable model around a *single* data point to explain local decisions.
            * **⚓ Anchor:** Explains decisions using high-precision "If-Then" rules (e.g., "IF Age < 30 AND Income > 50k THEN...").
            * **🔄 Counterfactual:** "What-if" analysis. Shows the minimal changes needed to get a different result (e.g., "Increase income by 5k to change prediction").
            """
        )
    train_btn = st.sidebar.button("🚀 Start Analysis", type="primary")

    # -------------------------------------------------------------------------
    # GLOBAL DATA PREPARATION
    # -------------------------------------------------------------------------
    active_df = None
    if df_raw is not None:
        active_df = st.session_state.get("processed_df", df)

    X, y = None, None
    if active_df is not None and feature_cols and target_col:
        missing_cols = [c for c in feature_cols if c not in active_df.columns]
        
        if not missing_cols and target_col in active_df.columns:
            X = active_df[feature_cols]
            y = active_df[target_col]
        else:
            st.error(f"⚠️ Column error: {missing_cols} not found in dataset.")

    # -------------------------------------------------------------------------
    # EXECUTION LOGIC (TRAINING)
    # -------------------------------------------------------------------------
    if "results" not in st.session_state:
        st.session_state["results"] = None

    if train_btn and X is not None and y is not None:
        
        st.session_state["results"] = {} # Reset results

        # --- GENERATE SCENARIOS ---
        scenarios = [("Raw", None, None)]

        # 2. Processed Data Scenario (If process selected or performed previously)
        is_processed_active = ("processed_df" in st.session_state) or outlier_ops or scaling_ops
        
        if is_processed_active:
            label_parts = []
            
            # A) Filling Info
            if "processed_df" in st.session_state and st.session_state.get("processing_log"):
                last_process = st.session_state["processing_log"][-1]
                if last_process.get("Filling"):
                    label_parts.append(f"Fill: {last_process['Filling']}")

            # B) Outlier Info
            if outlier_ops:
                short_outs = [o.split()[0] for o in outlier_ops]
                label_parts.append(f"Out: {'+'.join(short_outs)}")

            # C) Scaling Info
            if scaling_ops:
                short_scales = [s.split()[0] for s in scaling_ops]
                label_parts.append(f"Scl: {'+'.join(short_scales)}")

            # Create label
            final_label = " + ".join(label_parts) if label_parts else "Processed"
            scenarios.append((final_label, outlier_ops, scaling_ops))
        
        total_tasks = len(selected_models) * len(scenarios) * (1 + len(hpo_ops))
        temp_results = {}

        # --- MODERN UI: Status ---
        with st.status("🚀 AutoML Engine Running...", expanded=True) as status:
            log_container = st.container()
            current_task = 0
            prog_bar = st.progress(0)

            for model_name in selected_models:
                try:
                    if safe_model_factory(model_name) is None:
                        log_container.error(f"❌ Model '{model_name}' could not be loaded!")
                        continue
                except Exception as e:
                    log_container.error(f"❌ Model error: {e}")
                    continue

                for scen_label, scen_outliers, scen_scalers in scenarios:
                    
                    # --- 1. Default Run ---
                    display_key = f"{model_name} ({scen_label})"
                    
                    log_container.markdown(f"⚙️ **Training:** `{model_name}` | `{scen_label}`...")
                    
                    try:
                        metrics, model, Xt, Xte, ytr, yte, ytr_pr, ypr = train_and_evaluate(
                            X, y, 0.2, model_name, 
                            outlier_methods=scen_outliers,
                            scaling_methods=scen_scalers,
                            active_metrics=metric_ops
                        )
                        
                        temp_results[display_key] = {
                            'metrics': metrics, 
                            'model': model, 
                            'base_model': model_name,    
                            'proc_info': scen_label,     
                            'Xt': Xt, 'Xte': Xte, 'yte': yte, 'ypr': ypr, 'ytr': ytr, 'ytr_pr': ytr_pr
                        }
                        log_container.success(f"✅ Completed `{display_key}`")
                    except Exception as e:
                        log_container.error(f"❌ Error `{display_key}`: {e}")
                    
                    current_task += 1
                    prog_bar.progress(min(current_task / max(1, total_tasks), 1.0))
                    
                    # --- 2. HPO Runs ---
                    for hpo in hpo_ops:
                        hpo_key = f"{model_name} ({hpo} - {scen_label})"
                        status.write(f"🔧 **Optimizing:** `{hpo_key}`...")
                        try:
                            metrics_h, model_h, Xt_h, Xte_h, ytr_h, yte_h, ytr_pr_h, ypr_h = train_and_evaluate(
                                X, y, 0.2, model_name, hpo_method=hpo,
                                outlier_methods=scen_outliers, scaling_methods=scen_scalers,
                                active_metrics=metric_ops
                            )
                            temp_results[hpo_key] = {
                                'metrics': metrics_h, 
                                'model': model_h, 
                                'base_model': f"{model_name} ({hpo})", 
                                'proc_info': scen_label,        
                                'Xt': Xt_h, 'Xte': Xte_h, 'yte': yte_h, 'ypr': ypr_h, 'ytr': ytr_h, 'ytr_pr': ytr_h
                            }
                            log_container.success(f"✅ Completed `{hpo_key}`")
                        except Exception as e:
                            log_container.error(f"❌ HPO Error: {e}")
                        
                        current_task += 1
                        prog_bar.progress(min(current_task / max(1, total_tasks), 1.0))
            
            status.update(label="✅ All Models Trained!", state="complete", expanded=False)

        st.session_state["results"] = temp_results
        st.toast("Analysis complete!", icon="✅")
        time.sleep(1)
        st.rerun()

    # --- PREPROCESSING LOG VIEWER ---
    if "processing_log" in st.session_state and st.session_state["processing_log"]:
        st.subheader("🧩 Preprocessing Pipeline Viewer")
        for i, step in enumerate(reversed(st.session_state["processing_log"])):
            with st.expander(f"Step {len(st.session_state['processing_log']) - i} — {step['timestamp']}"):
                st.json(step)
                st.write(f"Missing: **{step['Before Missing']} → {step['After Missing']}**")

    # -------------------------------------------------------------------------
    # DASHBOARD DISPLAY Logic
    # -------------------------------------------------------------------------
    if st.session_state["results"]:
        try:
            run_dashboard(st.session_state["results"], selected_diag_plots, xai_ops)
        except NameError as e:
            st.error(f"⚠️ Dashboard variable error: {e}")
        except Exception as e:
            st.error(f"⚠️ Unexpected Dashboard error: {e}")