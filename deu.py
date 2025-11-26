import streamlit as st
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, GradientBoostingRegressor, RandomForestRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn.inspection import permutation_importance
from scipy import stats
import shap
# from lime.lime_tabular import LimeTabularExplainer # Optional
from alibi.explainers import AnchorTabular
import dice_ml

import streamlit.components.v1 as components
import warnings
import statsmodels.api as sm
from scipy.stats import gaussian_kde, probplot
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import time
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor

import config

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
    
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=random_state, **defaults)
    elif name == "RandomForest":
        return RandomForestRegressor(random_state=random_state, **defaults)
    elif name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=random_state, **defaults)
    elif name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=random_state, **defaults)
    elif name == "XGBoost":
        if XGBRegressor is None: 
            st.warning("XGBoost not installed. pip install xgboost")
            return None  
        return XGBRegressor(random_state=random_state, **defaults)
    elif name == "LightGBM":
        if LGBMRegressor is None:
            st.warning("LightGBM not installed. pip install lightgbm")
            return None
        return LGBMRegressor(random_state=random_state, **defaults)
    elif name == "CatBoost":
        if CatBoostRegressor is None:
            st.warning("CatBoost not installed. pip install catboost")
            return None
        return CatBoostRegressor(random_state=random_state, **defaults)
    elif name == "LinearRegression":
        return LinearRegression(**defaults)
    elif name == "Ridge":
        return Ridge(random_state=random_state, **defaults)
    elif name == "Lasso":
        return Lasso(random_state=random_state, **defaults)
    elif name == "ElasticNet":
        return ElasticNet(random_state=random_state, **defaults)
    elif name == "HuberRegressor":
        return HuberRegressor(**defaults)
    elif name == "DecisionTree":
        return DecisionTreeRegressor(random_state=random_state, **defaults)
    elif name == "SVR":
        return SVR(**defaults)
    elif name == "KNeighborsRegressor":
        return KNeighborsRegressor(**defaults)
    elif name == "MLPRegressor":
        return MLPRegressor(random_state=random_state, **defaults)
    else:
        st.warning(f"Model '{name}' not found. Using HistGradientBoosting.")
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

    # 3) HPO Spaces from CONFIG
    model_spaces = config.HPO_SPACES.get(model_name, {})
    
    if method == "Random Search":
        selected_space = model_spaces.get("random", {})
    else:
        selected_space = model_spaces.get("grid", {})

    if not selected_space:
        return pipeline

    if model_name in config.MODEL_DEFAULT_PARAMS:
        allowed_keys = config.MODEL_DEFAULT_PARAMS[model_name].keys()
        allowed_pipeline_keys = {f"model__{p}" for p in allowed_keys}
        
        selected_space = {k: v for k, v in selected_space.items() if k in allowed_pipeline_keys}
    
    if not selected_space:
         return pipeline

    # 5) Search Execution
    if method == "Random Search":
        search_engine = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=selected_space,
            n_iter=15, 
            cv=cv_strategy,
            scoring="neg_mean_squared_error",
            random_state=config.DATA_CONFIG["random_state"],
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
    metrics = {}
    if active_metrics is None:
        active_metrics = config.METRICS_CONFIG["defaults"]

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
    active_plots=None
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

    for i, tab_name in enumerate(final_tabs):
        with tabs[i]:
            if tab_name == "Advanced Scatter": 
                df_chart = pd.DataFrame({'Actual': y_test_arr, 'Predicted': y_pred_arr})
                fig_adv = px.scatter(
                    df_chart, x='Actual', y='Predicted', 
                    trendline="ols", trendline_color_override="red",
                    marginal_x="histogram", marginal_y="histogram",
                    opacity=0.6, height=layout_opts['height'],
                    title="Actual vs Predicted"
                )
                st.plotly_chart(fig_adv, use_container_width=True, key=f"{key_prefix}_adv")

            elif tab_name == "Overfitting Check" and has_train:
                fig = go.Figure(go.Bar(
                    x=["Train", "Test"], y=[train_rmse, test_rmse],
                    marker_color=["#1f77b4", "#ff7f0e"],
                    text=[f"{train_rmse:.4f}", f"{test_rmse:.4f}"], textposition="auto"
                ))
                fig.update_layout(title="RMSE Comparison (Train vs Test)", **layout_opts)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_over")
                
                overfit_threshold = 0.2
                rmse_diff = test_rmse - train_rmse
                ratio = test_rmse / train_rmse if train_rmse > 0 else np.inf
                if ratio > 1 + overfit_threshold:
                    st.warning(f"⚠️ Potential overfitting! Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")
                else:
                    st.success(f"✅ No significant overfitting. Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")

            elif tab_name == "Residuals":
                fig_res = px.scatter(
                    x=y_pred_arr, y=resid,
                    color=anomaly_score, color_continuous_scale="Turbo",
                    labels={'x': 'Predicted', 'y': 'Residual'},
                    title="Residuals vs Predicted", height=layout_opts['height']
                )
                fig_res.add_hline(y=0, line_dash='dash', line_color='black')
                st.plotly_chart(fig_res, use_container_width=True, key=f"{key_prefix}_res")

            elif tab_name == "Distribution":
                fig_dist = px.histogram(resid, nbins=50, marginal="box", title="Residual Distribution", height=layout_opts['height'])
                st.plotly_chart(fig_dist, use_container_width=True, key=f"{key_prefix}_dist")

            elif tab_name == "QQ Plot":
                qq = probplot(resid, dist='norm')
                fig_qq = px.scatter(x=qq[0][0], y=qq[0][1], labels={'x': 'Theoretical', 'y': 'Observed'}, title="Q-Q Plot", height=layout_opts['height'])
                fig_qq.add_shape(type="line", x0=min(qq[0][0]), y0=min(qq[0][0]), x1=max(qq[0][0]), y1=max(qq[0][0]), line=dict(color="red"))
                st.plotly_chart(fig_qq, use_container_width=True, key=f"{key_prefix}_qq")

            elif tab_name == "Influence":
                fig_inf = px.scatter(
                    x=leverage, y=cooks_d, color=anomaly_score, color_continuous_scale="Viridis",
                    labels={'x': 'Leverage', 'y': "Cook's D"}, title="Influence Plot", height=layout_opts['height']
                )
                fig_inf.add_hline(y=4/len(resid), line_dash="dash", line_color="red")
                st.plotly_chart(fig_inf, use_container_width=True, key=f"{key_prefix}_inf")

            elif tab_name == "Anomalies":
                df_anom = pd.DataFrame({
                    "Actual": y_test_arr, "Predicted": y_pred_arr, 
                    "Residual": resid, "AnomalyScore": anomaly_score, "Label": anomaly_label
                }).sort_values("AnomalyScore", ascending=False)
                st.dataframe(df_anom.head(50), use_container_width=True, height=400)

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

def run_xai_analysis(model, X_train, X_test, y_test, methods, key_suffix,
                     lime_sample=config.XAI_CONFIG["lime_sample_size"], 
                     lime_num_features=config.XAI_CONFIG["lime_num_features"], 
                     radar_top_k=config.XAI_CONFIG["radar_top_k"]):
    if not methods:
        st.info("No XAI methods selected.")
        return

    xai_tabs = st.tabs(methods)
    is_pipeline = isinstance(model, Pipeline)
    estimator = model.named_steps['model'] if is_pipeline else model

    shap_imp_df = None
    pfi_imp_df = None

    # SHAP
    if 'SHAP' in methods and shap is not None:
        with xai_tabs[methods.index('SHAP')]:
            st.markdown('## 🌈 SHAP Global & Local Explanations')
            try:
                X_shap = X_test.iloc[:min(300,len(X_test))].copy()
                if is_pipeline and 'scaler' in model.named_steps:
                    try:
                        X_shap = pd.DataFrame(model.named_steps['scaler'].transform(X_shap),
                                              columns=X_shap.columns, index=X_shap.index)
                    except: pass

                try:
                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(X_shap)
                except:
                    explainer = shap.Explainer(estimator, X_shap)
                    shap_values = explainer(X_shap)

                shap_imp_df = shap_importance_df(shap_values, X_shap)

                shap_tabs = st.tabs(['🎯 Summary Plot', '📊 Feature Importance', '🔎 Instance Waterfall', '⚡ Force Plot'])

                # Summary Plot
                with shap_tabs[0]:
                    fig, ax = plt.subplots(figsize=(7,5))
                    shap.summary_plot(shap_values, X_shap, show=False)
                    st.pyplot(fig)

                # Feature Importance
                with shap_tabs[1]:
                    st.dataframe(shap_imp_df)
                    fig_imp = px.bar(shap_imp_df.head(20), x='Importance', y='Feature',
                                     orientation='h', title='Top SHAP Feature Importance')
                    fig_imp.update_layout(template='plotly_white', height=450)
                    st.plotly_chart(fig_imp, use_container_width=True, key=f"shap_imp_{key_suffix}")

                # Waterfall Plot
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
                    df_w['Contribution'] = df_w['SHAP'].cumsum() + float(
                        explainer.expected_value if not isinstance(explainer.expected_value,(list,np.ndarray))
                        else explainer.expected_value[0]
                    )
                    fig_w = px.bar(df_w, x='SHAP', y='Feature', orientation='h',
                                   title=f'SHAP Waterfall – Instance {idx_w}', color='SHAP')
                    fig_w.update_layout(template='plotly_white', height=450)
                    st.plotly_chart(fig_w, use_container_width=True, key=f"shap_waterfall_{key_suffix}")

                # ⚡ Force Plot
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
                    st_shap(force_plot, height=300)

            except Exception as e: st.error(f'SHAP Error: {e}')

    # -------------------- PFI --------------------
    if 'PFI' in methods:
        with xai_tabs[methods.index('PFI')]:
            st.markdown('## 🚀 Permutation Feature Importance (PFI)')
            try:
                r = permutation_importance(estimator, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
                idx = r.importances_mean.argsort()[::-1]
                pfi_imp_df = pd.DataFrame({'Feature': X_test.columns[idx], 'Importance': r.importances_mean[idx],
                                           'Std Dev': r.importances_std[idx]})
                st.dataframe(pfi_imp_df)
                fig_pfi = px.bar(pfi_imp_df.head(20), x='Importance', y='Feature',
                                 orientation='h', error_x='Std Dev', title='Top 20 PFI Features')
                fig_pfi.update_layout(template='plotly_white', height=450)
                st.plotly_chart(fig_pfi, use_container_width=True, key=f"pfi_{key_suffix}")
            except Exception as e: st.error(f'PFI Error: {e}')

    # -------------------- LIME --------------------
    if 'LIME' in methods:
        with xai_tabs[methods.index('LIME')]:
            st.markdown('## 💡 LIME Local Explanations (with aggregated table)')
            try:
                explainer_lime = lime_tabular.LimeTabularExplainer(
                    X_train.values, feature_names=X_train.columns.tolist(),
                    mode='regression', discretize_continuous=True, verbose=False
                )
                idx = st.number_input('Select Test Instance', 0, len(X_test)-1, 0, key=f'lime_idx_{key_suffix}')
                exp = explainer_lime.explain_instance(X_test.iloc[idx].values,
                                                      lambda x:model.predict(x),
                                                      num_features=lime_num_features)
                if components is not None:
                    components.html(exp.as_html(), height=450, scrolling=True)
                else:
                    st.write(exp.as_list())
            except Exception as e: st.error(f'LIME Error: {e}')

    # -------------------- Anchor Explanations --------------------
    if 'Anchor' in methods:
        with xai_tabs[methods.index('Anchor')]:
            st.markdown("## 🪝 Anchor Explanations (PRO Edition)")

            try:
                from alibi.explainers import AnchorTabular
            except ImportError:
                st.error("`alibi` library is missing. Install with `pip install alibi`.")
            else:
                try:
                    feature_names = X_train.columns.tolist()
                    X_train_np = X_train.values
                    X_test_np = X_test.values

                    predict_fn = lambda x: estimator.predict(x)

                    explainer_anchor = AnchorTabular(
                        predict_fn,
                        feature_names=feature_names
                    )
                    explainer_anchor.fit(X_train_np)

                    idx = st.number_input(
                        "Select Test Instance for Anchor",
                        0, len(X_test_np) - 1, 0,
                        key=f"anchor_idx_{key_suffix}"
                    )

                    exp = explainer_anchor.explain(X_test_np[int(idx)])
                    anchor_rules = list(exp.anchor)

                    st.success(f"Anchor Rules Found: {anchor_rules}")
                    # ============================================================
                    # SUB-TAB STRUCTURE
                    # ============================================================
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(
                        [
                            "📌 Anchor Rules",
                            "🌊 Precision & Coverage (Liquid Gauge)",
                            "📈 Local Prediction + 3D Heatmap",
                            "🌳 Decision Rule Tree",
                            "📡 Neighborhood Stats"
                        ]
                    )
                    # ============================================================
                    # TAB 1 — Anchor Rules (Treemap + Bar)
                    # ============================================================
                    with tab1:
                        df_anchor = pd.DataFrame({
                            "Rule": anchor_rules,
                            "Weight": [1.0] * len(anchor_rules)
                        })

                        st.markdown("### 📌 Anchor Rule Structure")

                        c1, c2 = st.columns(2)

                        with c1:
                            fig_tree = px.treemap(
                                df_anchor,
                                path=["Rule"],
                                values="Weight",
                                title="Anchor Rule Treemap"
                            )
                            fig_tree.update_layout(height=450)
                            st.plotly_chart(fig_tree, use_container_width=True)

                        with c2:
                            fig_bar = px.bar(
                                df_anchor,
                                x="Weight",
                                y="Rule",
                                orientation="h",
                                title="Anchor Rule Strength"
                            )
                            fig_bar.update_layout(template="plotly_white", height=450)
                            st.plotly_chart(fig_bar, use_container_width=True)
                    # ============================================================
                    # TAB 2 — Precision & Coverage Liquid Gauges
                    # ============================================================
                    with tab2:

                        st.markdown("### 🌊 Liquid Gauge Visualization")

                        def liquid_gauge(value, title):
                            fig = {
                                "data": [{
                                    "type": "pie",
                                    "values": [value, 1 - value],
                                    "labels": ["Value", ""],
                                    "hole": 0.7,
                                    "textinfo": "none"
                                }],
                                "layout": {
                                    "title": {"text": f"{title}: {value:.2f}", "x": 0.5},
                                    "showlegend": False,
                                }
                            }
                            return fig

                        col1, col2 = st.columns(2)

                        with col1:
                            st.plotly_chart(liquid_gauge(exp.precision, "Precision"), use_container_width=True)

                        with col2:
                            st.plotly_chart(liquid_gauge(exp.coverage, "Coverage"), use_container_width=True)
                    # ============================================================
                    # TAB 3 — Prediction + 3D Heatmap
                    # ============================================================
                    with tab3:
                        pred_val = float(estimator.predict(X_test_np[[int(idx)]]))
                        act_val = float(y_test.iloc[int(idx)])

                        st.markdown("### 📈 Prediction Overview")
                        c1, c2 = st.columns(2)
                        c1.metric("Predicted Value", f"{pred_val:.4f}")
                        c2.metric("Actual Value", f"{act_val:.4f}")

                        st.markdown("### 🔥 3D Feature Heatmap")

                        df_feat = pd.DataFrame(
                            X_test_np[int(idx)].reshape(1, -1),
                            columns=feature_names
                        )

                        fig3d = go.Figure(data=[go.Surface(
                            z=[df_feat.values[0]],
                            x=list(range(len(feature_names))),
                            y=[0] * len(feature_names)
                        )])

                        fig3d.update_layout(
                            title="3D Heatmap of Feature Values",
                            scene=dict(
                                xaxis_title="Feature Index",
                                yaxis_title="Instance",
                                zaxis_title="Value"
                            ),
                            height=550
                        )

                        st.plotly_chart(fig3d, use_container_width=True)
                    # ============================================================
                    # TAB 4 — Decision Rule Tree
                    # ============================================================
                    with tab4:
                        st.markdown("### 🌳 Decision Rule Tree (Anchor Logic)")

                        rule_tree_text = ""
                        for i, r in enumerate(anchor_rules):
                            rule_tree_text += f"{' ' * (i*3)}└── {r}\n"

                        st.code(rule_tree_text, language="text")
                    # ============================================================
                    # TAB 5 — Neighborhood Stats
                    # ============================================================
                    with tab5:
                        st.markdown("### 📡 Neighborhood Precision & Coverage")

                        df_nei = pd.DataFrame({
                            "Metric": ["Precision", "Coverage"],
                            "Value": [exp.precision, exp.coverage]
                        })

                        fig_ns = px.bar(
                            df_nei,
                            x="Metric",
                            y="Value",
                            text="Value",
                            title="Anchor Neighborhood Performance"
                        )
                        fig_ns.update_layout(template="plotly_white", height=450)
                        st.plotly_chart(fig_ns, use_container_width=True)

                except Exception as e:
                    st.error(f"Anchor Error: {e}")

    # -------------------- Counterfactual Explanations --------------------
    if 'Counterfactual' in methods:
        with xai_tabs[methods.index('Counterfactual')]:
            st.markdown("## 🔄 Counterfactual Explanations (Advanced XAI Panel)")

            try:

                # 1️⃣ Is y_train available? If not create outcome automatically
                if 'y_train' in globals():
                    df_dice = pd.concat([X_train.reset_index(drop=True),
                                        y_train.reset_index(drop=True)], axis=1)
                    outcome_col = y_train.name
                else:
                    outcome_col = "target"
                    df_dice = X_train.copy()
                    df_dice[outcome_col] = estimator.predict(X_train)

                # 2️⃣ Is model classification or regression?
                is_classification = False
                try:
                    _ = estimator.predict_proba(X_test[:5])
                    is_classification = True
                except:
                    is_classification = False

                # 3️⃣ DiCE Data wrapper
                data_dice = dice_ml.Data(
                    dataframe=df_dice,
                    continuous_features=X_train.columns.tolist(),
                    outcome_name=outcome_col
                )

                # 4️⃣ Model wrapper
                if is_classification:
                    model_dice = dice_ml.Model(model=estimator, backend="sklearn")
                else:
                    model_dice = dice_ml.Model(model=estimator,
                                            backend="sklearn",
                                            model_type="regressor")

                # 5️⃣ CF engine
                exp = dice_ml.Dice(data_dice, model_dice, method="random")

                # 6️⃣ Select instance
                idx = st.number_input(
                    "Select Test Instance",
                    0, len(X_test) - 1, 0,
                    key=f"cf_idx_{key_suffix}"
                )
                x0 = X_test.iloc[[idx]]
                y0 = estimator.predict(x0)[0]

                # 7️⃣ If Regression use desired_range
                if not is_classification:
                    delta = abs(y0) * 0.2 if y0 != 0 else 1
                    desired_range = [y0 - delta, y0 + delta]
                    cf = exp.generate_counterfactuals(
                        x0, total_CFs=3, desired_range=desired_range
                    )
                else:
                    cf = exp.generate_counterfactuals(
                        x0, total_CFs=3, desired_class="opposite"
                    )

                cf_df = cf.cf_examples_list[0].final_cfs_df
                original_df = x0.copy()
                original_df["type"] = "Original"
                cf_df["type"] = "CF"
                merged_df = pd.concat([original_df, cf_df], ignore_index=True)

                # 8️⃣ SUB-TABS: Tabular, Heatmap, Radar, Minimal Change, Prediction Plot
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📄 Tabular View",
                    "📊 Difference Heatmap",
                    "🧭 Radar Plot",
                    "🔧 Minimal Feature Change",
                    "📈 CF Prediction Plot"
                ])

                # 📄 Tab 1: Counterfactual Table
                with tab1:
                    st.subheader("📄 Counterfactual Table")
                    st.dataframe(cf_df)

                # 📊 Tab 2: Difference Heatmap
                with tab2:
                    st.subheader("📊 Feature Differences (Original vs CF)")
                    diff = cf_df[X_train.columns] - x0.values
                    diff_df = diff.T.rename(columns={0: "Difference"})
                    fig_diff = px.imshow(
                        diff_df,
                        color_continuous_scale="RdBu",
                        title="Original – Counterfactual Feature Differences",
                    )
                    st.plotly_chart(fig_diff, use_container_width=True)

                # 🧭 Tab 3: Radar Plot
                with tab3:
                    st.subheader("🧭 Feature Change Radar Chart")
                    feat = X_train.columns.tolist()
                    base_vals = x0.values.flatten()
                    cf_vals = cf_df.iloc[0][feat].values
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=base_vals,
                        theta=feat,
                        fill='toself',
                        name='Original'
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=cf_vals,
                        theta=feat,
                        fill='toself',
                        name='Counterfactual'
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        title="Original vs Counterfactual Radar Comparison"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                # 🔧 Tab 4: Minimal Feature Change
                with tab4:
                    st.subheader("🔧 Minimal Feature Changes")
                    change_df = pd.DataFrame({
                        "Feature": feat,
                        "Original": base_vals,
                        "Counterfactual": cf_vals,
                        "Difference": cf_vals - base_vals
                    }).sort_values(by="Difference", key=abs, ascending=False)
                    st.dataframe(change_df)

                # 📈 Tab 5: CF Prediction Plot
                with tab5:
                    st.subheader("📈 Original vs CF Prediction")
                    y_orig = estimator.predict(x0)
                    y_cf = estimator.predict(cf_df[feat])
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Bar(name='Original', x=[0], y=[y_orig[0]]))
                    fig_pred.add_trace(go.Bar(name='Counterfactual', x=[0], y=[y_cf[0]]))
                    fig_pred.update_layout(title="Predicted Values Comparison", showlegend=True)
                    st.plotly_chart(fig_pred, use_container_width=True)

            except Exception as e:
                st.error(f"Counterfactual Error: {e}")

    # -------------------- Radar Chart --------------------
    if ('PFI' in methods or 'SHAP' in methods):
        try:
            with st.expander('📡 Combined Radar'):
                fig_radar = create_combined_radar(pfi_imp_df if pfi_imp_df is not None else pd.DataFrame({'Feature':[], 'Importance':[]}), shap_imp_df, top_k=radar_top_k)
                st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{key_suffix}")
        except Exception as e: st.error(f'Radar Error: {e}')

    return {'pfi': pfi_imp_df, 'shap': shap_imp_df}

def run_dashboard(results, selected_diag_plots, xai_ops):
    if not results:
        st.info("👈 Upload data & Start Training.")
        return
    
    with st.expander("### 🏆 Leaderboard", expanded=False):
        lb_tabs = st.tabs(["📋 Table", "📈 Chart", "⚙️ Parameters"])
        
# --- TABLE GENERATION ---
        table_data = []
        for k, v in results.items():
            row = v['metrics'].copy()
            row['Model'] = v.get('base_model', k) 
            row['Processing'] = v.get('proc_info', '-')
            table_data.append(row)
            
        df_res = pd.DataFrame(table_data)
        
        cols = ['Model', 'Processing'] + [c for c in df_res.columns if c not in ['Model', 'Processing']]
        df_res = df_res[cols]
        
        with lb_tabs[0]:
            st.dataframe(df_res.style.format(precision=4).highlight_min(subset=["MSE","RMSE"], color='lightblue').highlight_max(subset=["R2"], color='red'), use_container_width=True)

        # --- Tab 2: Chart  ---
        with lb_tabs[1]:
            df_res = pd.DataFrame([{**{'Model': k}, **v['metrics']} for k, v in results.items()])

            fig = go.Figure()

            metrics_to_plot = [col for col in df_res.columns if col != 'Model' and col != 'R2']

            if not metrics_to_plot and "R2" not in df_res.columns:
                st.warning("No calculated metrics found to plot.")
            else:

                for metric in metrics_to_plot:
                    fig.add_trace(go.Bar(
                        x=df_res['Model'],
                        y=df_res[metric],
                        name=metric,
                        text=[f"{v:.4f}" for v in df_res[metric]],
                        textposition="outside",
                        insidetextanchor="middle",
                        hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>",
                        yaxis="y1"
                    ))

                if "R2" in df_res.columns:
                    min_size = 12
                    max_size = 30
                    r2_min = df_res["R2"].min()
                    r2_max = df_res["R2"].max()
                    if r2_max != r2_min:
                        sizes = min_size + (df_res["R2"] - r2_min) / (r2_max - r2_min) * (max_size - min_size)
                    else:
                        sizes = [(min_size + max_size)/2] * len(df_res)

                    fig.add_trace(go.Scatter(
                        x=df_res['Model'],
                        y=df_res["R2"],
                        mode='markers+lines',
                        marker=dict(
                            symbol='star',
                            size=sizes,
                            color='green',  
                            line=dict(width=1, color='black')
                        ),
                        line=dict(color='green', dash='dash', width=2),
                        name="R2 Trend",
                        hovertemplate="<b>%{x}</b><br>R2: %{y:.6f}<extra></extra>",
                        yaxis="y2"
                    ))

            # Layout settings
            fig.update_layout(
                title="Leaderboard Metrics Comparison (Dual Y-Axis & R² Size)",
                xaxis=dict(title="Model", tickangle=-45),
                yaxis=dict(title="Metric Value", side="left"),
                yaxis2=dict(title="R2 Value", overlaying="y", side="right"),
                barmode='group',
                bargap=0.25,
                bargroupgap=0.1,
                template="plotly_white",
                height=550,
                legend=dict(title="Metrics"),
                hovermode="x unified"
            )

            fig.update_traces(textfont_size=10)

            st.plotly_chart(fig, use_container_width=True, key="leaderboard_chart_r2size")

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
                            row[disp_name] = params[k]
                param_list.append(row)
            
            if param_list:
                st.dataframe(pd.DataFrame(param_list).fillna("-"), use_container_width=True)

    with st.expander("### 📊 Visualization"):
        model_tabs = st.tabs(list(results.keys()))
        for i, model_name in enumerate(results.keys()):
            res = results[model_name]
            with model_tabs[i]:
                ftab, dtab, xtab = st.tabs(["📉 Forecast", "🔍 Diagnostics", "🧠 XAI"])
                with ftab:
                    limit = min(200, len(res['yte']))
                    df_viz = pd.DataFrame({'Actual': res['yte'].iloc[:limit], 'Predicted': res['ypr'][:limit]})
                    st.plotly_chart(px.line(df_viz, markers=True, title=f"{model_name} Forecast"), use_container_width=True, key=f"{model_name}_fc")
                with dtab:
                    display_diagnostic_plots(res['yte'], res['ypr'], res.get('ytr'), res.get('ytr_pr'), key_prefix=model_name, active_plots=selected_diag_plots)
                with xtab:
                    run_xai_analysis(res['model'], res['Xt'], res['Xte'], res['yte'], xai_ops, key_suffix=model_name)

    with st.expander("### 📥 Exports"):
        c1, c2 = st.columns(2)
        c1.download_button("Download Metrics (CSV)", df_res.to_csv(index=False).encode('utf-8'), "metrics.csv", "text/csv")
        if c2.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf_data = generate_pdf_report(results, selected_diag_plots, xai_ops)
                    st.download_button("Download PDF", pdf_data, "AutoML_Report.pdf", "application/pdf")
                except Exception as e: st.error(f"PDF Error: {e}")

def generate_pdf_report(results, active_diag_plots, active_xai_methods):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.6, config.APP_CONFIG["pdf_title"], ha='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.5, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ha='center', fontsize=14)
        pdf.savefig(fig); plt.close()

        if results:
            dfm = pd.DataFrame([{**{'Model':k}, **v['metrics']} for k,v in results.items()])
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('tight'); ax.axis('off')
            table = ax.table(cellText=dfm.round(4).values, colLabels=dfm.columns, loc='center')
            table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.2)
            ax.set_title("Leaderboard", fontweight='bold')
            pdf.savefig(fig); plt.close()

        for model_name, res in results.items():
            fig = plt.figure(figsize=(8.5, 11))
            gs = fig.add_gridspec(3, 1)
            ax_title = fig.add_subplot(gs[0, :]); ax_title.axis('off')
            ax_title.text(0.5, 0.5, f"Model: {model_name}", ha='center', fontsize=18, fontweight='bold')
            ax_fc = fig.add_subplot(gs[1:, :])
            limit = min(150, len(res['yte']))
            ax_fc.plot(res['yte'].values[:limit], label='Actual', color='black', alpha=0.7)
            ax_fc.plot(res['ypr'][:limit], label='Predicted', color='red', linestyle='--', alpha=0.7)
            ax_fc.legend()
            pdf.savefig(fig); plt.close()

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
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], help="Drag & drop your CSV file here.")
    df_raw = None
    feature_cols, target_col, date_col = [], None, None
    outlier_ops, scaler_ops = [], []

    if uploaded_file:
        df_raw = load_data(uploaded_file)
        if df_raw is not None:
            cols = df_raw.columns.tolist()
            date_col = st.selectbox("📅 Date Column (Optional)", [None] + cols)
            target_col = st.selectbox("🎯 Target Column", cols, index=len(cols)-1)
            feature_cols = st.multiselect("🧩 Select Features", [c for c in cols if c != target_col], default=None)
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
        filling_method = st.selectbox(
            "Select Filling Method",
            config.DATA_CONFIG.get("imputation_methods", []),
            index=None,                         
            placeholder="Select a filling method...",
            help="Mean, Median, Mode or Zero"
        )

        st.markdown("### 🧹 Outlier Handling")
        outlier_ops = st.multiselect(
            "Choose Outlier Treatment",
            config.DATA_CONFIG.get("outlier_methods", []),
            placeholder="Choose outlier treatments...", 
            help="IQR Capping, Z-Score Capping or Isolation Forest Drop"
        )

        st.markdown("### 📏 Scaling Methods")
        scaling_ops = st.multiselect(
            "Select Scaling Method",
            config.DATA_CONFIG.get("scaling_methods", []),
            placeholder="Select scaling methods...", 
            help="MinMax, Standard, Robust, MaxAbs, Log Transformation"
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
        avail_models = ["HistGradientBoosting", "RandomForest", "GradientBoosting"]
        if 'XGBRegressor' in globals(): avail_models.append("XGBoost")
        if 'LGBMRegressor' in globals(): avail_models.append("LightGBM")
        if 'CatBoostRegressor' in globals(): avail_models.append("CatBoost")
        
        selected_models = st.multiselect(
            "Select Models", 
            config.AVAILABLE_MODELS, 
            default=["HistGradientBoosting"],
            help="Pick one or more models to train."
        )

        st.markdown("**⚙️ Advanced Settings**")
        hpo_ops = st.multiselect(
            "Hyperparameter Optimization", 
            ["Random Search","Grid Search"], 
            help="Automatically find the best parameters."
        )
        metric_ops = st.multiselect(
            "Evaluation Metrics", 
            config.METRICS_CONFIG["available"], 
            default=config.METRICS_CONFIG["defaults"],
            help="Select metrics to evaluate model performance."
        )
# -----------------------------
# --- SECTION 4: ANALYSIS & XAI ---
# -----------------------------
    st.sidebar.header("4️⃣ Analysis & Outputs")

    with st.sidebar.expander("📊 Diagnostic Plots & XAI", expanded=False):
        selected_diag_plots = st.multiselect(
            "Diagnostic Plots", 
            config.DIAGNOSTIC_PLOTS, default=config.DIAGNOSTIC_DEFAULTS,
            help="Visualize model performance."
        )
        xai_ops = st.multiselect(
            "Explainable AI (XAI)", 
            config.XAI_CONFIG["methods"], 
            help="Understand why the model makes its predictions."
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
