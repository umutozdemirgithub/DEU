
# config.py
import numpy as np
from scipy import stats
import os
# =============================================================================
# 1. APPLICATION & UI SETTINGS
# =============================================================================
APP_CONFIG = {
    "page_title": "Auto ML Model Diagnostic Dashboard (Optimized)",
    "page_layout": "wide",
    "sidebar_title": "AutoML Dashboard",
    "sidebar_icon": os.path.join(os.path.dirname(__file__), "logo.png"),
    "upload_help": "Drag & drop CSV, XLSX or MAT file.",
    "pdf_title": "AutoML Comprehensive Report"
}

# =============================================================================
# 2. DATA PROCESSING
# =============================================================================
DATA_CONFIG = {
    "random_state": 42,
    "supported_extensions": ["csv","xlsx","mat","txt"],
    "imputation_methods": ["Mean", "Median", "Mode", "Zero"],
    "outlier_methods": ["IQR Capping", "Z-Score Capping", "Isolation Forest Drop"],
    "scaling_methods": ["Min-Max Scaling (0-1)", "Standard Scaling (Z-Score)", "Robust Scaling (IQR based)", "MaxAbs Scaling (-1 to 1)", "Log Transformation (np.log1p)"]
}

# =============================================================================
# 3. MODELS AND DEFAULT PARAMETERS
# =============================================================================
MODEL_GROUPS = {
    "Tree & Ensemble (Boosting/Bagging)": [
        "HistGradientBoosting", 
        "RandomForest", 
        "GradientBoosting", 
        "XGBoost", 
        "LightGBM", 
        "CatBoost",
        "ExtraTrees",
        "AdaBoost"  
    ],
    "Linear & Regularized": [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGDRegressor" 
    ],
    "Bayesian & Robust": [
        "BayesianRidge",  
        "ARDRegression",   
        "HuberRegressor",
        "TheilSenRegressor", 
        "RANSACRegressor"    
    ],
    "Support Vector Machines": [
        "SVR",
        "LinearSVR", 
        "NuSVR"      
    ],
    "Neighbors & Gaussian": [
        "KNeighborsRegressor",
        "GaussianProcessRegressor" 
    ],
    "Neural Networks & Others": [
        "MLPRegressor",
        "DecisionTree"
    ]
}

AVAILABLE_MODELS = [model for group in MODEL_GROUPS.values() for model in group]

MODEL_DEFAULT_PARAMS = {
    "HistGradientBoosting": {
        "learning_rate": 0.05, "max_iter": 300, "max_depth": None, 
        "l2_regularization": 0.0, "min_samples_leaf": 30
    },
    "RandomForest": {
        "n_estimators": 300, "max_depth": None, "min_samples_split": 2, 
        "min_samples_leaf": 1, "max_features": "sqrt", "n_jobs": 2
    },
    "GradientBoosting": {
        "learning_rate": 0.05, "n_estimators": 200, "max_depth": 3, "subsample": 0.8
    },
    "XGBoost": {
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, 
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0, "verbosity": 0
    },
    "LightGBM": {
        "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, 
        "max_depth": -1, "subsample": 0.8, "reg_alpha": 0.0, "reg_lambda": 0.0, "verbose": -1
    },
    "CatBoost": {
        "iterations": 300, "learning_rate": 0.05, "depth": 6, 
        "l2_leaf_reg": 3.0, "subsample": 0.8, "verbose": 0, "allow_writing_files": False
    },
    "AdaBoost": {
        "n_estimators": 50, "learning_rate": 1.0, "loss": "linear", "random_state": 42
    },
    "SGDRegressor": {
        "loss": "squared_error", "penalty": "l2", "alpha": 0.0001, 
        "max_iter": 1000, "tol": 1e-3, "random_state": 42
    },
    "BayesianRidge": {
        "max_iter": 300, 
        "tol": 1e-3, 
        "alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6
    },
    "ARDRegression": {
        "max_iter": 300, 
        "tol": 1e-3, 
        "alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6
    },
    "TheilSenRegressor": {
        "max_subpopulation": 10000, "random_state": 42, "n_jobs": -1
    },
    "RANSACRegressor": {
        "min_samples": None, "residual_threshold": None, "random_state": 42
    },
    "LinearSVR": {
        "epsilon": 0.0, "tol": 1e-4, "C": 1.0, "loss": "epsilon_insensitive", "random_state": 42
    },
    "NuSVR": {
        "nu": 0.5, "C": 1.0, "kernel": "rbf", "gamma": "scale"
    },
    "GaussianProcessRegressor": {
        "alpha": 1e-10, "normalize_y": True, "random_state": 42
    },
    "LinearRegression": {
        "fit_intercept": True, "n_jobs": -1
    },
    "Ridge": {
        "alpha": 1.0, "solver": "auto", "random_state": 42
    },
    "Lasso": {
        "alpha": 1.0, "random_state": 42, "selection": "cyclic"
    },
    "ElasticNet": {
        "alpha": 1.0, "l1_ratio": 0.5, "random_state": 42
    },
    "DecisionTree": {
        "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42
    },
    "ExtraTrees": {
        "n_estimators": 300, "max_depth": None, "min_samples_split": 2, 
        "min_samples_leaf": 1, "max_features": "sqrt", "n_jobs": -1, "random_state": 42
    },
    "SVR": {
        "kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"
    },
    "KNeighborsRegressor": {
        "n_neighbors": 5, "weights": "uniform", "algorithm": "auto", "n_jobs": -1
    },
    "HuberRegressor": {
        "epsilon": 1.35, "max_iter": 100, "alpha": 0.0001
    },
    "MLPRegressor": {
        "hidden_layer_sizes": (100,), "activation": "relu", "solver": "adam", 
        "alpha": 0.0001, "learning_rate": "constant", "max_iter": 500, "random_state": 42
    }
}

UNIFIED_PARAM_NAMES = {
    "n_estimators": "Estimators / Iter",
    "max_iter": "Estimators / Iter",
    "iterations": "Estimators / Iter",
    "learning_rate": "Learning Rate",
    "max_depth": "Max Depth",
    "depth": "Max Depth",
    "l2_regularization": "L2 Regularization",
    "reg_lambda": "L2 Regularization",
    "l2_leaf_reg": "L2 Regularization",
    "reg_alpha": "L1 Regularization",
    "subsample": "Subsample Ratio",
    "max_features": "Col Sample / Max Feat",
    "colsample_bytree": "Col Sample / Max Feat",
    "min_samples_leaf": "Min Samples Leaf",
    "min_samples_split": "Min Samples Split",
    "num_leaves": "Max Leaves (LGBM)",
    
    "alpha": "Alpha (Reg Strength)",
    "l1_ratio": "L1 Ratio (ElasticNet)",
    "n_neighbors": "K Neighbors",
    "C": "Regularization (C)",
    "epsilon": "Epsilon (SVR/Huber)",
    "kernel": "Kernel Type",
    "hidden_layer_sizes": "Hidden Layers"
}

# =============================================================================
# 4. HPO (HYPERPARAMETER OPTIMIZATION) SPACES
# =============================================================================
AVAILABLE_HPO_METHODS = ["Random Search", "Grid Search", "Optuna", "Hyperband", "Bayesian Optimization","Artificial Bee Colony"]

HPO_SPACES = {
    "HistGradientBoosting": {
        "random": {
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__max_iter": stats.randint(100, 500),
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_leaf": stats.randint(20, 100),
            "model__l2_regularization": stats.loguniform(1e-9, 10),
        },
        "grid": {
            "model__learning_rate": [0.01, 0.1],
            "model__max_iter": [100, 300],
            "model__max_depth": [None, 5, 10],
            "model__l2_regularization": [0, 0.1, 1.0],
        }
    },
    "RandomForest": {
        "random": {
            "model__n_estimators": stats.randint(100, 400),
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": stats.randint(2, 15),
            "model__max_features": ["sqrt", "log2", None],
        },
        "grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, None],
            "model__min_samples_split": [2, 5],
        }
    },
    "GradientBoosting": {
        "random": {
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__n_estimators": stats.randint(100, 300),
            "model__max_depth": stats.randint(3, 8),
            "model__min_samples_split": stats.randint(2, 20),
            "model__subsample": stats.uniform(0.7, 0.3),
        },
        "grid": {
            "model__learning_rate": [0.01, 0.1],
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
        }
    },
    "XGBoost": {
        "random": {
            "model__n_estimators": stats.randint(100, 500),
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__max_depth": stats.randint(3, 10),
            "model__subsample": stats.uniform(0.6, 0.4),
            "model__colsample_bytree": stats.uniform(0.6, 0.4),
            "model__reg_alpha": stats.loguniform(1e-5, 10),
            "model__reg_lambda": stats.loguniform(1e-5, 10),
        },
        "grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 6],
        }
    },
    "AdaBoost": {
        "random": {
            "model__n_estimators": stats.randint(50, 500),
            "model__learning_rate": stats.loguniform(0.01, 2.0),
            "model__loss": ["linear", "square", "exponential"]
        },
        "grid": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.1, 1.0],
        }
    },
    "SGDRegressor": {
        "random": {
            "model__alpha": stats.loguniform(1e-5, 1e-1),
            "model__penalty": ["l2", "l1", "elasticnet"],
            "model__learning_rate": ["constant", "optimal", "invscaling"]
        },
        "grid": {
            "model__alpha": [0.0001, 0.01],
            "model__penalty": ["l2", "elasticnet"]
        }
    },
    "BayesianRidge": {
        "random": {
            "model__alpha_1": stats.loguniform(1e-7, 1e-5),
            "model__lambda_1": stats.loguniform(1e-7, 1e-5)
        },
        "grid": {
            "model__alpha_1": [1e-6, 1e-5],
            "model__lambda_1": [1e-6, 1e-5]
        }
    },
    "LinearSVR": {
        "random": {
            "model__C": stats.loguniform(0.1, 100),
            "model__epsilon": stats.uniform(0, 1)
        },
        "grid": {
            "model__C": [0.1, 1.0, 10.0],
            "model__epsilon": [0, 0.1]
        }
    },
    "GaussianProcessRegressor": {
        "random": {
            "model__alpha": stats.loguniform(1e-10, 1e-2)
        },
        "grid": {
            "model__alpha": [1e-10, 1e-5, 1e-2]
        }
    },
    "LightGBM": {
        "random": {
            "model__n_estimators": stats.randint(100, 500),
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__num_leaves": stats.randint(20, 150),
            "model__max_depth": stats.randint(-1, 15),
            "model__reg_alpha": stats.uniform(0, 1),
            "model__reg_lambda": stats.uniform(0, 1),
            "model__subsample": stats.uniform(0.6, 0.4),
        },
        "grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.1],
            "model__num_leaves": [31, 63],
        }
    },
    "CatBoost": {
        "random": {
            "model__iterations": stats.randint(100, 500),
            "model__learning_rate": stats.loguniform(0.01, 0.3),
            "model__depth": stats.randint(4, 10),
            "model__l2_leaf_reg": stats.randint(1, 10),
            "model__subsample": stats.uniform(0.6, 0.4),
        },
        "grid": {
            "model__iterations": [200],
            "model__learning_rate": [0.03, 0.1],
            "model__depth": [6, 8],
        }
    },

    "LinearRegression": {
        "random": { "model__fit_intercept": [True, False] },
        "grid": { "model__fit_intercept": [True, False] }
    },
    "Ridge": {
        "random": { "model__alpha": stats.loguniform(0.1, 100.0) },
        "grid": { "model__alpha": [0.1, 1.0, 10.0, 100.0] }
    },
    "Lasso": {
        "random": { "model__alpha": stats.loguniform(0.001, 10.0) },
        "grid": { "model__alpha": [0.001, 0.01, 0.1, 1.0] }
    },
    "ElasticNet": {
        "random": {
            "model__alpha": stats.loguniform(0.001, 10.0),
            "model__l1_ratio": stats.uniform(0.1, 0.8)
        },
        "grid": {
            "model__alpha": [0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }
    },
    "DecisionTree": {
        "random": {
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": stats.randint(2, 20),
            "model__min_samples_leaf": stats.randint(1, 10)
        },
        "grid": {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5, 10]
        }
    },
    "ExtraTrees": {
        "random": {
            "model__n_estimators": stats.randint(100, 400),
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": stats.randint(2, 15),
            "model__max_features": ["sqrt", "log2", None],
        },
        "grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, None],
            "model__min_samples_split": [2, 5],
        }
    },
    "SVR": {
        "random": {
            "model__C": stats.loguniform(0.1, 100),
            "model__epsilon": stats.loguniform(0.01, 1.0),
            "model__kernel": ["linear", "rbf", "poly"]
        },
        "grid": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear"]
        }
    },
    "KNeighborsRegressor": {
        "random": {
            "model__n_neighbors": stats.randint(2, 30),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        },
        "grid": {
            "model__n_neighbors": [3, 5, 10, 20],
            "model__weights": ["uniform", "distance"]
        }
    },
    "HuberRegressor": {
        "random": {
            "model__epsilon": stats.uniform(1.0, 1.0), # Between 1.0 and 2.0
            "model__alpha": stats.loguniform(0.0001, 0.1)
        },
        "grid": {
            "model__epsilon": [1.1, 1.35, 1.5, 1.75],
            "model__alpha": [0.0001, 0.001, 0.01]
        }
    },
    "MLPRegressor": {
        "random": {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
            "model__alpha": stats.loguniform(0.0001, 0.1),
            "model__learning_rate_init": stats.loguniform(0.001, 0.1)
        },
        "grid": {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "model__alpha": [0.0001, 0.01],
            "model__activation": ["relu", "tanh"]
        }
    }
}

# =============================================================================
# 5. METRICS AND PLOTS
# =============================================================================
METRICS_CONFIG = {
    "available": [
        "MSE", 
        "RMSE", 
        "MAE", 
        "R2", 
        "Adj_R2",          
        "MAPE", 
        "MedAE", 
        "MaxErr",           
        "ExpVar",           
        "MSLE",             
        "RMSLE"             
    ],
    "defaults": ["MSE", "RMSE", "R2", "Adj_R2"]
}

DIAGNOSTIC_PLOTS = [
    "Advanced Scatter",
    "Residuals",
    "Distribution",
    "QQ Plot",
    "Influence",
    "Anomalies",
    "Overfitting Check",
    "Residual vs Actual",
    "Error Bands",
    "KDE Distribution",
    "Bubble Influence"
]

DIAGNOSTIC_DEFAULTS = ["Advanced Scatter", "Overfitting Check"]

# =============================================================================
# 6. XAI (EXPLAINABLE AI) SETTINGS
# =============================================================================
XAI_CONFIG = {
    "methods": ["SHAP", "PFI", "LIME", "Anchor", "Counterfactual"],
    "lime_sample_size": 50,
    "lime_num_features": 12,
    "pfi_repeats": 15,
    "shap_sample_limit": 300,
    "radar_top_k": 12
}