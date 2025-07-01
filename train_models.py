import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
from config import TrainingConfig
from sklearn.feature_selection import mutual_info_regression
import logging
from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from config import get_path  # ensure this is imported


def validate_training_arrays(X_train, X_test, y_train, y_test):
    """Validate shape and compatibility of NumPy arrays for training."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"X_train rows ({X_train.shape[0]}) != y_train length ({y_train.shape[0]})")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"X_test rows ({X_test.shape[0]}) != y_test length ({y_test.shape[0]})")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Feature dimension mismatch: X_train has {X_train.shape[1]} features, "
                         f"X_test has {X_test.shape[1]}")

def compute_feature_weights(df, target_col):
    """
    Computes feature weights using mutual information.

    Parameters:
    - df (pd.DataFrame): Input data with all features.
    - target_col (str): The target variable for weighting.

    Returns:
    - weights (dict): Normalized feature weights.
    """
    features = df.drop(columns=[target_col]).select_dtypes(include=['float64', 'int64'])
    target = df[target_col]

    mi_scores = mutual_info_regression(features, target)
    weights = {feature: score for feature, score in zip(features.columns, mi_scores)}
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    logging.info("\n=== Feature Weights (Normalized) ===")
    for k, v in normalized_weights.items():
        logging.info(f"{k}: {v:.4f}")

    return normalized_weights

def train_all_models(X_train, X_test, y_train, y_test, model_info=None):
    """Train multiple regression models and compute an ensemble. Returns predictions, metrics, and logging data."""
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective="reg:squarederror",
        ),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
    }

    predictions = {}
    performance = {}
    all_metrics = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        predictions[name] = preds
        performance[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}

        all_metrics.append({
            "Tech": model_info["tech"],
            "Target": model_info["target"],
            "Model": name,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae
        })

    # Ensemble
    ensemble_preds = np.mean(np.column_stack(list(predictions.values())), axis=1)
    r2 = r2_score(y_test, ensemble_preds)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    mae = mean_absolute_error(y_test, ensemble_preds)

    predictions["Ensemble"] = ensemble_preds
    performance["Ensemble"] = {"R2": r2, "RMSE": rmse, "MAE": mae}
    all_metrics.append({
        "Tech": model_info["tech"],
        "Target": model_info["target"],
        "Model": "Ensemble",
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    })

    return predictions, performance, all_metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    cfg = TrainingConfig.from_yaml(Path("config.yaml"))
    
    # Choose correct dataset based on KG toggle
    feature_suffix = "_kg" if cfg.use_kg_features else ""
    feature_file = f"results/ml_feature_dataset_multi_tech{feature_suffix}.csv"

    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"❌ Feature file not found: {feature_file}")
    
    df = pd.read_csv(feature_file)

    # Define net performance as PV - RC cooling penalty
    def cooling_penalty(rc):
        return 0.5 * rc  # You can tune this value
    
    df["Net_Performance"] = df["PV_Potential"] - cooling_penalty(df["RC_potential"])
    target_cols = ["PV_Potential", "RC_potential", "Net_Performance"]

    all_metrics = []  # ✅ Initialize outside the loops

    for tech in df["Tech"].unique():
        logging.info(f"\n🔍 Training models for technology: {tech}")
        df_tech = df[df["Tech"] == tech].copy()

        for target_col in target_cols:
            logging.info(f"🎯 Target: {target_col}")

            # Select features (exclude tech and target columns)
            feature_cols = [col for col in df_tech.columns if col not in ["Tech"] + target_cols]
            X = df_tech[feature_cols].values
            y = df_tech[target_col].values

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Feature importance
            df_train = pd.DataFrame(X_train, columns=feature_cols)
            df_train[target_col] = y_train
            compute_feature_weights(df_train, target_col=target_col)

            # Validate shapes
            validate_training_arrays(X_train, X_test, y_train, y_test)

            # Train models and collect metrics
            model_info = {"tech": tech, "target": target_col}
            predictions, performance, metrics = train_all_models(X_train, X_test, y_train, y_test, model_info=model_info)
            all_metrics.extend(metrics)

            # Log performance
            for model_name, m in performance.items():
                logging.info(f"{model_name}: R2 = {m['R2']:.3f}, RMSE = {m['RMSE']:.3f}, MAE = {m['MAE']:.3f}")

    # ✅ Save all metrics once after all techs and targets
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(get_path("results_path"), "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✅ Saved model evaluation metrics to {metrics_path}")
