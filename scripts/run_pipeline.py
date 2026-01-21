#!/usr/bin/env python3
"""
End-to-end Telco Churn training pipeline with MLflow
"""

import os
import sys
import time
import argparse
import json
import joblib
import pandas as pd

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from xgboost import XGBClassifier

# =====================================================
# Project path setup
# =====================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Local imports
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


def main(args):

    # =====================================================
    # MLflow setup
    # =====================================================
    mlruns_path = args.mlflow_uri or os.path.join(PROJECT_ROOT, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():

        # -------------------------------------------------
        # Log config params
        # -------------------------------------------------
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # =================================================
        # Load + validate data
        # =================================================
        print("Loading data...")
        df = load_data(args.input)

        print("Validating data...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), "failed_expectations.json")
            raise RuntimeError("Data validation failed")

        # =================================================
        # Preprocess
        # =================================================
        df = preprocess_data(df)

        # =================================================
        # Feature engineering
        # =================================================
        target = args.target
        df_enc = build_features(df, target_col=target)

        for col in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[col] = df_enc[col].astype(int)

        feature_cols = list(df_enc.drop(columns=[target]).columns)

        # =================================================
        # Save preprocessing artifacts
        # =================================================
        artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        preprocessing_path = os.path.join(artifacts_dir, "preprocessing.pkl")
        joblib.dump(
            {"feature_columns": feature_cols, "target": target},
            preprocessing_path
        )

        mlflow.log_artifact(preprocessing_path)
        mlflow.log_text("\n".join(feature_cols), "feature_columns.txt")

        # =================================================
        # Train / test split
        # =================================================
        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )

        # =================================================
        # Train model
        # =================================================
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.95,
            colsample_bytree=0.98,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        mlflow.log_metric("train_time", time.time() - start_time)

        # =================================================
        # Evaluate
        # =================================================
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)

        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))

        print(classification_report(y_test, y_pred, digits=3))

        # =================================================
        # Save model as artifact (pickle)
        # =================================================
        model_path = os.path.join(artifacts_dir, "xgboost_model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # =================================================
        # Log model to MLflow (deployable format)
        # =================================================
        print("Logging model to MLflow...")

        input_example = X_train.iloc[:5]
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.xgboost.log_model(
            model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        print("Model successfully logged to MLflow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--experiment", default="Telco Churn")
    parser.add_argument("--mlflow_uri", default=None)

    args = parser.parse_args()
    main(args)
