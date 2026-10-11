#!/usr/bin/env python3
"""
Script de Entrenamiento Parametrizado con Seguimiento en MLflow
Módulo: train.py
Proyecto: TC5061 - Experimentación y MLOps
Dataset: Breast Cancer Wisconsin (Diagnostic)
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import pandas as pd

# Configurar variables de entorno antes de importar matplotlib y mlflow
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import mlflow
import mlflow.sklearn


def parse_arguments():
    """
    Configura y analiza los argumentos de línea de comandos con argparse.
    Expone al menos tres hiperparámetros del modelo, la semilla y opciones de ejecución.
    """
    parser = argparse.ArgumentParser(
        description="Script modular de entrenamiento para clasificación con seguimiento en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Hiperparámetros del modelo
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización. Valores menores implican mayor penalización."
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear", "saga", "newton-cg"],
        help="Algoritmo de optimización para el ajuste de la regresión logística."
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones para que el solver converja."
    )

    # Semilla y partición
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para la partición de datos y reproducibilidad del modelo."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Proporción del conjunto de datos destinado a evaluación (test)."
    )

    # Configuración de MLflow y rutas
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/breast_cancer.csv",
        help="Ruta relativa o absoluta al archivo CSV con los datos."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Optimizacion_Cancer_Mama_Hiperparametros",
        help="Nombre descriptivo del experimento en MLflow Tracking."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Nombre identificador de la corrida en MLflow (si es None, se autogenera)."
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="URI del backend de seguimiento de MLflow."
    )

    return parser.parse_args()


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV y valida su existencia.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: No se encontró el dataset en la ruta especificada: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[Carga] Datos cargados exitosamente desde '{filepath}'. Dimensiones iniciales: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame, test_size: float, seed: int):
    """
    Aplica limpieza estandarizada, eliminación de columnas no predictivas,
    codificación del target y partición estratificada con escalado robusto.
    """
    df_clean = df.copy()

    # 1. Tratamiento de duplicados
    n_dups = df_clean.duplicated().sum()
    if n_dups > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"[Limpieza] Se eliminaron {n_dups} registros duplicados.")

    # 2. Imputación de valores nulos con la mediana
    cols_nulos = df_clean.columns[df_clean.isnull().any()].tolist()
    if cols_nulos:
        for col in cols_nulos:
            mediana = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(mediana)
        print(f"[Limpieza] Imputados valores nulos en columnas {cols_nulos} usando la mediana.")

    # 3. Eliminar ID no predictivo
    if "id" in df_clean.columns:
        df_clean = df_clean.drop(columns=["id"])

    # 4. Codificación de la variable objetivo (M=1, B=0)
    if "diagnosis" in df_clean.columns:
        if df_clean["diagnosis"].dtype != int:
            df_clean["diagnosis"] = df_clean["diagnosis"].map({"M": 1, "B": 0}).astype(int)

    X = df_clean.drop(columns=["diagnosis"])
    y = df_clean["diagnosis"]

    # 5. División estratificada con semilla fija
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # 6. Estandarización sin fuga de datos (ajuste solo en Train)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    print(f"[Preprocesamiento] Train: {X_train_scaled.shape[0]} muestras | Test: {X_test_scaled.shape[0]} muestras.")
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


def train_model(X_train: pd.DataFrame, y_train: pd.Series, C: float, solver: str, max_iter: int, seed: int):
    """
    Instancia y ajusta el modelo base de Regresión Logística con los hiperparámetros indicados.
    """
    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=seed
    )
    model.fit(X_train, y_train)
    print(f"[Entrenamiento] Modelo LogisticRegression entrenado (C={C}, solver='{solver}', max_iter={max_iter}, seed={seed}).")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Calcula predicciones y genera métricas clínicas clave (Accuracy, Recall, Precision, Specificity, F1, ROC-AUC).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    print("-" * 65)
    print("MÉTRICAS DE EVALUACIÓN:")
    print(f"  Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Recall:      {rec:.4f} ({rec*100:.2f}%)  <-- Métrica clínica clave")
    print(f"  Precision:   {prec:.4f} ({prec*100:.2f}%)")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  Matriz Conf: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("-" * 65)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return metrics, cm, (fpr, tpr, auc), y_pred


def generate_artifacts(cm, roc_data, y_test, y_pred, output_dir: str):
    """
    Crea y guarda los gráficos y reportes textuales como artefactos reproducibles.
    """
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, auc = roc_data

    # 1. Matriz de confusión
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Matriz de Confusión", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks, ["Benigno (0)", "Maligno (1)"])
    ax.set_yticks(tick_marks, ["Benigno (0)", "Maligno (1)"])
    thresh = cm.max() / 2.0
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({labels[i][j]})",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    ax.set_ylabel("Etiqueta Real", fontsize=10)
    ax.set_xlabel("Predicción", fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=140)
    plt.close()

    # 2. Curva ROC
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2.2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Azar")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Especificidad (FPR)", fontsize=10)
    plt.ylabel("Sensibilidad / Recall (TPR)", fontsize=10)
    plt.title("Curva ROC", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=140)
    plt.close()

    # 3. Reporte de clasificación en texto
    report_path = os.path.join(output_dir, "classification_report.txt")
    rep_text = classification_report(y_test, y_pred, target_names=["Benigno (0)", "Maligno (1)"])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REPORTE DETALLADO DE CLASIFICACIÓN\n")
        f.write("=" * 50 + "\n")
        f.write(rep_text)

    return [cm_path, roc_path, report_path]


def log_to_mlflow(args, metrics, model, artifact_paths, X_test):
    """
    Registra parámetros, métricas, artefactos visuales y el modelo entrenado en MLflow Tracking.
    """
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or f"LogReg_C{args.C}_solv_{args.solver}_seed{args.seed}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[MLflow] Registrando corrida '{run_name}' (ID: {run_id}) en experimento '{args.experiment_name}'...")

        # 1. Registrar Parámetros
        mlflow.log_params({
            "C": float(args.C),
            "solver": str(args.solver),
            "max_iter": int(args.max_iter),
            "seed": int(args.seed),
            "test_size": float(args.test_size),
            "model_type": "LogisticRegression",
            "scaler": "StandardScaler",
            "dataset": os.path.basename(args.data_path)
        })

        # 2. Registrar Métricas
        mlflow.log_metrics(metrics)

        # 3. Registrar Artefactos
        for art_path in artifact_paths:
            mlflow.log_artifact(art_path, artifact_path="evaluacion")

        # 4. Registrar Modelo con Firma de Inferencia
        input_example = X_test.head(5)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo_logistic_regression",
            input_example=input_example
        )

        print(f"[MLflow] Corrida completada y registrada exitosamente.")
        return run_id


def main():
    args = parse_arguments()
    print("=" * 70)
    print("EJECUCIÓN DEL SCRIPT PARAMETRIZADO: train.py")
    print("=" * 70)

    # 1. Carga
    df = load_data(args.data_path)

    # 2. Preprocesamiento
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        df, test_size=args.test_size, seed=args.seed
    )

    # 3. Entrenamiento
    model = train_model(
        X_train, y_train,
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        seed=args.seed
    )

    # 4. Evaluación
    metrics, cm, roc_data, y_pred = evaluate_model(model, X_test, y_test)

    # 5. Generación de Artefactos Temporales
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_paths = generate_artifacts(cm, roc_data, y_test, y_pred, tmp_dir)
        
        # 6. Registro en MLflow
        run_id = log_to_mlflow(args, metrics, model, artifact_paths, X_test)

    print("=" * 70)
    print(f"Éxito: Corrida registrada en MLflow con ID: {run_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
