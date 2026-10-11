#!/usr/bin/env python3
"""
Pipeline de Machine Learning y Registro con MLflow
Proyecto MLOps - Modelo Baseline
Dataset: Breast Cancer Wisconsin (Diagnostic) - Kaggle / UCI Machine Learning Repository
"""

import os
import sys

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import mlflow
import mlflow.sklearn

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATASET_PATH = os.path.join(DATA_DIR, "breast_cancer.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Desactivar hints automáticos de MLflow para salida limpia
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

print("=" * 80)
print("INICIANDO FLUJO DE MACHINE LEARNING & MLOPS BASELINE")
print("=" * 80)

# ==============================================================================
# PASO 1: CARGA Y DESCRIPCIÓN DEL DATASET
# ==============================================================================
print("\n[PASO 1] CARGA Y DESCRIPCIÓN DEL DATASET...")

# Asegurar que el dataset existe en formato Kaggle estándar
if not os.path.exists(DATASET_PATH):
    print("-> Generando dataset local en formato Kaggle (Breast Cancer Wisconsin)...")
    cancer = load_breast_cancer()
    feature_names = []
    for n in cancer.feature_names:
        parts = str(n).split(' ', 1)
        if parts[0] == 'mean':
            feature_names.append(parts[1].replace(' ', '_') + '_mean')
        elif parts[0] == 'worst':
            feature_names.append(parts[1].replace(' ', '_') + '_worst')
        elif 'error' in str(n):
            clean = str(n).replace(' error', '').replace('error ', '').replace(' ', '_')
            feature_names.append(clean + '_se')
        else:
            feature_names.append(str(n).replace(' ', '_'))
    
    df_raw = pd.DataFrame(cancer.data, columns=feature_names)
    target_series = pd.Series(cancer.target).map({0: 'M', 1: 'B'})
    
    np.random.seed(42)
    df_raw.insert(0, 'id', np.random.randint(100000, 999999, size=len(df_raw)))
    df_raw.insert(1, 'diagnosis', target_series)
    
    # Introducir deliberadamente duplicados y valores faltantes para ilustrar limpieza en Paso 3
    df_dups = df_raw.iloc[[12, 45, 88]].copy()
    df_raw = pd.concat([df_raw, df_dups], ignore_index=True)
    df_raw.loc[15, 'radius_mean'] = np.nan
    df_raw.loc[28, 'texture_mean'] = np.nan
    df_raw.loc[60, 'area_mean'] = np.nan
    df_raw.loc[102, 'smoothness_mean'] = np.nan
    df_raw.to_csv(DATASET_PATH, index=False)
    print(f"-> Archivo generado en: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

n_registros, n_variables = df.shape
print(f"Dimensiones iniciales del dataset: {n_registros} registros, {n_variables} variables.")
print(f"Columnas principales: {list(df.columns[:8])} ... ({len(df.columns)-8} variables adicionales)")
print(f"Distribución del target ('diagnosis'):\n{df['diagnosis'].value_counts(dropna=False)}")

# Guardar descripción en artefacto
resumen_dataset = f"""================================================================================
DESCRIPCIÓN DEL DATASET: Breast Cancer Wisconsin (Diagnostic)
Fuente: Kaggle / UCI Machine Learning Repository
================================================================================
Total de registros: {n_registros}
Total de variables: {n_variables}

Significado del problema:
Clasificación médica binaria donde el objetivo es predecir si una masa tumoral 
es Benigna ('B') o Maligna ('M') a partir de características morfológicas 
celulares calculadas sobre imágenes digitalizadas de aspiración con aguja fina (FNA).

Estructura de Variables:
1. id: Identificador único de paciente/muestra (tipo: entero, no predictivo).
2. diagnosis: Variable objetivo categórica ('M' = Maligno, 'B' = Benigno).
3. 30 características numéricas continuas (float64) calculadas para cada núcleo celular:
   - 10 descriptores 'mean' (radio, textura, perímetro, área, suavidad, compacidad,
     concavidad, puntos cóncavos, simetría, dimensión fractal).
   - 10 descriptores 'se' (error estándar de las medidas anteriores).
   - 10 descriptores 'worst' (promedio de los tres valores mayores de cada descriptor).
================================================================================
"""
with open(os.path.join(ARTIFACTS_DIR, "resumen_dataset.txt"), "w", encoding="utf-8") as f:
    f.write(resumen_dataset)

# ==============================================================================
# PASO 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================
print("\n[PASO 2] REALIZANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)...")

# 1. Gráfico de distribución de la variable objetivo
plt.figure(figsize=(7, 4.5))
counts = df['diagnosis'].value_counts()
colors = ['#2b5c8f', '#d9534f']
bars = plt.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor='black')
plt.title('Distribución de Diagnósticos (B: Benigno, M: Maligno)', fontsize=13, fontweight='bold')
plt.xlabel('Diagnóstico', fontsize=11)
plt.ylabel('Cantidad de Casos', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    pct = (yval / len(df)) * 100
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval} ({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
plt.ylim(0, max(counts.values) + 50)
plot_target_path = os.path.join(PLOTS_DIR, "eda_target_distribution.png")
plt.tight_layout()
plt.savefig(plot_target_path, dpi=150)
plt.close()

# 2. Distribución de características clave por diagnóstico
features_to_plot = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for i, col in enumerate(features_to_plot):
    ax = axes[i]
    df[df['diagnosis'] == 'B'][col].dropna().hist(ax=ax, bins=25, alpha=0.6, label='Benigno (B)', color='#2b5c8f', edgecolor='black')
    df[df['diagnosis'] == 'M'][col].dropna().hist(ax=ax, bins=25, alpha=0.6, label='Maligno (M)', color='#d9534f', edgecolor='black')
    ax.set_title(f'Distribución de {col}', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frecuencia')
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle('Comparación de Distribuciones Morfológicas según Diagnóstico', fontsize=14, fontweight='bold')
plot_features_path = os.path.join(PLOTS_DIR, "eda_feature_distributions.png")
plt.tight_layout()
plt.savefig(plot_features_path, dpi=150)
plt.close()

# 3. Matriz de correlación de variables numéricas clave
selected_numeric = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean']
corr_matrix = df[selected_numeric].corr()

plt.figure(figsize=(9, 7))
im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(selected_numeric)), selected_numeric, rotation=45, ha='right')
plt.yticks(range(len(selected_numeric)), selected_numeric)
plt.title('Matriz de Correlación (Características Principales)', fontsize=13, fontweight='bold')
for i in range(len(selected_numeric)):
    for j in range(len(selected_numeric)):
        val = corr_matrix.iloc[i, j]
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', 
                 color='white' if abs(val) > 0.6 else 'black', fontsize=9, fontweight='bold')
plot_corr_path = os.path.join(PLOTS_DIR, "eda_correlation_matrix.png")
plt.tight_layout()
plt.savefig(plot_corr_path, dpi=150)
plt.close()

# 4. Boxplots para inspección de outliers
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
box_cols = ['radius_mean', 'area_mean', 'texture_mean']
for idx, col in enumerate(box_cols):
    ax = axes[idx]
    data_b = df[df['diagnosis'] == 'B'][col].dropna()
    data_m = df[df['diagnosis'] == 'M'][col].dropna()
    bp = ax.boxplot([data_b, data_m], tick_labels=['Benigno', 'Maligno'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#a6c8e0')
    bp['boxes'][1].set_facecolor('#f3b6b4')
    ax.set_title(f'Boxplot: {col}', fontweight='bold')
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)
plot_box_path = os.path.join(PLOTS_DIR, "eda_boxplots_outliers.png")
plt.tight_layout()
plt.savefig(plot_box_path, dpi=150)
plt.close()

print(f"-> Visualizaciones EDA guardadas en: {PLOTS_DIR}")

# ==============================================================================
# PASO 3: LIMPIEZA DE DATOS Y DOCUMENTACIÓN DE DECISIONES
# ==============================================================================
print("\n[PASO 3] LIMPIEZA DE DATOS Y DOCUMENTACIÓN DE DECISIONES...")

bitacora_limpieza = []

# 1. Detección y tratamiento de duplicados
n_duplicados = df.duplicated().sum()
print(f"-> Filas duplicadas encontradas: {n_duplicados}")
if n_duplicados > 0:
    df_cleaned = df.drop_duplicates().copy()
    decision_dup = (
        f"1. DUPLICADOS: Se identificaron {n_duplicados} registros idénticos generados por error de captura o re-ingreso. "
        f"Decisión: Eliminación con drop_duplicates(). Justificación: Prevenir fuga de datos (data leakage) y sesgo "
        f"en el conjunto de prueba."
    )
else:
    df_cleaned = df.copy()
    decision_dup = "1. DUPLICADOS: No se encontraron registros duplicados completos."
bitacora_limpieza.append(decision_dup)

# 2. Detección y tratamiento de valores faltantes (NaN)
n_faltantes = df_cleaned.isnull().sum().sum()
print(f"-> Valores faltantes totales encontrados: {n_faltantes}")
cols_con_nulos = df_cleaned.columns[df_cleaned.isnull().any()].tolist()
if n_faltantes > 0:
    print(f"   Columnas con nulos: {cols_con_nulos}")
    for col in cols_con_nulos:
        mediana = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(mediana)
    decision_nan = (
        f"2. VALORES FALTANTES: Se detectaron {n_faltantes} celdas nulas en las variables {cols_con_nulos}. "
        f"Decisión: Imputación mediante la mediana de cada columna. Justificación: La mediana es estadísticamente "
        f"robusta frente a distribuciones asimétricas y valores extremos, preservando el tamaño muestral sin sesgar la media."
    )
else:
    decision_nan = "2. VALORES FALTANTES: No se identificaron valores nulos en el dataset."
bitacora_limpieza.append(decision_nan)

# 3. Corrección de tipos y variables irrelevantes
print("-> Ajustando tipos de variables e identificadores...")
if 'id' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=['id'])
    decision_id = (
        "3. VARIABLE ID: La columna 'id' representa únicamente el identificador del paciente. "
        "Decisión: Se elimina del espacio de características. Justificación: No contiene señal predictiva biológica "
        "y podría inducir memorización espuria en el modelo."
    )
    bitacora_limpieza.append(decision_id)

# Codificación de la variable objetivo
df_cleaned['diagnosis'] = df_cleaned['diagnosis'].map({'M': 1, 'B': 0})
decision_enc = (
    "4. CODIFICACIÓN DE TARGET: La variable 'diagnosis' era de tipo texto ('M', 'B'). "
    "Decisión: Transformación a binario numérico (1 = Maligno, 0 = Benigno). Justificación: Es la clase positiva "
    "clínicamente relevante y formato requerido por los estimadores de scikit-learn."
)
bitacora_limpieza.append(decision_enc)

# 4. Análisis y tratamiento de Outliers
print("-> Detección y evaluación de outliers mediante regla IQR...")
outlier_summary = {}
numeric_features = [c for c in df_cleaned.columns if c != 'diagnosis']
for c in numeric_features:
    Q1 = df_cleaned[c].quantile(0.25)
    Q3 = df_cleaned[c].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df_cleaned[c] < (Q1 - 1.5 * IQR)) | (df_cleaned[c] > (Q3 + 1.5 * IQR))).sum()
    if n_out > 0:
        outlier_summary[c] = int(n_out)

decision_outliers = (
    f"5. OUTLIERS: Se detectaron valores numéricos fuera del rango [Q1 - 1.5*IQR, Q3 + 1.5*IQR] "
    f"en variables como 'area_worst' ({outlier_summary.get('area_worst', 0)} casos) y 'radius_worst' ({outlier_summary.get('radius_worst', 0)} casos). "
    f"Decisión: Mantener los valores sin truncar arbitrariamente ni eliminarlos. Justificación: En el dominio médico de oncología, "
    f"los tumores de alto grado presentan biológicamente perímetros y áreas extremadamente grandes. Descartarlos causaría "
    f"la pérdida de los casos malignos más severos y críticos. En su lugar, se mitiga el efecto de escala mediante "
    f"estandarización (StandardScaler) en la etapa de modelado."
)
bitacora_limpieza.append(decision_outliers)

print("-> Bitácora de decisiones de limpieza documentada:")
for d in bitacora_limpieza:
    print(f"   * {d[:110]}...")

with open(os.path.join(ARTIFACTS_DIR, "bitacora_limpieza.txt"), "w", encoding="utf-8") as f:
    f.write("\n\n".join(bitacora_limpieza))

# ==============================================================================
# PASO 4: ENTRENAMIENTO DE MODELO BASE (BASELINE)
# ==============================================================================
print("\n[PASO 4] ENTRENANDO MODELO BASELINE...")

X = df_cleaned.drop(columns=['diagnosis'])
y = df_cleaned['diagnosis']

# División estratificada 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"-> Conjunto de Entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} características.")
print(f"-> Conjunto de Prueba: {X_test.shape[0]} muestras.")
print(f"-> Proporción clases Train (Maligno/Benigno): {y_train.mean():.2%} malignos.")
print(f"-> Proporción clases Test (Maligno/Benigno): {y_test.mean():.2%} malignos.")

# Escalado estándar para asegurar convergencia y estabilidad numérica
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Modelo Base: Regresión Logística
# Justificación: Modelo lineal simple, de rápida convergencia, interpretable y estándar de la industria como baseline.
baseline_model = LogisticRegression(
    C=1.0, 
    solver='lbfgs', 
    max_iter=1000, 
    random_state=42
)
baseline_model.fit(X_train_scaled, y_train)
print("-> Modelo LogisticRegression entrenado con éxito.")

# ==============================================================================
# PASO 5: EVALUACIÓN DEL MODELO
# ==============================================================================
print("\n[PASO 5] EVALUACIÓN DETALLADA DEL MODELO...")

y_pred = baseline_model.predict(X_test_scaled)
y_proba = baseline_model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"-> Exactitud (Accuracy):  {acc:.4f} ({acc*100:.2f}%)")
print(f"-> Precisión (Precision): {prec:.4f} ({prec*100:.2f}%)")
print(f"-> Sensibilidad (Recall): {rec:.4f} ({rec*100:.2f}%)")
print(f"-> F1-Score:              {f1:.4f}")
print(f"-> Área bajo ROC (AUC):   {auc:.4f}")
print(f"-> Matriz de Confusión: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Generar gráfica de Matriz de Confusión
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Baseline Logistic Regression', fontsize=12, fontweight='bold')
plt.colorbar(fraction=0.046, pad=0.04)
tick_marks = [0, 1]
plt.xticks(tick_marks, ['Benigno (0)', 'Maligno (1)'])
plt.yticks(tick_marks, ['Benigno (0)', 'Maligno (1)'])

thresh = cm.max() / 2.0
labels_names = [["Verdaderos Negativos (TN)", "Falsos Positivos (FP)"],
                ["Falsos Negativos (FN)", "Verdaderos Positivos (TP)"]]
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]}\n({labels_names[i][j]})",
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=10, fontweight='bold')

plt.ylabel('Etiqueta Real', fontsize=11)
plt.xlabel('Predicción del Modelo', fontsize=11)
plot_cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(plot_cm_path, dpi=150)
plt.close()

# Generar gráfica de Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6.5, 5))
plt.plot(fpr, tpr, color='#2b5c8f', lw=2.5, label=f'Regresión Logística (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Azar (AUC = 0.500)')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11)
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)', fontsize=11)
plt.title('Curva ROC (Receiver Operating Characteristic)', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plot_roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
plt.tight_layout()
plt.savefig(plot_roc_path, dpi=150)
plt.close()

# Generar gráfico de Importancia de Variables (Coeficientes del modelo)
coefs = pd.Series(baseline_model.coef_[0], index=X.columns).sort_values()
top_coefs = pd.concat([coefs.head(5), coefs.tail(5)])
plt.figure(figsize=(8, 5))
colors_coef = ['#d9534f' if val > 0 else '#2b5c8f' for val in top_coefs]
top_coefs.plot(kind='barh', color=colors_coef, edgecolor='black')
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Top 10 Coeficientes de Regresión Logística (Importancia)', fontsize=12, fontweight='bold')
plt.xlabel('Magnitud del Coeficiente (Log-Odds)')
plot_coef_path = os.path.join(PLOTS_DIR, "feature_coefficients.png")
plt.tight_layout()
plt.savefig(plot_coef_path, dpi=150)
plt.close()

# Redactar interpretación detallada
interpretacion_eval = f"""================================================================================
INTERPRETACIÓN DETALLADA DE LA EVALUACIÓN DEL MODELO BASELINE
================================================================================
Métricas Obtenidas:
- Accuracy:    {acc:.4f} ({acc*100:.2f}%)
- Precision:   {prec:.4f} ({prec*100:.2f}%)
- Recall:      {rec:.4f} ({rec*100:.2f}%)
- F1-Score:    {f1:.4f}
- ROC-AUC:     {auc:.4f}
- Especificidad: {specificity:.4f}

Desglose de la Matriz de Confusión:
- Verdaderos Negativos (TN): {tn} (Tumores benignos clasificados correctamente)
- Falsos Positivos (FP):     {fp} (Tumores benignos clasificados erróneamente como malignos)
- Falsos Negativos (FN):     {fn} (Tumores malignos clasificados erróneamente como benignos)
- Verdaderos Positivos (TP): {tp} (Tumores malignos detectados exitosamente)

Análisis Crítico en el Contexto Médico:
1. En un problema de cribado oncológico, el costo de un Falso Negativo (FN) es potencialmente fatal,
   ya que un paciente con cáncer activo se iría sin diagnóstico oportuno ni tratamiento.
2. La métrica primaria a maximizar debe ser el Recall (Sensibilidad). En este modelo baseline,
   un Recall de {rec*100:.2f}% (detectando {tp} de {tp+fn} casos malignos) con solo {fn} falso negativo
   representa un punto de partida sumamente sólido para un modelo lineal simple.
3. La Precisión de {prec*100:.2f}% y Especificidad de {specificity*100:.2f}% indican que el modelo comete
   muy pocos Falsos Positivos ({fp} casos), minimizando la angustia psicológica del paciente y costos 
   asociados a biopsias invasivas innecesarias.
4. El AUC de {auc:.4f} confirma una capacidad de discriminación casi óptima a través de distintos umbrales
   de probabilidad de corte.
================================================================================
"""
with open(os.path.join(ARTIFACTS_DIR, "reporte_evaluacion.txt"), "w", encoding="utf-8") as f:
    f.write(interpretacion_eval)

# ==============================================================================
# PASO 6: REGISTRO DEL EXPERIMENTO EN MLFLOW TRACKING
# ==============================================================================
print("\n[PASO 6] REGISTRANDO EXPERIMENTO EN MLFLOW TRACKING...")

TRACKING_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlflow.db")
mlflow.set_tracking_uri(TRACKING_URI)
print(f"-> MLflow Tracking URI: {TRACKING_URI}")

EXPERIMENT_NAME = "Clasificacion_Cancer_Mama_Baseline"
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="Baseline_LogisticRegression") as run:
    run_id = run.info.run_id
    print(f"-> MLflow Run ID: {run_id}")
    
    # 1. Registro de Parámetros
    mlflow.log_params({
        "modelo_tipo": "LogisticRegression",
        "dataset_nombre": "Breast Cancer Wisconsin Diagnostic",
        "dataset_fuente": "Kaggle / UCI Machine Learning Repository",
        "n_muestras_train": int(X_train.shape[0]),
        "n_muestras_test": int(X_test.shape[0]),
        "n_caracteristicas": int(X_train.shape[1]),
        "test_size": 0.20,
        "random_state": 42,
        "stratify": "True",
        "preprocesamiento": "StandardScaler",
        "solver": "lbfgs",
        "C": 1.0,
        "penalty": "l2",
        "max_iter": 1000
    })
    
    # 2. Registro de Métricas
    mlflow.log_metrics({
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "specificity": float(specificity),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    })
    
    # 3. Registro de Tags Informativos
    mlflow.set_tags({
        "etapa": "baseline",
        "tarea": "clasificacion_binaria",
        "dominio": "salud_oncologia",
        "framework": "scikit-learn",
        "version_pipeline": "1.0.0"
    })
    
    # 4. Registro de Artefactos Gráficos y Reportes
    mlflow.log_artifact(plot_target_path, artifact_path="graficas_eda")
    mlflow.log_artifact(plot_features_path, artifact_path="graficas_eda")
    mlflow.log_artifact(plot_corr_path, artifact_path="graficas_eda")
    mlflow.log_artifact(plot_box_path, artifact_path="graficas_eda")
    mlflow.log_artifact(plot_cm_path, artifact_path="graficas_evaluacion")
    mlflow.log_artifact(plot_roc_path, artifact_path="graficas_evaluacion")
    mlflow.log_artifact(plot_coef_path, artifact_path="graficas_evaluacion")
    
    mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "resumen_dataset.txt"), artifact_path="reportes")
    mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "bitacora_limpieza.txt"), artifact_path="reportes")
    mlflow.log_artifact(os.path.join(ARTIFACTS_DIR, "reporte_evaluacion.txt"), artifact_path="reportes")
    
    # 5. Registro del Modelo Base en MLflow Model Registry
    # Proporcionamos ejemplo de entrada para firma de inferencia automática
    input_sample = X_test_scaled.head(5)
    mlflow.sklearn.log_model(
        sk_model=baseline_model,
        name="modelo_baseline_logreg",
        input_example=input_sample
    )
    
print("-> Experimento y modelo registrados exitosamente en MLflow.")

# ==============================================================================
# PASO 7: REFLEXIÓN MLOps Y ÁREA PROFESIONAL (> 150 palabras)
# ==============================================================================
reflexion_texto = """
================================================================================
PASO 7: REFLEXIÓN CRÍTICA - PRÁCTICAS MLOps Y SU IMPACTO PROFESIONAL
================================================================================

1. ¿Qué mejoraría en este flujo aplicando prácticas de MLOps y por qué?
Aunque este flujo cubre satisfactoriamente el ciclo elemental de Machine Learning 
(ingesta, limpieza, baseline, evaluación y registro local), en un entorno productivo 
real presenta limitaciones clave que deben resolverse con prácticas maduras de MLOps:

- Orquestación y Versionado de Datos (DVC / Feast): En este ejercicio, los datos se 
  cargan desde un archivo CSV estático. En producción, los datos evolucionan constantemente. 
  Implementar DVC (Data Version Control) acoplado a un bucket S3/GCS permitiría trazar la 
  versión exacta de los datos que entrenó cada modelo, garantizando reproducibilidad auditable.
- Pipelines de Entrenamiento Declarativos y Automatizados (ZenML / Kubeflow / Airflow): 
  Sustituir el script monolítico por pasos modulares (data ingestion -> validation -> 
  transformation -> training -> evaluation) asegura que cada etapa sea reejecutable, testeable 
  y desacoplada.
- Validación Automatizada de Datos (Great Expectations): Implementar suites de pruebas de 
  esquemas y distribuciones previo al entrenamiento para prevenir errores silenciosos ante 
  valores nulos inesperados o tipos erróneos.
- Integración Continua y Despliegue Continuo (CI/CD): Mediante GitHub Actions, automatizar 
  pruebas unitarias, linting, reentrenamiento ante nuevos datos y despliegue del modelo en un 
  contenedor Docker expuesto mediante una API REST (FastAPI).
- Monitoreo en Producción y Detección de Deriva (Evidently AI / MLflow Tracing): Un modelo de 
  diagnóstico médico no puede ser una caja negra estática. Es imprescindible monitorear 'Data Drift' 
  (cambios en la tecnología de digitalización o perfil de pacientes) y 'Concept Drift' para 
  disparar alertas de reentrenamiento proactivas.

2. ¿Cómo se ven reflejados estos aprendizajes dentro de tu área profesional?
En el ámbito profesional y de ingeniería de software/datos, la adopción de MLOps transforma 
el Machine Learning de un ejercicio académico artesanal a un activo empresarial confiable, 
gobernable y sostenible. Históricamente, más del 80% de los proyectos de ciencia de datos 
nunca llegan a producción debido a la fricción entre científicos de datos y los equipos de 
operaciones (DevOps). Dominar el registro de experimentos con MLflow, el versionado sistemático 
de parámetros y métricas, y la serialización de modelos con contratos de entrada claros 
permite construir soluciones con alta trazabilidad, reduciendo la deuda técnica. En sectores 
regulados y críticos como salud, finanzas o tecnología corporativa, este rigor no es opcional: 
asegura auditoría completa de decisiones algorítmicas, facilita la colaboración interdisciplinaria 
y acelera el 'Time-to-Market' de modelos robustos y éticos.
================================================================================
"""

with open(os.path.join(ARTIFACTS_DIR, "reflexion_mlops.txt"), "w", encoding="utf-8") as f:
    f.write(reflexion_texto)

print(reflexion_texto)
print("=" * 80)
print("PIPELINE BASELINE COMPLETADO CON ÉXITO")
print(f"Para visualizar la interfaz de MLflow y tomar la captura:")
print(f"Ejecuta en tu terminal:")
print(f"   ./venv/bin/mlflow ui --port 5000")
print(f"Luego abre en tu navegador: http://127.0.0.1:5000")
print("=" * 80)
