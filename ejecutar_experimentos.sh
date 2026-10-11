#!/bin/bash
# Script de automatización de experimentos para MLflow Tracking
# Ejecuta 5 corridas sistemáticas con diferentes hiperparámetros

set -e

PYTHON_EXEC="./venv/bin/python"
EXPERIMENT="Optimizacion_Cancer_Mama_Hiperparametros"

echo "========================================================================"
echo "INICIANDO EJECUCIÓN DE 5 CORRIDAS EXPERIMENTALES EN MLFLOW"
echo "Experimento: $EXPERIMENT"
echo "========================================================================"

# Corrida 1: Baseline por defecto
echo -e "\n>>> [1/5] Ejecutando Corrida 1: Baseline por defecto (C=1.0, solver=lbfgs)..."
$PYTHON_EXEC train.py \
    --C 1.0 \
    --solver lbfgs \
    --max_iter 1000 \
    --seed 42 \
    --experiment_name "$EXPERIMENT" \
    --run_name "Corrida_1_Baseline_Default"

# Corrida 2: Alta regularización (C bajo = 0.01)
echo -e "\n>>> [2/5] Ejecutando Corrida 2: Alta regularización (C=0.01, solver=lbfgs)..."
$PYTHON_EXEC train.py \
    --C 0.01 \
    --solver lbfgs \
    --max_iter 1000 \
    --seed 42 \
    --experiment_name "$EXPERIMENT" \
    --run_name "Corrida_2_Alta_Regularizacion"

# Corrida 3: Baja regularización (C alto = 10.0)
echo -e "\n>>> [3/5] Ejecutando Corrida 3: Baja regularización (C=10.0, solver=lbfgs)..."
$PYTHON_EXEC train.py \
    --C 10.0 \
    --solver lbfgs \
    --max_iter 1000 \
    --seed 42 \
    --experiment_name "$EXPERIMENT" \
    --run_name "Corrida_3_Baja_Regularizacion"

# Corrida 4: Solver Liblinear con regularización intermedia (C=0.5)
echo -e "\n>>> [4/5] Ejecutando Corrida 4: Solver Liblinear (C=0.5, solver=liblinear)..."
$PYTHON_EXEC train.py \
    --C 0.5 \
    --solver liblinear \
    --max_iter 1000 \
    --seed 42 \
    --experiment_name "$EXPERIMENT" \
    --run_name "Corrida_4_Solver_Liblinear"

# Corrida 5: Solver SAGA con regularización flexible (C=2.0, max_iter=2000)
echo -e "\n>>> [5/5] Ejecutando Corrida 5: Solver SAGA (C=2.0, solver=saga, max_iter=2000)..."
$PYTHON_EXEC train.py \
    --C 2.0 \
    --solver saga \
    --max_iter 2000 \
    --seed 42 \
    --experiment_name "$EXPERIMENT" \
    --run_name "Corrida_5_Solver_Saga"

echo -e "\n========================================================================"
echo "TODAS LAS 5 CORRIDAS FUERON REGISTRADAS CON ÉXITO EN MLFLOW"
echo "Para comparar las corridas en la interfaz web, ejecuta:"
echo "    ./venv/bin/mlflow ui --port 5000"
echo "Luego abre en tu navegador: http://127.0.0.1:5000"
echo "========================================================================"
