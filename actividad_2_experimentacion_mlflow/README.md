# Actividad 2: Experimentación y Seguimiento con MLflow

Esta carpeta contiene el código fuente, datos y dependencias de la **Actividad 2** de la materia TC5061 - Experimentación y MLOps.

> 📄 **Documentación y Reporte:** El análisis comparativo de las 5 corridas y las capturas de pantalla de MLflow se encuentran organizados en la carpeta [`../documentacion_actividad_2/`](../documentacion_actividad_2/).

---

## 📁 Estructura de la Carpeta

```text
actividad_2_experimentacion_mlflow/
├── train.py                          <- Script modular y parametrizado de entrenamiento (CLI)
├── ejecutar_experimentos.sh          <- Script bash automatizado para reproducir las 5 corridas
├── requirements.txt                  <- Archivo de dependencias con versiones fijadas
├── notebook_mlops_baseline.ipynb     <- Notebook reservado exclusivamente para exploración (EDA)
├── data/
│   └── breast_cancer.csv             <- Dataset tabular en formato estándar Kaggle / UCI
└── README.md                         <- Documentación de ejecución de esta actividad
```

---

## ⚙️ Instalación y Requisitos

1. Asegúrate de tener Python 3.10 o superior y activar un entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Instala las dependencias con versiones fijadas:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Ejecución del Script de Entrenamiento (`train.py`)

El script `train.py` está modularizado en funciones con responsabilidad única (`load_data`, `preprocess_data`, `train_model`, `evaluate_model`, `generate_artifacts`, `log_to_mlflow`) y expone hiperparámetros mediante `argparse`.

### 1. Ejecución con hiperparámetros por defecto:
```bash
python train.py
```
*Valores por defecto:* `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `seed=42`, `test_size=0.20`.

### 2. Ejecución con hiperparámetros personalizados (Mejor configuración encontrada):
```bash
python train.py \
    --C 0.5 \
    --solver liblinear \
    --max_iter 1000 \
    --seed 42 \
    --experiment_name "Optimizacion_Cancer_Mama_Hiperparametros" \
    --run_name "Corrida_4_Solver_Liblinear"
```

### 3. Argumentos disponibles en la terminal (`python train.py --help`):
* `--C` *(float, default: 1.0)*: Inverso de la fuerza de regularización $L_2$.
* `--solver` *(choices: ['lbfgs', 'liblinear', 'saga', 'newton-cg'], default: 'lbfgs')*: Algoritmo de optimización.
* `--max_iter` *(int, default: 1000)*: Número máximo de iteraciones para convergencia.
* `--seed` *(int, default: 42)*: Semilla de reproducibilidad estricta.
* `--test_size` *(float, default: 0.20)*: Proporción del conjunto de prueba.
* `--experiment_name` *(str)*: Nombre del experimento en MLflow.
* `--run_name` *(str)*: Identificador de la corrida.
* `--data_path` *(str, default: 'data/breast_cancer.csv')*: Ruta al archivo CSV.

---

## 🧪 Ejecución Automatizada de las 5 Corridas

Para ejecutar en lote las 5 corridas experimentales contrastadas y registrarlas en MLflow:

```bash
chmod +x ejecutar_experimentos.sh
./ejecutar_experimentos.sh
```

---

## 📊 Visualización de Resultados en MLflow UI

Para visualizar y comparar las corridas en la interfaz web de MLflow:

```bash
mlflow ui --port 5000
```
Luego abre tu navegador en: `http://127.0.0.1:5000`.

