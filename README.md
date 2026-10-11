# Proyecto MLOps: Experimentación y Seguimiento con MLflow

Este repositorio contiene la migración del modelo de aprendizaje automático desde un entorno exploratorio hacia un script de entrenamiento parametrizado, modular y reproducible (`train.py`), completamente instrumentado con **MLflow Tracking**.

> **Repositorio de GitHub:** `https://github.com/[TU_USUARIO]/[TU_REPOSITORIO]` *(Placeholder)*  
> **Asignatura:** TC5061 - Experimentación y MLOps  
> **Dataset:** *Breast Cancer Wisconsin (Diagnostic)* (Kaggle / UCI Machine Learning Repository)

---

## 📁 Estructura del Repositorio

```text
mlops-proyecto/
├── data/
│   └── breast_cancer.csv             <- Dataset tabular en formato estándar Kaggle / UCI
├── train.py                          <- Script modular y parametrizado de entrenamiento (CLI)
├── ejecutar_experimentos.sh          <- Script bash automatizado para ejecutar las 5 corridas
├── requirements.txt                  <- Archivo de dependencias con versiones fijadas
├── notebook_mlops_baseline.ipynb     <- Notebook reservado exclusivamente para exploración (EDA)
├── reporte_analisis.md               <- Análisis comparativo y reporte formal de resultados
├── reporte_analisis.txt              <- Versión en texto plano del reporte de resultados
├── mlflow.db                         <- Base de datos SQLite del backend de MLflow
├── mlruns/                           <- Repositorio local de artefactos y modelos registrados
└── README.md                         <- Documentación completa de ejecución y reproducibilidad
```

---

## ⚙️ Instalación y Entorno

1. Clona el repositorio:
   ```bash
   git clone https://github.com/[TU_USUARIO]/[TU_REPOSITORIO].git
   cd [TU_REPOSITORIO]
   ```

2. Crea y activa un entorno virtual de Python (>= 3.10):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Instala las dependencias con versiones fijadas:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Ejecución del Script Parametrizado (`train.py`)

El script `train.py` expone hiperparámetros del modelo, la semilla aleatoria y configuraciones de tracking mediante `argparse`. Puede ejecutarse con valores por defecto o con parámetros personalizados.

### 1. Ejecución con valores por defecto
```bash
python train.py
```
*Hiperparámetros por defecto:* `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `seed=42`, `test_size=0.20`.

### 2. Ejecución con hiperparámetros personalizados
```bash
python train.py \
    --C 0.5 \
    --solver liblinear \
    --max_iter 1000 \
    --seed 42 \
    --experiment_name "Optimizacion_Cancer_Mama_Hiperparametros" \
    --run_name "Mi_Corrida_Personalizada"
```

### 3. Argumentos disponibles en CLI (`python train.py --help`)
* `--C` *(float, default: 1.0)*: Inverso de la fuerza de regularización (menor valor = mayor penalización).
* `--solver` *(choices: ['lbfgs', 'liblinear', 'saga', 'newton-cg'], default: 'lbfgs')*: Algoritmo de optimización.
* `--max_iter` *(int, default: 1000)*: Número máximo de iteraciones para convergencia.
* `--seed` *(int, default: 42)*: Semilla para partición estratificada y estimador.
* `--test_size` *(float, default: 0.20)*: Proporción del conjunto de prueba.
* `--experiment_name` *(str)*: Nombre del experimento en MLflow.
* `--run_name` *(str)*: Nombre identificador de la corrida.
* `--data_path` *(str, default: 'data/breast_cancer.csv')*: Ruta al archivo de datos.

---

## 🧪 Ejecución Automatizada de las 5 Corridas

Para ejecutar en lote las 5 corridas experimentales y registrarlas bajo el experimento `Optimizacion_Cancer_Mama_Hiperparametros`, ejecuta:

```bash
chmod +x ejecutar_experimentos.sh
./ejecutar_experimentos.sh
```

### Resumen de las 5 Configuraciones Contrastadas:
1. **Corrida 1 (Baseline Default):** `C=1.0`, `solver=lbfgs`, `max_iter=1000`, `seed=42`
2. **Corrida 2 (Alta Regularización):** `C=0.01`, `solver=lbfgs`, `max_iter=1000`, `seed=42`
3. **Corrida 3 (Baja Regularización):** `C=10.0`, `solver=lbfgs`, `max_iter=1000`, `seed=42`
4. **Corrida 4 (Solver Liblinear):** `C=0.5`, `solver=liblinear`, `max_iter=1000`, `seed=42` *(Mejor desempeño)*
5. **Corrida 5 (Solver SAGA):** `C=2.0`, `solver=saga`, `max_iter=2000`, `seed=42`

---

## 📊 Visualización y Comparación en la Interfaz de MLflow

1. Inicia el servidor de seguimiento local:
   ```bash
   mlflow ui --port 5000
   ```
   *(o con el venv: `./venv/bin/mlflow ui --port 5000`)*

2. Abre en tu navegador:
   ```
   http://127.0.0.1:5000
   ```

3. Navega al experimento: **`Optimizacion_Cancer_Mama_Hiperparametros`**.
4. **Vista de Comparación:** Marca las casillas de las 5 corridas y haz clic en el botón **"Compare"**. Podrás ver tablas comparativas de parámetros vs. métricas, diagramas de coordenadas paralelas y gráficos de dispersión.
5. **Detalle de la Corrida:** Haz clic sobre **`Corrida_4_Solver_Liblinear`** para inspeccionar sus parámetros, métricas y la pestaña **Artifacts** con la matriz de confusión, curva ROC, reporte textual y el modelo empaquetado `modelo_logistic_regression`.

---

## 🔒 Reproducibilidad

La reproducibilidad estricta está garantizada mediante:
1. **Fijación de la semilla aleatoria (`--seed 42`):** Controla tanto la partición estratificada (`train_test_split`) como el generador de números pseudoaleatorios del optimizador (`LogisticRegression`). Dos ejecuciones con los mismos argumentos producen exactamente las mismas métricas hasta el último decimal.
2. **Dependencias versionadas:** Declaradas explícitamente en `requirements.txt`.
3. **Aislamiento de preprocesamiento:** El escalador `StandardScaler` se ajusta exclusivamente con datos de entrenamiento para prevenir fuga de información (*data leakage*).
