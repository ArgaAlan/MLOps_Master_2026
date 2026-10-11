# Documentación: Actividad 2 - Experimentación con MLflow

Esta carpeta contiene la **documentación formal, el reporte técnico de análisis y las evidencias visuales** correspondientes a la implementación ubicada en [`../actividad_2_experimentacion_mlflow/`](../actividad_2_experimentacion_mlflow/).

---

## 📁 Contenido de esta Carpeta

```text
documentacion_actividad_2/
├── README.md                         <- Este documento de guía de la documentación
├── reporte_analisis.md               <- Reporte técnico formateado en Markdown (listo para PDF)
├── reporte_analisis.txt              <- Reporte técnico en formato texto plano
└── capturas_mlflow/                  <- Capturas de pantalla oficiales de la interfaz de MLflow
    ├── SS 1.png                      <- Vista general de corridas registradas
    ├── SS comparativa 1.png          <- Vista comparativa de parámetros vs. métricas
    ├── SS comparativa 2.png          <- Vista comparativa extendida
    └── SS Corrida_4_solver_liblinear.png <- Detalle de la mejor corrida con parámetros y artefactos
```

---

## 📊 Resumen Ejecutivo del Reporte de Análisis

### 1. Comando Exacto de Ejecución
```bash
python train.py --C 0.5 --solver liblinear --max_iter 1000 --seed 42 \
    --experiment_name "Optimizacion_Cancer_Mama_Hiperparametros" \
    --run_name "Corrida_4_Solver_Liblinear"
```

### 2. Tabla Comparativa de las 5 Corridas en MLflow

| Corrida | $C$ | Solver | Max Iter | Accuracy | Recall | Precision | F1-Score | ROC-AUC | FN | FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corrida_1_Baseline_Default** | 1.0 | `lbfgs` | 1000 | 0.9649 | 0.9286 | 0.9750 | 0.9512 | 0.9960 | 3 | 1 |
| **Corrida_2_Alta_Regularizacion** | 0.01 | `lbfgs` | 1000 | 0.9474 | 0.8571 | 1.0000 | 0.9231 | 0.9977 | 6 | 0 |
| **Corrida_3_Baja_Regularizacion** | 10.0 | `lbfgs` | 1000 | 0.9737 | 0.9524 | 0.9756 | 0.9639 | 0.9864 | 2 | 1 |
| **Corrida_4_Solver_Liblinear** ⭐ | **0.5** | **`liblinear`** | **1000** | **0.9825** | **0.9524** | **1.0000** | **0.9756** | **0.9974** | **2** | **0** |
| **Corrida_5_Solver_Saga** | 2.0 | `saga` | 2000 | 0.9737 | 0.9524 | 0.9756 | 0.9639 | 0.9940 | 2 | 1 |

### 3. Puntos Clave del Análisis
* **Mejor Configuración:** `Corrida_4_Solver_Liblinear` ($C=0.5$, solver `liblinear`). Logró la mayor exactitud (**98.25%**), precisión perfecta (**100.0%**) y minimizó los Falsos Negativos a solo 2 casos (**Recall: 95.24%**).
* **Hiperparámetro más influyente:** $C$ (inverso de la regularización $L_2$). Una penalización excesiva ($C=0.01$) causó subajuste severo duplicando los falsos negativos a 6 pacientes sin detectar.
* **Limitaciones:** Supuesto de frontera de decisión lineal, multicolinealidad severa entre radio/perímetro/área y tamaño muestral (114 observaciones de prueba).

---

## 📝 Preparación del Entregable PDF para Canvas

1. Abre [`reporte_analisis.md`](reporte_analisis.md) o [`reporte_analisis.txt`](reporte_analisis.txt).
2. Asegúrate de incluir tu nombre y matrícula.
3. Incrusta las capturas de la carpeta `capturas_mlflow/`:
   * **Captura 1:** `capturas_mlflow/SS comparativa 1.png`
   * **Captura 2:** `capturas_mlflow/SS Corrida_4_solver_liblinear.png`
4. Guarda y exporta el archivo como:  
   **`TC5061_Experimentacion_MLFlow_[TuApellidoTuNombre].pdf`**

