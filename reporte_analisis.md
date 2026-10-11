# Reporte Técnico: Experimentación y Seguimiento de Modelos con MLflow

**Asignatura:** TC5061 - Experimentación y MLOps  
**Nombre del Entregable:** `TC5061_Experimentacion_MLFlow_[ApellidoNombre].pdf`  
**Estudiante:** [Nombre Completo del Estudiante]  
**Matrícula:** [Matrícula / ID]  
**Repositorio de GitHub:** `https://github.com/argaalan/MLOps_Master_2026`  
**Formato:** Fuente Arial o Times New Roman, 11 pt, interlineado 1.5. Extensión: $\le 2$ cuartillas (sin contar capturas ni anexos).

---

## 1. Comando Exacto de Ejecución

El script de entrenamiento parametrizado fue desarrollado de forma modular en [`train.py`](file:///home/argaalan/mlops-proyecto/train.py) utilizando `argparse` y Scikit-Learn, instrumentado integralmente con MLflow Tracking.

* **Comando para reproducir la mejor configuración obtenida:**
  ```bash
  python train.py --C 0.5 --solver liblinear --max_iter 1000 --seed 42 \
      --experiment_name "Optimizacion_Cancer_Mama_Hiperparametros" \
      --run_name "Corrida_4_Solver_Liblinear"
  ```
* **Comando para reproducir las 5 corridas experimentales en lote:**
  ```bash
  ./ejecutar_experimentos.sh
  ```
* **Comando para iniciar el servidor de seguimiento de MLflow:**
  ```bash
  mlflow ui --port 5000
  ```

---

## 2. Tabla Comparativa de Corridas en MLflow Tracking

**Experimento:** `Optimizacion_Cancer_Mama_Hiperparametros`  
**Dataset:** *Breast Cancer Wisconsin Diagnostic* (569 observaciones tras limpieza, partición estratificada 80/20 con semilla fija `seed=42`).

| Corrida | $C$ | Solver | Max Iter | Accuracy | Recall | Precision | F1-Score | ROC-AUC | FN | FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corrida_1_Baseline_Default** | 1.0 | `lbfgs` | 1000 | 0.9649 | 0.9286 | 0.9750 | 0.9512 | 0.9960 | 3 | 1 |
| **Corrida_2_Alta_Regularizacion** | 0.01 | `lbfgs` | 1000 | 0.9474 | 0.8571 | 1.0000 | 0.9231 | 0.9977 | 6 | 0 |
| **Corrida_3_Baja_Regularizacion** | 10.0 | `lbfgs` | 1000 | 0.9737 | 0.9524 | 0.9756 | 0.9639 | 0.9864 | 2 | 1 |
| **Corrida_4_Solver_Liblinear** ⭐ | **0.5** | **`liblinear`** | **1000** | **0.9825** | **0.9524** | **1.0000** | **0.9756** | **0.9974** | **2** | **0** |
| **Corrida_5_Solver_Saga** | 2.0 | `saga` | 2000 | 0.9737 | 0.9524 | 0.9756 | 0.9639 | 0.9940 | 2 | 1 |

---

## 3. Análisis Técnico de Resultados

### A. Identificación de la Mejor Configuración
La configuración con mejor desempeño global corresponde a la **`Corrida_4_Solver_Liblinear`** ($C=0.5$, solver `liblinear`, max_iter 1000, semilla 42). 

Esta corrida alcanzó:
* La exactitud global más elevada (**Accuracy: 98.25%**, 112 de 114 casos correctamente clasificados).
* Una **Precisión perfecta del 100.0%** (0 Falsos Positivos).
* El mayor balance armónico (**F1-Score: 0.9756** y **ROC-AUC: 0.9974**).

En el contexto médico oncológico, la métrica reina es el **Recall (Sensibilidad)**, cuyo propósito es minimizar los Falsos Negativos ($FN$, pacientes con cáncer activo sin diagnóstico oportuno). La Corrida 4 redujo los falsos negativos a solo 2 casos (**Recall: 95.24%**), superando al baseline por defecto (que cometió 3 falsos negativos con Recall de 92.86%) y garantizando al mismo tiempo que ningún paciente sano sea sometido a la angustia y riesgo de biopsias invasivas innecesarias ($FP = 0$).

### B. Hiperparámetro Más Influyente
El hiperparámetro más influyente fue **$C$ (inverso de la fuerza de regularización $L_2$):**

1. **Subajuste por regularización excesiva ($C = 0.01$):** Provocó que los pesos de las características se comprimieran drásticamente hacia cero. Como consecuencia, el Recall cayó dramáticamente a un **85.71%**, duplicando los falsos negativos a 6 casos. Aunque la precisión fue del 100%, un modelo con esta tasa de omisión es clínicamente inviable.
2. **Flexibilidad óptima ($C \in [0.5, 10.0]$):** Al reducir la penalización, el modelo adquirió la capacidad de delimitar adecuadamente las densidades celulares malignas y benignas, estabilizando el Recall en **95.24%** (solo 2 falsos negativos).
3. **Influencia del Solver:** El algoritmo `liblinear` (basado en descenso por coordenadas) convergió hacia una solución ligeramente superior y más esparsa para este volumen muestral tabular en comparación con los solvers cuasi-Newton (`lbfgs`) y estocásticos (`saga`).

### C. Limitaciones Observadas
1. **Supuesto de Linealidad:** La regresión logística asume un hiperplano de separación lineal en el espacio de características estandarizado. Aunque alcanza un AUC sobresaliente (> 0.99), es incapaz de modelar interacciones biológicas no lineales complejas entre morfologías celulares (p. ej., variaciones no lineales acopladas de concavidad y textura).
2. **Multicolinealidad Severa:** Atributos como radio, perímetro y área exhiben coeficientes de correlación mayores a 0.98. Si bien la regularización $L_2$ asegura estabilidad numérica en la inversión matricial, infla la varianza de los coeficientes, dificultando atribuir causalidad clínica aislada a cada biomarcador.
3. **Escala Muestral:** Con 114 muestras de prueba, cada error de predicción individual representa un impacto de ~0.88% en la exactitud. En un entorno de producción médica, se requiere validación cruzada estratificada repetida ($K$-Fold) para obtener intervalos de confianza estadísticamente rigurosos.

---

## 4. Evidencias Visuales de MLflow Tracking

### Captura 1: Tabla Comparativa de Corridas en la Interfaz de MLflow
*(Inserta aquí la captura tomada desde http://127.0.0.1:5000 mostrando las 5 corridas seleccionadas en la vista "Compare", con sus parámetros y métricas).*

---

### Captura 2: Detalle de la Mejor Corrida (`Corrida_4_Solver_Liblinear`) y sus Artefactos
*(Inserta aquí la captura del detalle de la corrida seleccionada, mostrando los parámetros registrados, métricas y el panel de "Artifacts" desplegado con `confusion_matrix.png`, `roc_curve.png`, `classification_report.txt` y `modelo_logistic_regression`).*
