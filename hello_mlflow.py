import mlflow
import os

# 1. Crear un archivo de texto simple para usar como artefacto
artifact_name = "artefacto_prueba.txt"
with open(artifact_name, "w") as f:
    f.write("Hola, este es mi artefacto para la tarea de MLOps.")

# Asignar un nombre al experimento (recomendado para organización)
mlflow.set_experiment("Tarea_Hello_World")

# 2. Iniciar el registro del experimento en MLFlow
with mlflow.start_run(run_name="Ejecucion_Inicial"):
    
    # 3. Registrar parámetros (mlflow.log_param)
    mlflow.log_param("modelo_tipo", "Regresion")
    mlflow.log_param("learning_rate", 0.05)
    
    # 4. Registrar métricas (mlflow.log_metric)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("error", 0.08)
    
    # 5. Registrar artefacto (mlflow.log_artifact)
    mlflow.log_artifact(artifact_name)

print("¡Experimento Hello World registrado con éxito!")