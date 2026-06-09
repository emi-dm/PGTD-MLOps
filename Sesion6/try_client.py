import mlflow
from experiment_tracking import EXPERIMENT_NAME

client = mlflow.MlflowClient()
experiment = client.get_experiment_by_name("sentiment-analysis-hf")
print(f"Experiment ID: {experiment}")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)

# Se muestra un resumen mínimo de la mejor ejecución.
best_run = runs[0]
print("\nMejor run detectada:")
print(f"run_id={best_run.info.run_id}")
print(f"accuracy={best_run.data.metrics.get('accuracy', 'N/A')}")
print(f"params={best_run.data.params}")
