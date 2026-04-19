import os

# Use current directory as root
root = "."

structure = {
    "api": [
        "main.py",
        "predict.py",
        "logger.py",
        "metrics.py",
        "schemas.py"
    ],
    "training": [
        "train.py",
        "evaluate.py",
        "register_model.py"
    ],
    "data": [
        "ingest.py",
        "validate.py",
        "features.py",
        "drift_injector.py"
    ],
    "monitoring": [
        "drift_report.py",
        "quality_report.py",
        "prometheus_exporter.py"
    ],
    "airflow/dags": [
        "training_dag.py",
        "drift_check_dag.py",
        "retrain_trigger_dag.py"
    ],
    "alerting": [
        "grafana_alerts.json",
        "notify.py"
    ],
    "dockerfiles": [
        "Dockerfile.api",
        "Dockerfile.airflow"
    ],
    "grafana": [
        "dashboards.json",
        "datasources.yaml"
    ],
    "prometheus": [
        "prometheus.yml"
    ],
    ".github/workflows": [
        "ci.yml"
    ]
}

root_files = [
    "docker-compose.yml",
    ".env.example",
    "requirements.txt",
    "README.md",
    "init_db.sql"
]

def create_files():
    for folder, files in structure.items():
        folder_path = os.path.join(root, folder)

        # Ensure folder exists (safe even if already exists)
        os.makedirs(folder_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(folder_path, file)

            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("")

    # Root-level files
    for file in root_files:
        file_path = os.path.join(root, file)

        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")

    print("✅ Files created successfully!")

if __name__ == "__main__":
    create_files()