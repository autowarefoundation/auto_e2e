variable "cluster_name" { type = string }
variable "artifacts_bucket" { type = string }
variable "region" { type = string }
variable "rds_host" { type = string }
variable "rds_password" {
  type      = string
  sensitive = true
}

resource "helm_release" "mlflow" {
  name             = "mlflow"
  repository       = "https://community-charts.github.io/helm-charts"
  chart            = "mlflow"
  version          = "1.8.5"
  namespace        = "mlflow"
  create_namespace = true
  timeout          = 600
  wait             = false
  disable_openapi_validation = true

  values = [file("${path.module}/../../../helm-values/mlflow.yaml")]

  # Backend store: RDS Postgres
  set {
    name  = "backendStore.postgres.enabled"
    value = "true"
  }
  set {
    name  = "backendStore.postgres.host"
    value = var.rds_host
  }
  set {
    name  = "backendStore.postgres.port"
    value = "5432"
  }
  set {
    name  = "backendStore.postgres.database"
    value = "mlflow"
  }
  set {
    name  = "backendStore.postgres.user"
    value = "pgadmin"
  }
  set_sensitive {
    name  = "backendStore.postgres.password"
    value = var.rds_password
  }
  set {
    name  = "backendStore.postgres.driver"
    value = "psycopg2"
  }

  # Artifact storage: S3
  set {
    name  = "artifactRoot.s3.enabled"
    value = "true"
  }
  set {
    name  = "artifactRoot.s3.bucket"
    value = var.artifacts_bucket
  }
  set {
    name  = "artifactRoot.s3.path"
    value = "mlflow"
  }
  set {
    name  = "service.type"
    value = "ClusterIP"
  }
  set {
    name  = "artifactRoot.proxiedArtifactStorage"
    value = "true"
  }

  # SA for Pod Identity
  set {
    name  = "serviceAccount.create"
    value = "true"
  }
  set {
    name  = "serviceAccount.name"
    value = "mlflow"
  }

  # MLflow uvicorn mode (no gunicorn): enables security middleware + CORS
  set {
    name  = "extraEnvVars.MLFLOW_SERVER_ALLOWED_HOSTS"
    value = "*"
  }
  set {
    name  = "extraEnvVars.MLFLOW_SERVER_CORS_ALLOWED_ORIGINS"
    value = "*"
  }

  # AWS region via env var
  set {
    name  = "extraEnvVars.AWS_DEFAULT_REGION"
    value = var.region
  }
}
