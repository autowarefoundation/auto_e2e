variable "cluster_name" { type = string }
variable "artifacts_bucket" { type = string }
variable "region" { type = string }

resource "helm_release" "mlflow" {
  name       = "mlflow"
  repository = "https://community-charts.github.io/helm-charts"
  chart      = "mlflow"
  version    = "1.8.5"
  namespace  = "mlflow"

  values = [file("${path.module}/../../../helm-values/mlflow.yaml")]

  set {
    name  = "backendStore.postgres.enabled"
    value = "true"
  }
  set {
    name  = "backendStore.postgres.existingSecret"
    value = "mlflow-db-secret"
  }
  set {
    name  = "artifactRoot.proxiedArtifactStorage"
    value = "true"
  }
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
    name  = "artifactRoot.s3.awsRegion"
    value = var.region
  }
  set {
    name  = "serviceAccount.create"
    value = "true"
  }
  set {
    name  = "serviceAccount.name"
    value = "mlflow"
  }
}
