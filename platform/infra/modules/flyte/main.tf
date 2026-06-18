variable "cluster_name" { type = string }
variable "artifacts_bucket" { type = string }
variable "region" { type = string }
variable "rds_endpoint" { type = string }

resource "helm_release" "flyte" {
  name       = "flyte-backend"
  repository = "https://flyteorg.github.io/flyte"
  chart      = "flyte-binary"
  version    = "0.1.10"
  namespace  = "flyte"

  values = [file("${path.module}/../../../helm-values/flyte.yaml")]

  # Database
  set {
    name  = "configuration.database.postgres.host"
    value = split(":", var.rds_endpoint)[0]
  }
  set {
    name  = "configuration.database.postgres.port"
    value = "5432"
  }
  set {
    name  = "configuration.database.postgres.dbname"
    value = "flyteadmin"
  }
  set {
    name  = "configuration.database.postgres.username"
    value = "pgadmin"
  }
  set {
    name  = "configuration.database.postgres.passwordPath"
    value = "/etc/flyte/db-pass/POSTGRES_PASSWORD"
  }

  # Storage (S3)
  set {
    name  = "configuration.storage.metadataContainer"
    value = var.artifacts_bucket
  }
  set {
    name  = "configuration.storage.userDataContainer"
    value = var.artifacts_bucket
  }
  set {
    name  = "configuration.storage.provider"
    value = "s3"
  }
  set {
    name  = "configuration.storage.providerConfig.s3.region"
    value = var.region
  }

  # Enable pytorch plugin
  set {
    name  = "configuration.inline.plugins.k8s.enabled-plugins[0]"
    value = "container"
  }
  set {
    name  = "configuration.inline.plugins.k8s.enabled-plugins[1]"
    value = "sidecar"
  }
  set {
    name  = "configuration.inline.plugins.k8s.enabled-plugins[2]"
    value = "k8s-array"
  }
  set {
    name  = "configuration.inline.plugins.k8s.enabled-plugins[3]"
    value = "pytorch"
  }
  set {
    name  = "configuration.inline.plugins.k8s.default-for-task-types.container"
    value = "container"
  }
  set {
    name  = "configuration.inline.plugins.k8s.default-for-task-types.pytorch"
    value = "pytorch"
  }

  # Mount DB secret
  set {
    name  = "configuration.externalSecretRef"
    value = "flyte-db-pass"
  }

  # SA for Pod Identity
  set {
    name  = "serviceAccount.create"
    value = "true"
  }
  set {
    name  = "serviceAccount.name"
    value = "flyte-backend-flyte-binary"
  }
}
