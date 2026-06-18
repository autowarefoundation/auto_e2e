variable "cluster_name" { type = string }
variable "artifacts_bucket" { type = string }
variable "region" { type = string }
variable "rds_host" { type = string }
variable "rds_password" {
  type      = string
  sensitive = true
}

resource "helm_release" "flyte" {
  name             = "flyte-backend"
  repository       = "https://flyteorg.github.io/flyte"
  chart            = "flyte-binary"
  version          = "2.0.23"
  namespace        = "flyte"
  create_namespace = true
  timeout          = 600

  values = [file("${path.module}/../../../helm-values/flyte.yaml")]

  # Database
  set {
    name  = "configuration.database.host"
    value = var.rds_host
  }
  set {
    name  = "configuration.database.port"
    value = "5432"
  }
  set {
    name  = "configuration.database.dbname"
    value = "flyteadmin"
  }
  set {
    name  = "configuration.database.username"
    value = "pgadmin"
  }
  set_sensitive {
    name  = "configuration.database.password"
    value = var.rds_password
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
  set {
    name  = "configuration.storage.providerConfig.s3.authType"
    value = "iam"
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
