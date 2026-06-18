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
  wait             = false

  values = [
    file("${path.module}/../../../helm-values/flyte.yaml"),
    yamlencode({
      configuration = {
        database = {
          username = "pgadmin"
          password = var.rds_password
          host     = var.rds_host
          port     = 5432
          dbname   = "flyteadmin"
          options  = "sslmode=require"
        }
        storage = {
          metadataContainer = var.artifacts_bucket
          userDataContainer = var.artifacts_bucket
          provider          = "s3"
          providerConfig = {
            s3 = {
              region   = var.region
              authType = "iam"
            }
          }
        }
      }
    })
  ]

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
