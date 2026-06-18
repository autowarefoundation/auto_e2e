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
    yamlencode({
      flyte-core-components = {
        admin = {
          seedProjects = ["auto-e2e"]
        }
      }
      configuration = {
        externalConfigMap = "flyte-custom-config"
        auth = {
          enabled = false
        }
      }
      deployment = {
        resources = {
          requests = { cpu = "500m", memory = "1Gi" }
          limits   = { memory = "2Gi" }
        }
      }
      serviceAccount = {
        create = true
        name   = "flyte-backend-flyte-binary"
      }
    })
  ]
}
