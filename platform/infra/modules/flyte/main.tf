variable "cluster_name" { type = string }
variable "artifacts_bucket" { type = string }
variable "region" { type = string }
variable "rds_host" { type = string }
variable "rds_password" {
  type      = string
  sensitive = true
}

data "aws_caller_identity" "current" {}

resource "helm_release" "flyte" {
  name             = "flyte"
  repository       = "https://flyteorg.github.io/flyte"
  chart            = "flyte-core"
  version          = "1.16.7"
  namespace        = "flyte"
  create_namespace = true
  timeout          = 600
  wait             = false

  values = [
    file("${path.module}/../../../helm-values/flyte-core-eks.yaml"),
  ]

  set_sensitive {
    name  = "userSettings.dbPassword"
    value = var.rds_password
  }
  set {
    name  = "userSettings.rdsHost"
    value = var.rds_host
  }
  set {
    name  = "userSettings.bucketName"
    value = var.artifacts_bucket
  }
  set {
    name  = "userSettings.accountNumber"
    value = data.aws_caller_identity.current.account_id
  }
  set {
    name  = "userSettings.accountRegion"
    value = var.region
  }
  set {
    name  = "userSettings.certificateArn"
    value = ""
  }
  set {
    name  = "postgres.enabled"
    value = "false"
  }
  set {
    name  = "common.ingress.enabled"
    value = "false"
  }
  set {
    name  = "flyteadmin.serviceAccount.name"
    value = "flyteadmin"
  }
  set {
    name  = "flyteadmin.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.cluster_name}-s3-access"
  }
  set {
    name  = "db.admin.database.username"
    value = "pgadmin"
  }
  set {
    name  = "db.datacatalog.database.username"
    value = "pgadmin"
  }
  set {
    name  = "db.scheduler.database.username"
    value = "pgadmin"
  }
}
