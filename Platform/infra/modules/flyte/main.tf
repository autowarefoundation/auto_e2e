variable "cluster_name" { type = string }
variable "artifacts_bucket" { type = string }
variable "datasets_bucket" {
  description = "Datasets bucket; Flyte publishes immutable packed snapshots here."
  type        = string
}
variable "checkpoints_bucket" {
  description = "Versioned bucket for immutable training checkpoints."
  type        = string
  default     = ""
}
variable "console_dynamo_table_name" {
  description = "Console table receiving overlay and geographic publication pointers."
  type        = string
  default     = "auto-e2e-console"
}
variable "region" { type = string }
variable "rds_host" { type = string }
variable "rds_password" {
  type      = string
  sensitive = true
}

variable "flyte_s3_access_key" {
  description = "Static AWS access key for Flyte S3 (stow library doesn't support IRSA)"
  type        = string
  default     = ""
}

variable "flyte_s3_secret_key" {
  description = "Static AWS secret key for Flyte S3"
  type        = string
  sensitive   = true
  default     = ""
}

variable "oidc_provider_arn" {
  description = "EKS OIDC provider ARN for IRSA"
  type        = string
  default     = ""
}

variable "oidc_provider_url" {
  description = "EKS OIDC provider URL (without https://)"
  type        = string
  default     = ""
}

data "aws_caller_identity" "current" {}

# IAM role assumed by Flyte task pods (default SA in auto-e2e-* namespaces)
# Created by cluster-resource-sync as the defaultIamRole annotation target.
resource "aws_iam_role" "flyte_user" {
  count = var.oidc_provider_url != "" ? 1 : 0
  name  = "flyte-user-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = var.oidc_provider_arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringLike = {
          "${var.oidc_provider_url}:sub" = "system:serviceaccount:auto-e2e-*:default"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "flyte_user_s3" {
  count = var.oidc_provider_url != "" ? 1 : 0
  name  = "s3-access"
  role  = aws_iam_role.flyte_user[0].name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Effect = "Allow"
          Action = "s3:*"
          Resource = [
            "arn:aws:s3:::${var.artifacts_bucket}",
            "arn:aws:s3:::${var.artifacts_bucket}/*",
          ]
        },
        # Packed snapshot publication uses conditional object writes and
        # server-side copies. It never deletes or mutates an existing version.
        {
          Effect = "Allow"
          Action = ["s3:GetObject", "s3:PutObject"]
          Resource = [
            "arn:aws:s3:::${var.datasets_bucket}/*",
          ]
        },
        {
          Effect = "Allow"
          Action = ["s3:ListBucket"]
          Resource = [
            "arn:aws:s3:::${var.datasets_bucket}",
          ]
        },
        {
          Effect = "Allow"
          Action = ["dynamodb:GetItem", "dynamodb:PutItem"]
          Resource = [
            "arn:aws:dynamodb:${var.region}:${data.aws_caller_identity.current.account_id}:table/${var.console_dynamo_table_name}",
          ]
        },
      ],
      var.checkpoints_bucket != "" ? [
        {
          Effect = "Allow"
          Action = ["s3:GetObject", "s3:PutObject"]
          Resource = [
            "arn:aws:s3:::${var.checkpoints_bucket}/*",
          ]
        },
      ] : []
    )
  })
}

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
    name  = "userSettings.rawDataBucketName"
    value = var.artifacts_bucket
  }
  set {
    name  = "userSettings.logGroup"
    value = "/flyte/${var.cluster_name}"
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

  # Storage: custom config with stow + static S3 credentials
  # (flyte-core chart template only supports 'iam' for type=s3)
  # storage.type=custom + stow accesskey config is defined in the values file.
  set {
    name  = "userSettings.s3AccessKey"
    value = var.flyte_s3_access_key
  }
  set_sensitive {
    name  = "storage.custom.stow.config.secret_access_key"
    value = var.flyte_s3_secret_key
  }
}

# Post-apply: patch Flyte control-plane storage configmaps to use accesskey auth.
# The flyte-core chart only renders `iam` auth for type=s3; the stow/minio-go S3
# client (AWS SDK v1) does not support IRSA/Pod Identity, so static keys are required.
# This runs on every apply and whenever the keys change (idempotent kubectl patch).
resource "null_resource" "flyte_storage_accesskey_patch" {
  triggers = {
    access_key = var.flyte_s3_access_key
    secret_sha = sha256(var.flyte_s3_secret_key)
    bucket     = var.artifacts_bucket
    region     = var.region
  }

  provisioner "local-exec" {
    interpreter = ["/bin/bash", "-c"]
    command     = <<-EOT
      set -e
      STORAGE_YAML='storage:
        type: s3
        container: "${var.artifacts_bucket}"
        connection:
          auth-type: accesskey
          access-key: ${var.flyte_s3_access_key}
          secret-key: ${var.flyte_s3_secret_key}
          region: ${var.region}
        enable-multicontainer: false
        limits:
          maxDownloadMBs: 10
        cache:
          max_size_mbs: 1024
          target_gc_percent: 70'

      for cm in flyte-propeller-config flyte-admin-base-config datacatalog-config; do
        kubectl get configmap $cm -n flyte -o json | python3 -c "
import sys,json
cm=json.load(sys.stdin)
cm['data']['storage.yaml']='''$STORAGE_YAML'''
if 'core.yaml' in cm['data'] and '<RAW_DATA_BUCKET_NAME>' in cm['data']['core.yaml']:
    cm['data']['core.yaml']=cm['data']['core.yaml'].replace('s3://<RAW_DATA_BUCKET_NAME>/','s3://${var.artifacts_bucket}/raw-output/')
for k in ['creationTimestamp','resourceVersion','uid','managedFields']:
    cm.get('metadata',{}).pop(k,None)
cm.get('metadata',{}).get('annotations',{}).pop('kubectl.kubernetes.io/last-applied-configuration',None)
print(json.dumps(cm))
" | kubectl replace -f -
      done

      kubectl rollout restart deployment/flytepropeller deployment/flyteadmin deployment/datacatalog -n flyte
    EOT
  }

  depends_on = [helm_release.flyte]
}
