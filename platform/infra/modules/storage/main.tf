variable "cluster_name" { type = string }
variable "environment" { type = string }

# Pod Identity associations: each entry maps one (namespace, service_account)
# pair to the s3-access IAM role.  Add a row here when a new component (Flyte,
# MLflow, data-prep, ...) needs S3 access — no trust-policy edits required.
variable "pod_identity_associations" {
  description = "List of {namespace, service_account} that get the s3-access role."
  type = list(object({
    namespace       = string
    service_account = string
  }))
  default = [
    { namespace = "auto-e2e-training", service_account = "training-sa" },
    { namespace = "flyte", service_account = "flyteadmin" },
    { namespace = "mlflow", service_account = "mlflow" },
  ]
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  buckets = {
    datasets    = "${var.cluster_name}-datasets-${local.account_id}"
    checkpoints = "${var.cluster_name}-checkpoints-${local.account_id}"
    artifacts   = "${var.cluster_name}-artifacts-${local.account_id}"
  }
}

resource "aws_s3_bucket" "this" {
  for_each = local.buckets
  bucket   = each.value

  tags = { Name = each.value, Purpose = each.key }
}

resource "aws_s3_bucket_versioning" "checkpoints" {
  bucket = aws_s3_bucket.this["checkpoints"].id
  versioning_configuration { status = "Enabled" }
}

# Pod Identity: IAM role whose trust principal is the EKS Pod Identity service.
# No OIDC Provider, no per-SA annotations.  Associations below bind it to
# specific (namespace, service_account) pairs — explicit by construction.
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

resource "aws_iam_role" "s3_access" {
  name = "${var.cluster_name}-s3-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [{
        Effect    = "Allow"
        Principal = { Service = "pods.eks.amazonaws.com" }
        Action    = ["sts:AssumeRole", "sts:TagSession"]
      }],
      var.oidc_provider_arn != "" ? [{
        Effect    = "Allow"
        Principal = { Federated = var.oidc_provider_arn }
        Action    = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringLike = {
            "${var.oidc_provider_url}:sub" = "system:serviceaccount:flyte:flyteadmin"
          }
        }
      }] : []
    )
  })
}

resource "aws_iam_role_policy" "s3_access" {
  name = "s3-readwrite"
  role = aws_iam_role.s3_access.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "s3:DeleteObject",
      ]
      Resource = flatten([
        for b in aws_s3_bucket.this : [b.arn, "${b.arn}/*"]
      ])
    }]
  })
}

# Pod Identity associations — one per (namespace, service_account) that needs
# S3 access.  Extend pod_identity_associations variable when adding components.
resource "aws_eks_pod_identity_association" "s3_access" {
  for_each = {
    for a in var.pod_identity_associations : "${a.namespace}/${a.service_account}" => a
  }

  cluster_name    = var.cluster_name
  namespace       = each.value.namespace
  service_account = each.value.service_account
  role_arn        = aws_iam_role.s3_access.arn
}

output "bucket_names" {
  value = { for k, b in aws_s3_bucket.this : k => b.bucket }
}

output "s3_access_role_arn" {
  value = aws_iam_role.s3_access.arn
}

# Mountpoint for S3 CSI driver — enables Flyte ingest pods to write directly
# to S3 via filesystem mount. NOT used for training DataLoader (WebDataset + EBS).
resource "aws_eks_addon" "s3_csi" {
  cluster_name = var.cluster_name
  addon_name   = "aws-mountpoint-s3-csi-driver"
}
