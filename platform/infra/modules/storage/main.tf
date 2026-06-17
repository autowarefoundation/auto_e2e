variable "cluster_name" { type = string }
variable "oidc_provider_arn" { type = string }
variable "oidc_provider_url" { type = string }
variable "environment" { type = string }

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

# IRSA: allow Pods in the EKS cluster to read/write these buckets
resource "aws_iam_role" "s3_access" {
  name = "${var.cluster_name}-s3-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Federated = var.oidc_provider_arn }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringLike = {
          "${var.oidc_provider_url}:sub" = "system:serviceaccount:*:*"
        }
      }
    }]
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

output "bucket_names" {
  value = { for k, b in aws_s3_bucket.this : k => b.bucket }
}

output "s3_access_role_arn" {
  value = aws_iam_role.s3_access.arn
}
