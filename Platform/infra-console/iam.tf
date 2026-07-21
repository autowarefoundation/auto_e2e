# Pod Identity for console-api: read-only S3 (datasets + artifacts) and the
# single DynamoDB cache table. The k8s ServiceAccount console/console-api has NO
# role-arn annotation — Pod Identity binds the role to the SA here. Without this
# the dashboard's API pods cannot read shards/stats and the UI is non-functional,
# so it belongs in the dashboard-only deploy (unlike auth/Cognito, which do not).

data "aws_caller_identity" "current" {}

variable "datasets_bucket_name" {
  type        = string
  default     = null
  description = "Optional override; defaults to the Platform datasets bucket in expected_aws_account_id"
}

variable "artifacts_bucket_name" {
  type        = string
  default     = null
  description = "Optional override; defaults to the Platform artifacts bucket in expected_aws_account_id"
}

variable "dynamo_table_name" {
  type    = string
  default = "auto-e2e-console"
}

locals {
  datasets_bucket_name  = coalesce(var.datasets_bucket_name, "${var.cluster_name}-datasets-${var.expected_aws_account_id}")
  artifacts_bucket_name = coalesce(var.artifacts_bucket_name, "${var.cluster_name}-artifacts-${var.expected_aws_account_id}")
}

resource "aws_iam_role" "console_api" {
  name = "${var.cluster_name}-console-api"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "pods.eks.amazonaws.com" }
      Action    = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })

  tags = { Service = "DataModelConsole" }

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_iam_role_policy" "console_api_s3_readonly" {
  name = "s3-readonly"
  role = aws_iam_role.console_api.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"]
      Resource = [
        "arn:aws:s3:::${local.datasets_bucket_name}",
        "arn:aws:s3:::${local.datasets_bucket_name}/*",
        "arn:aws:s3:::${local.artifacts_bucket_name}",
        "arn:aws:s3:::${local.artifacts_bucket_name}/*",
      ]
    }]
  })
}

# Least-privilege on the single console table + its gsi1. No DeleteItem / no
# table admin.
resource "aws_iam_role_policy" "console_api_dynamo" {
  name = "dynamo-cache"
  role = aws_iam_role.console_api.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:BatchWriteItem", "dynamodb:Query"]
      Resource = [
        "arn:aws:dynamodb:${var.aws_region}:${data.aws_caller_identity.current.account_id}:table/${var.dynamo_table_name}",
        "arn:aws:dynamodb:${var.aws_region}:${data.aws_caller_identity.current.account_id}:table/${var.dynamo_table_name}/index/gsi1",
      ]
    }]
  })
}

resource "aws_eks_pod_identity_association" "console_api" {
  cluster_name    = var.cluster_name
  namespace       = "console"
  service_account = "console-api"
  role_arn        = aws_iam_role.console_api.arn

  lifecycle {
    prevent_destroy = true
  }
}

output "console_api_role_arn" {
  value = aws_iam_role.console_api.arn
}
