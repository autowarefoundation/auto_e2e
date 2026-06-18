variable "cluster_name" { type = string }
variable "environment" { type = string }

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

resource "aws_iam_role" "codebuild" {
  name = "${var.cluster_name}-codebuild"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "codebuild" {
  name = "codebuild-policy"
  role = aws_iam_role.codebuild.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:GetBucketLocation"]
        Resource = ["${aws_s3_bucket.cache.arn}", "${aws_s3_bucket.cache.arn}/*"]
      },
    ]
  })
}

resource "aws_s3_bucket" "cache" {
  bucket = "${var.cluster_name}-codebuild-cache-${local.account_id}"
  tags   = { Purpose = "codebuild-cache" }
}

resource "aws_codebuild_project" "images" {
  name         = "${var.cluster_name}-build-images"
  service_role = aws_iam_role.codebuild.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  cache {
    type     = "S3"
    location = aws_s3_bucket.cache.bucket
  }

  environment {
    compute_type                = "BUILD_GENERAL1_MEDIUM"
    image                       = "aws/codebuild/amazonlinux-x86_64-standard:5.0"
    type                        = "LINUX_CONTAINER"
    privileged_mode             = true
    image_pull_credentials_type = "CODEBUILD"
  }

  source {
    type      = "S3"
    location  = "${aws_s3_bucket.cache.bucket}/source.zip"
    buildspec = "platform/buildspec.yml"
  }

  logs_config {
    cloudwatch_logs {
      group_name = "/codebuild/${var.cluster_name}-build-images"
    }
  }

  tags = { Purpose = "container-image-build" }
}

output "project_name" {
  value = aws_codebuild_project.images.name
}
