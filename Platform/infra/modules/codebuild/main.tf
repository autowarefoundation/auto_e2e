variable "cluster_name" { type = string }
variable "environment" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "cluster_security_group_id" { type = string }

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
          "ecr:DescribeImages",
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
    buildspec = "Platform/buildspec.yml"
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

# --- Flyte Register (VPC内, flyteadminにアクセス可能) ---

resource "aws_security_group" "flyte_register" {
  name_prefix = "${var.cluster_name}-flyte-register-"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.cluster_name}-flyte-register-sg" }
}

resource "aws_iam_role_policy" "flyte_register_vpc" {
  name = "flyte-register-vpc"
  role = aws_iam_role.codebuild.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ec2:CreateNetworkInterface",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DeleteNetworkInterface",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeDhcpOptions",
        "ec2:DescribeVpcs",
        "ec2:CreateNetworkInterfacePermission",
      ]
      Resource = "*"
    }]
  })
}

resource "aws_codebuild_project" "flyte_register" {
  name         = "${var.cluster_name}-flyte-register"
  service_role = aws_iam_role.codebuild.arn

  artifacts { type = "NO_ARTIFACTS" }

  environment {
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/amazonlinux-x86_64-standard:5.0"
    type                        = "LINUX_CONTAINER"
    image_pull_credentials_type = "CODEBUILD"
  }

  source {
    type      = "S3"
    location  = "${aws_s3_bucket.cache.bucket}/source.zip"
    buildspec = "Platform/buildspec-register.yml"
  }

  vpc_config {
    vpc_id             = var.vpc_id
    subnets            = var.private_subnet_ids
    security_group_ids = [aws_security_group.flyte_register.id]
  }

  logs_config {
    cloudwatch_logs {
      group_name = "/codebuild/${var.cluster_name}-flyte-register"
    }
  }

  tags = { Purpose = "flyte-workflow-register" }
}

output "flyte_register_project" {
  value = aws_codebuild_project.flyte_register.name
}

# --- Trajectory overlay launch (VPC-local Flyte client) ---

resource "aws_codebuild_project" "overlay_launch" {
  name         = "${var.cluster_name}-overlay-launch"
  service_role = aws_iam_role.codebuild.arn

  artifacts { type = "NO_ARTIFACTS" }

  environment {
    compute_type                = "BUILD_GENERAL1_SMALL"
    image                       = "aws/codebuild/amazonlinux-x86_64-standard:5.0"
    type                        = "LINUX_CONTAINER"
    image_pull_credentials_type = "CODEBUILD"
  }

  source {
    type      = "S3"
    location  = "${aws_s3_bucket.cache.bucket}/source.zip"
    buildspec = "Platform/buildspec-launch-overlay.yml"
  }

  vpc_config {
    vpc_id             = var.vpc_id
    subnets            = var.private_subnet_ids
    security_group_ids = [aws_security_group.flyte_register.id]
  }

  logs_config {
    cloudwatch_logs {
      group_name = "/codebuild/${var.cluster_name}-overlay-launch"
    }
  }

  tags = { Purpose = "trajectory-overlay-launch" }
}

output "overlay_launch_project" {
  value = aws_codebuild_project.overlay_launch.name
}

# Allow CodeBuild flyte-register to reach flyteadmin (gRPC port 81)
resource "aws_security_group_rule" "codebuild_to_flyteadmin" {
  type                     = "ingress"
  from_port                = 81
  to_port                  = 81
  protocol                 = "tcp"
  security_group_id        = var.cluster_security_group_id
  source_security_group_id = aws_security_group.flyte_register.id
  description              = "CodeBuild flyte-register to flyteadmin gRPC"
}
