terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "cluster_name" {
  default = "auto-e2e-platform"
}

variable "vpc_id" {
  type = string
}

variable "console_alb_arn" {
  type        = string
  description = "ARN of the internal ALB created by the K8s Ingress controller"
}

variable "console_alb_dns" {
  type        = string
  description = "DNS name of the internal ALB"
}

variable "cognito_user_pool_id" {
  type = string
}

variable "cognito_client_id" {
  type = string
}

variable "auth_lambda_arn" {
  type        = string
  description = "ARN of the existing auth-edge Lambda@Edge function"
}

variable "acm_cert_arn_us_east_1" {
  type        = string
  description = "ACM certificate ARN in us-east-1 for CloudFront"
}

variable "datasets_bucket_name" {
  default = "auto-e2e-platform-datasets-381491877296"
}

variable "artifacts_bucket_name" {
  default = "auto-e2e-platform-artifacts-381491877296"
}

data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

# Security Group: ALB only accepts traffic from CloudFront
resource "aws_security_group" "console_alb" {
  name_prefix = "${var.cluster_name}-console-alb-"
  description = "Console ALB - CloudFront origin-facing only"
  vpc_id      = var.vpc_id

  ingress {
    description     = "HTTPS from CloudFront only"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-console-alb-sg"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# CloudFront VPC Origin
resource "aws_cloudfront_vpc_origin" "console" {
  vpc_origin_endpoint_config {
    name                   = "console-alb"
    arn                    = var.console_alb_arn
    http_port              = 80
    https_port             = 443
    origin_protocol_policy = "https-only"

    origin_ssl_protocols {
      items    = ["TLSv1.2"]
      quantity = 1
    }
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "console" {
  enabled         = true
  is_ipv6_enabled = true
  comment         = "DataModelConsole"
  # No default_root_object: Next.js serves "/" from the web pod. CloudFront
  # rejects a leading-slash value like "/" here, so leave it unset.
  price_class = "PriceClass_200"

  origin {
    domain_name = var.console_alb_dns
    origin_id   = "console-alb"

    vpc_origin_config {
      vpc_origin_id            = aws_cloudfront_vpc_origin.console.id
      origin_keepalive_timeout = 5
      origin_read_timeout      = 30
    }
  }

  default_cache_behavior {
    # Phase 1 API is read-only; do not let write verbs reach the origin.
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "console-alb"
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Host", "Origin"]
      cookies {
        forward = "all"
      }
    }

    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = var.auth_lambda_arn
      include_body = false
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 86400
  }

  # Streamed camera JPEGs are immutable content — cache them at the edge.
  # query_string IS part of the cache key so ?presign=true (a short-lived
  # 15-min JSON URL) and the plain JPEG get DISTINCT cache entries; the presign
  # branch sends Cache-Control: no-store (see datasets.go) so it is never
  # cached past its expiry. Lambda@Edge auth MUST be repeated here — CloudFront
  # associations are per-behavior, so omitting it would leave this path
  # (including ?presign=true → whole-shard URL) open to anonymous callers.
  ordered_cache_behavior {
    path_pattern           = "/api/v1/datasets/*/image/*"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "console-alb"
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = true
      headers      = ["Authorization"]
      cookies {
        forward = "none"
      }
    }

    lambda_function_association {
      event_type   = "viewer-request"
      lambda_arn   = var.auth_lambda_arn
      include_body = false
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = var.acm_cert_arn_us_east_1
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Service = "DataModelConsole"
  }
}

# IAM Role for Console API Pod Identity (read-only S3)
resource "aws_iam_role" "console_api" {
  name = "${var.cluster_name}-console-api"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "pods.eks.amazonaws.com"
        }
        Action = ["sts:AssumeRole", "sts:TagSession"]
      }
    ]
  })
}

resource "aws_iam_role_policy" "console_api_s3_readonly" {
  name = "s3-readonly"
  role = aws_iam_role.console_api.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          "arn:aws:s3:::${var.datasets_bucket_name}",
          "arn:aws:s3:::${var.datasets_bucket_name}/*",
          "arn:aws:s3:::${var.artifacts_bucket_name}",
          "arn:aws:s3:::${var.artifacts_bucket_name}/*"
        ]
      }
    ]
  })
}

resource "aws_eks_pod_identity_association" "console_api" {
  cluster_name    = var.cluster_name
  namespace       = "console"
  service_account = "console-api"
  role_arn        = aws_iam_role.console_api.arn
}

output "cloudfront_domain" {
  value = aws_cloudfront_distribution.console.domain_name
}

output "cloudfront_distribution_id" {
  value = aws_cloudfront_distribution.console.id
}

output "console_alb_sg_id" {
  value = aws_security_group.console_alb.id
}

output "console_api_role_arn" {
  value = aws_iam_role.console_api.arn
}
