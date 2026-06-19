variable "cluster_name" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "environment" { type = string }
variable "cluster_security_group_id" { type = string }

# Allow ALB to reach pods on service ports
resource "aws_security_group_rule" "alb_to_pods_mlflow" {
  type                     = "ingress"
  from_port                = 5000
  to_port                  = 5000
  protocol                 = "tcp"
  security_group_id        = var.cluster_security_group_id
  source_security_group_id = aws_security_group.alb.id
  description              = "ALB to MLflow pods"
}

resource "aws_security_group_rule" "alb_to_pods_flyte" {
  type                     = "ingress"
  from_port                = 8088
  to_port                  = 8088
  protocol                 = "tcp"
  security_group_id        = var.cluster_security_group_id
  source_security_group_id = aws_security_group.alb.id
  description              = "ALB to Flyte admin pods"
}

resource "aws_security_group_rule" "alb_to_pods_console" {
  type                     = "ingress"
  from_port                = 8080
  to_port                  = 8080
  protocol                 = "tcp"
  security_group_id        = var.cluster_security_group_id
  source_security_group_id = aws_security_group.alb.id
  description              = "ALB to Flyte console pods"
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
}

# Lookup CloudFront VPC Origin managed security group (created after VPC Origin deploys)
data "aws_security_group" "cf_vpc_origin" {
  vpc_id = var.vpc_id
  filter {
    name   = "group-name"
    values = ["CloudFront-VPCOrigins-Service-SG"]
  }
}

# Separate rule to avoid cycle (CF SG only exists after VPC Origin is created)
resource "aws_security_group_rule" "alb_from_cf_vpc_origin" {
  type                     = "ingress"
  from_port                = 80
  to_port                  = 80
  protocol                 = "tcp"
  security_group_id        = aws_security_group.alb.id
  source_security_group_id = data.aws_security_group.cf_vpc_origin.id
  description              = "CloudFront VPC Origin managed SG"

  depends_on = [aws_cloudfront_vpc_origin.alb]
}

# --- Internal ALB (private subnet, only CloudFront can reach it) ---

resource "aws_lb" "internal" {
  name               = "${var.cluster_name}-ui"
  internal           = true
  load_balancer_type = "application"
  subnets            = var.private_subnet_ids
  security_groups    = [aws_security_group.alb.id]

  tags = { Name = "${var.cluster_name}-ui-alb" }
}

# SG: only allow traffic from CloudFront VPC Origin ENIs
resource "aws_security_group" "alb" {
  name_prefix = "${var.cluster_name}-ui-alb-"
  vpc_id      = var.vpc_id

  # CloudFront VPC Origin creates ENIs with its own managed SG.
  # Allow from VPC CIDR (covers VPC Origin ENIs in private subnets)
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["10.100.0.0/16"]
    description = "VPC internal (CloudFront VPC Origin ENIs)"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.cluster_name}-ui-alb-sg" }
}

# Listener: forward all to Flyte console (default)
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.internal.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "text/plain"
      message_body = "OK"
      status_code  = "200"
    }
  }
}

# --- CloudFront VPC Origin ---

resource "aws_cloudfront_vpc_origin" "alb" {
  vpc_origin_endpoint_config {
    name                   = "${var.cluster_name}-ui-origin"
    arn                    = aws_lb.internal.arn
    http_port              = 80
    https_port             = 443
    origin_protocol_policy = "http-only"
    origin_ssl_protocols {
      items    = ["TLSv1.2"]
      quantity = 1
    }
  }
}

# --- CloudFront Distribution ---

resource "aws_cloudfront_distribution" "this" {
  enabled         = true
  comment         = "${var.cluster_name} platform UIs (Cognito auth)"
  price_class     = "PriceClass_100"
  is_ipv6_enabled = true

  origin {
    domain_name = aws_lb.internal.dns_name
    origin_id   = "internal-alb"

    vpc_origin_config {
      vpc_origin_id = aws_cloudfront_vpc_origin.alb.id
    }
  }

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "internal-alb"

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Accept", "Content-Type"]
      cookies {
        forward = "all"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = { Name = "${var.cluster_name}-ui" }
}

# --- Cognito (for future Lambda@Edge auth) ---

resource "aws_cognito_user_pool" "this" {
  name = "${var.cluster_name}-users"

  password_policy {
    minimum_length    = 8
    require_lowercase = true
    require_numbers   = true
    require_symbols   = false
    require_uppercase = true
  }

  auto_verified_attributes = ["email"]
}

resource "aws_cognito_user_pool_domain" "this" {
  domain       = "${var.cluster_name}-${local.account_id}"
  user_pool_id = aws_cognito_user_pool.this.id
}

resource "aws_cognito_user_pool_client" "this" {
  name                                 = "${var.cluster_name}-app"
  user_pool_id                         = aws_cognito_user_pool.this.id
  generate_secret                      = true
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email", "profile"]
  callback_urls                        = ["https://${aws_cloudfront_distribution.this.domain_name}/_callback"]
  logout_urls                          = ["https://${aws_cloudfront_distribution.this.domain_name}/"]
  supported_identity_providers         = ["COGNITO"]
}

# --- Outputs ---

output "cloudfront_domain" {
  value = aws_cloudfront_distribution.this.domain_name
}

output "cloudfront_url" {
  value = "https://${aws_cloudfront_distribution.this.domain_name}"
}

output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.this.id
}

output "cognito_domain" {
  value = "${var.cluster_name}-${local.account_id}.auth.${data.aws_region.current.name}.amazoncognito.com"
}

output "alb_arn" {
  value = aws_lb.internal.arn
}

output "alb_dns" {
  value = aws_lb.internal.dns_name
}

data "aws_region" "current" {}
