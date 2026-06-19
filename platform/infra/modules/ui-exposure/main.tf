variable "cluster_name" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "environment" { type = string }

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
}

# --- Internal ALB ---

resource "aws_lb" "internal" {
  name               = "${var.cluster_name}-internal"
  internal           = true
  load_balancer_type = "application"
  subnets            = var.private_subnet_ids
  security_groups    = [aws_security_group.alb.id]

  tags = { Name = "${var.cluster_name}-internal-alb" }
}

resource "aws_security_group" "alb" {
  name_prefix = "${var.cluster_name}-alb-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]  # VPC + CloudFront VPC origin
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# MLflow target group
resource "aws_lb_target_group" "mlflow" {
  name        = "${var.cluster_name}-mlflow"
  port        = 5000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path     = "/health"
    port     = "5000"
    matcher  = "200,403"  # MLflow returns 403 for non-localhost
  }
}

# Flyte target group
resource "aws_lb_target_group" "flyte" {
  name        = "${var.cluster_name}-flyte"
  port        = 80
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path    = "/"
    port    = "80"
    matcher = "200,301,302,404"
  }
}

# Listener with path-based routing
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.internal.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }
}

resource "aws_lb_listener_rule" "flyte" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.flyte.arn
  }

  condition {
    path_pattern { values = ["/console*", "/flyte*", "/api/v1/*"] }
  }
}

# --- Cognito ---

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

# --- CloudFront + VPC Origin ---

resource "aws_cloudfront_vpc_origin" "alb" {
  vpc_origin_endpoint_config {
    name                   = "${var.cluster_name}-alb-origin"
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

resource "aws_cloudfront_distribution" "this" {
  enabled         = true
  comment         = "${var.cluster_name} platform UIs"
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
      headers      = ["Host", "Authorization"]
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

# --- Outputs ---

output "cloudfront_domain" {
  value = aws_cloudfront_distribution.this.domain_name
}

output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.this.id
}

output "alb_dns" {
  value = aws_lb.internal.dns_name
}
