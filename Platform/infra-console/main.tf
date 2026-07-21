# DataModelConsole-dedicated infra: an internal ALB security group locked to
# CloudFront, plus a CloudFront distribution that fronts the console. ISOLATED
# state (see versions.tf) so this never touches existing Platform resources.
#
# All cluster subnets are PRIVATE, so the ALB the k8s Ingress creates is internal
# and NOT internet-reachable. CloudFront reaches it through a VPC ORIGIN — it
# places an AWS-managed ENI inside this VPC and connects from that private
# address, NOT from public edge IPs. Viewers hit CloudFront over HTTPS (default
# *.cloudfront.net cert); CloudFront→ALB is plain HTTP on alb_port. No ACM /
# Cognito / Lambda@Edge for this dashboard-only deploy.

locals {
  locked_phase = var.deployment_phase == "locked"
}

data "aws_eks_cluster" "target" {
  name = var.cluster_name
}

# --- ALB security group: the gate that enforces "CloudFront only" -----------
resource "aws_security_group" "console_alb" {
  name_prefix = "${var.cluster_name}-console-alb-"
  description = "Console ALB - CloudFront VPC origin only"
  vpc_id      = var.vpc_id

  tags = { Name = "${var.cluster_name}-console-alb-sg", Service = "DataModelConsole" }

  lifecycle {
    create_before_destroy = true
    prevent_destroy       = true

    precondition {
      condition = (
        var.deployment_phase == "bootstrap"
        ? var.alb_arn == "" && var.alb_dns == ""
        : startswith(
          var.alb_arn,
          "arn:aws:elasticloadbalancing:${var.aws_region}:${var.expected_aws_account_id}:loadbalancer/app/"
        ) && can(regex("\\.${var.aws_region}\\.elb\\.amazonaws\\.com$", var.alb_dns))
      )
      error_message = "bootstrap requires empty alb_arn/alb_dns; locked requires this account and region's ALB ARN and DNS name."
    }

    precondition {
      condition = (
        data.aws_eks_cluster.target.status == "ACTIVE" &&
        var.vpc_id == data.aws_eks_cluster.target.vpc_config[0].vpc_id
      )
      error_message = "The target EKS cluster must be ACTIVE and belong to vpc_id in the expected account and region."
    }
  }
}

# The service-managed SG AWS creates for CloudFront VPC origins. It only exists
# once a VPC origin has been created in this VPC (Phase 2), so the lookup is
# gated on the VPC origin and skipped entirely in Phase 1.
data "aws_security_group" "cloudfront_vpc_origin" {
  count = local.locked_phase ? 1 : 0

  filter {
    name   = "group-name"
    values = ["CloudFront-VPCOrigins-Service-SG"]
  }
  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }
  depends_on = [aws_cloudfront_vpc_origin.console]
}

# Ingress. Phase 1 (bootstrap): admit alb_port from the VPC CIDR so the internal
# ALB is reachable in-VPC before the managed SG exists (still not
# internet-exposed — internal scheme). Phase 2 (locked): admit alb_port ONLY
# from CloudFront's managed VPC-origin ENIs; nothing else in the VPC can reach
# the ALB.
resource "aws_vpc_security_group_ingress_rule" "from_cloudfront_sg" {
  count                        = local.locked_phase ? 1 : 0
  security_group_id            = aws_security_group.console_alb.id
  description                  = "console ALB: CloudFront VPC-origin ENIs only"
  from_port                    = var.alb_port
  to_port                      = var.alb_port
  ip_protocol                  = "tcp"
  referenced_security_group_id = data.aws_security_group.cloudfront_vpc_origin[0].id

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_vpc_security_group_ingress_rule" "from_vpc_bootstrap" {
  count             = local.locked_phase ? 0 : 1
  security_group_id = aws_security_group.console_alb.id
  description       = "console ALB bootstrap: VPC CIDR on alb_port (internal ALB)"
  from_port         = var.alb_port
  to_port           = var.alb_port
  ip_protocol       = "tcp"
  cidr_ipv4         = var.vpc_cidr
}

# Egress: ALB → target pods anywhere in the VPC.
resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.console_alb.id
  description       = "allow all egress"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

# --- CloudFront (Phase 2: only once the ALB exists) -------------------------
resource "aws_cloudfront_vpc_origin" "console" {
  count = local.locked_phase ? 1 : 0

  vpc_origin_endpoint_config {
    name                   = "${var.cluster_name}-console"
    arn                    = var.alb_arn
    http_port              = var.alb_port
    https_port             = 443
    origin_protocol_policy = "http-only" # internal ALB is HTTP; CF terminates viewer TLS

    origin_ssl_protocols {
      items    = ["TLSv1.2"]
      quantity = 1
    }
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_cloudfront_distribution" "console" {
  count = local.locked_phase ? 1 : 0

  comment         = "DataModelConsole"
  enabled         = true
  is_ipv6_enabled = true
  price_class     = "PriceClass_200"

  origin {
    domain_name = var.alb_dns
    origin_id   = "console-alb"

    vpc_origin_config {
      vpc_origin_id            = aws_cloudfront_vpc_origin.console[0].id
      origin_read_timeout      = 30
      origin_keepalive_timeout = 5
    }
  }

  # Dashboard-only: no edge auth, all methods forwarded (the API drives the UI),
  # caching disabled so the dynamic dashboard is never served stale. AllViewer
  # forwards headers/cookies/query so same-origin /api works unchanged.
  default_cache_behavior {
    target_origin_id       = "console-alb"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]

    cache_policy_id          = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad" # Managed-CachingDisabled
    origin_request_policy_id = "216adef6-5c7f-47e4-b989-5492eafa07d3" # Managed-AllViewer
    compress                 = true
  }

  restrictions {
    geo_restriction { restriction_type = "none" }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = { Service = "DataModelConsole" }

  lifecycle {
    prevent_destroy = true
  }
}

output "console_alb_sg_id" {
  value = aws_security_group.console_alb.id
}

output "cloudfront_url" {
  value = length(aws_cloudfront_distribution.console) > 0 ? "https://${aws_cloudfront_distribution.console[0].domain_name}" : "(Phase 2: set alb_arn/alb_dns after the Ingress creates the ALB)"
}

output "cloudfront_distribution_id" {
  value = length(aws_cloudfront_distribution.console) > 0 ? aws_cloudfront_distribution.console[0].id : ""
}
