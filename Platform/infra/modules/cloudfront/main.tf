variable "services" {
  description = "Map of service name to NLB ARN and DNS"
  type = map(object({
    nlb_arn = string
    nlb_dns = string
  }))
}

variable "cluster_name" { type = string }

variable "lambda_edge_arn" {
  description = "Lambda@Edge qualified ARN for viewer-request auth (optional)"
  type        = string
  default     = ""
}

variable "auth_excluded_services" {
  description = "Services to exclude from Lambda@Edge auth"
  type        = list(string)
  default     = []
}

resource "aws_cloudfront_vpc_origin" "this" {
  for_each = var.services

  vpc_origin_endpoint_config {
    name                   = "${each.key}-v2"
    arn                    = each.value.nlb_arn
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
  for_each = var.services

  comment         = "AutoE2E ${each.key}"
  enabled         = true
  is_ipv6_enabled = true

  origin {
    domain_name = each.value.nlb_dns
    origin_id   = each.key

    vpc_origin_config {
      vpc_origin_id            = aws_cloudfront_vpc_origin.this[each.key].id
      origin_read_timeout      = 30
      origin_keepalive_timeout = 5
    }
  }

  default_cache_behavior {
    target_origin_id       = each.key
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]

    cache_policy_id          = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad" # CachingDisabled
    origin_request_policy_id = "216adef6-5c7f-47e4-b989-5492eafa07d3" # AllViewer
    compress                 = true

    dynamic "lambda_function_association" {
      for_each = var.lambda_edge_arn != "" && !contains(var.auth_excluded_services, each.key) ? [1] : []
      content {
        event_type   = "viewer-request"
        lambda_arn   = var.lambda_edge_arn
        include_body = false
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = { Service = each.key }
}

output "urls" {
  value = { for k, v in aws_cloudfront_distribution.this : k => "https://${v.domain_name}" }
}

output "distribution_ids" {
  value = { for k, v in aws_cloudfront_distribution.this : k => v.id }
}

output "domain_names" {
  value = { for k, v in aws_cloudfront_distribution.this : k => v.domain_name }
}
