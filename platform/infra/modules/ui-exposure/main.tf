variable "cluster_name" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "environment" { type = string }

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
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

# --- Outputs (ALB created by K8s Ingress, CloudFront added later if needed) ---

output "cognito_user_pool_id" {
  value = aws_cognito_user_pool.this.id
}

output "cognito_domain" {
  value = aws_cognito_user_pool_domain.this.domain
}
