terraform {
  required_providers {
    aws = {
      source                = "hashicorp/aws"
      configuration_aliases = [aws.us_east_1]
    }
  }
}

variable "cluster_name" { type = string }
variable "user_email" { type = string }
variable "user_password" {
  type      = string
  sensitive = true
}
variable "callback_urls" {
  description = "Callback URLs for each CF distribution (https://<domain>/_callback)"
  type        = list(string)
}

# --- Cognito (us-east-1 for Lambda@Edge cookie domain compatibility) ---

resource "aws_cognito_user_pool" "this" {
  provider = aws.us_east_1
  name     = "${var.cluster_name}-auth"

  auto_verified_attributes = []
  username_attributes      = ["email"]

  password_policy {
    minimum_length    = 8
    require_lowercase = false
    require_uppercase = false
    require_numbers   = false
    require_symbols   = false
  }

  admin_create_user_config {
    allow_admin_create_user_only = true
  }

  schema {
    name                = "email"
    attribute_data_type = "String"
    required            = true
    mutable             = true
    string_attribute_constraints {
      min_length = 1
      max_length = 256
    }
  }
}

resource "aws_cognito_user_pool_domain" "this" {
  provider     = aws.us_east_1
  domain       = "${var.cluster_name}-auth"
  user_pool_id = aws_cognito_user_pool.this.id
}

resource "aws_cognito_user_pool_client" "this" {
  provider     = aws.us_east_1
  name         = "${var.cluster_name}-cloudfront"
  user_pool_id = aws_cognito_user_pool.this.id

  generate_secret                      = true
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email"]
  supported_identity_providers         = ["COGNITO"]
  callback_urls                        = concat(var.callback_urls, [for url in var.callback_urls : replace(url, "/_callback", "")])
  logout_urls                          = [for url in var.callback_urls : replace(url, "/_callback", "")]

  explicit_auth_flows = [
    "ALLOW_ADMIN_USER_PASSWORD_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
  ]
}

resource "aws_cognito_user" "admin" {
  provider     = aws.us_east_1
  user_pool_id = aws_cognito_user_pool.this.id
  username     = var.user_email
  password     = var.user_password

  attributes = {
    email          = var.user_email
    email_verified = "true"
  }

  message_action = "SUPPRESS"
}

# --- Lambda@Edge (must be us-east-1) ---

locals {
  lambda_source = templatefile("${path.module}/lambda/index.js.tpl", {
    region         = "us-east-1"
    user_pool_id   = aws_cognito_user_pool.this.id
    client_id      = aws_cognito_user_pool_client.this.id
    client_secret  = aws_cognito_user_pool_client.this.client_secret
    cognito_domain = "${aws_cognito_user_pool_domain.this.domain}.auth.us-east-1.amazoncognito.com"
  })
}

data "archive_file" "lambda" {
  type        = "zip"
  output_path = "${path.module}/lambda/auth-edge.zip"

  source {
    content  = local.lambda_source
    filename = "index.js"
  }
}

# Kept for backwards compat with state (no-op)
resource "null_resource" "npm_install" {
  triggers = {
    source = sha256(local.lambda_source)
  }

  provisioner "local-exec" {
    command = "true"
  }

  depends_on = []
}

resource "aws_iam_role" "lambda_edge" {
  provider = aws.us_east_1
  name     = "${var.cluster_name}-auth-edge"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = ["lambda.amazonaws.com", "edgelambda.amazonaws.com"]
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  provider   = aws.us_east_1
  role       = aws_iam_role.lambda_edge.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_lambda_function" "auth_edge" {
  provider         = aws.us_east_1
  function_name    = "${var.cluster_name}-auth-edge"
  role             = aws_iam_role.lambda_edge.arn
  handler          = "index.handler"
  runtime          = "nodejs20.x"
  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256
  publish          = true # Lambda@Edge requires published version
  timeout          = 30
  memory_size      = 128
}

output "lambda_arn" {
  value = aws_lambda_function.auth_edge.qualified_arn
}

output "cognito_domain" {
  value = "${aws_cognito_user_pool_domain.this.domain}.auth.us-east-1.amazoncognito.com"
}
