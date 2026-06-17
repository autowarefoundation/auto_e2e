terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.13"
    }
  }

  backend "s3" {
    bucket         = "auto-e2e-platform-tfstate"
    key            = "infra/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "auto-e2e-platform-tflock"
    encrypt        = true
    profile        = "autowarefoundation"
  }
}

provider "aws" {
  region  = var.region
  profile = "autowarefoundation"
  # us-west-2: ODCR confirmed for g6e.4xlarge @ us-west-2b

  default_tags {
    tags = {
      Project     = "auto-e2e-platform"
      ManagedBy   = "terraform"
      Environment = var.environment
    }
  }
}
