locals {
  # VPC spans 3 AZs for EKS HA; GPU NodePool is pinned to ODCR AZ only
  vpc_azs  = ["us-west-2a", "us-west-2b", "us-west-2c"]
  gpu_azs  = var.gpu_azs
}

module "vpc" {
  source = "./modules/vpc"

  name        = var.cluster_name
  cidr        = var.vpc_cidr
  azs         = local.vpc_azs
  environment = var.environment
}

module "eks" {
  source = "./modules/eks"

  cluster_name       = var.cluster_name
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  gpu_instance_types = var.gpu_instance_types
  gpu_azs            = local.gpu_azs
  environment        = var.environment
}

module "storage" {
  source = "./modules/storage"

  cluster_name = var.cluster_name
  environment  = var.environment
}

module "ecr" {
  source = "./modules/ecr"

  environment = var.environment
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "ecr_repositories" {
  value = module.ecr.repository_urls
}

output "s3_buckets" {
  value = module.storage.bucket_names
}
