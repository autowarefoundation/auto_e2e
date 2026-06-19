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

  cluster_name      = var.cluster_name
  environment       = var.environment
  oidc_provider_arn = module.eks.oidc_provider_arn
  oidc_provider_url = module.eks.oidc_provider_url
}

module "ecr" {
  source = "./modules/ecr"

  environment = var.environment
}

# --- Phase 2: Queue + Orchestration + Tracking ---

module "rds" {
  source = "./modules/rds"

  cluster_name                  = var.cluster_name
  vpc_id                        = module.vpc.vpc_id
  private_subnet_ids            = module.vpc.private_subnet_ids
  cluster_security_group_id     = module.eks.cluster_security_group_id
  eks_cluster_security_group_id = data.aws_eks_cluster.this.vpc_config[0].cluster_security_group_id
  environment                   = var.environment
}

module "training_operator" {
  source = "./modules/training-operator"

  cluster_name = var.cluster_name

  depends_on = [module.eks]
}

module "kueue" {
  source = "./modules/kueue"

  cluster_name = var.cluster_name

  depends_on = [module.training_operator]
}

module "mlflow" {
  source = "./modules/mlflow"

  cluster_name     = var.cluster_name
  artifacts_bucket = module.storage.bucket_names["artifacts"]
  region           = var.region
  rds_host         = module.rds.address
  rds_password     = module.rds.master_password

  depends_on = [module.rds, module.storage]
}

module "flyte" {
  source = "./modules/flyte"

  cluster_name     = var.cluster_name
  artifacts_bucket = module.storage.bucket_names["artifacts"]
  region           = var.region
  rds_host         = module.rds.address
  rds_password     = module.rds.master_password

  depends_on = [module.rds, module.storage, module.training_operator, module.kueue]
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

output "rds_endpoint" {
  value = module.rds.endpoint
}

# --- Phase 3: Data Pipeline ---

module "codebuild" {
  source = "./modules/codebuild"

  cluster_name = var.cluster_name
  environment  = var.environment
}

output "codebuild_project" {
  value = module.codebuild.project_name
}

# --- UI Exposure: CloudFront + Cognito ---

module "ui_exposure" {
  source = "./modules/ui-exposure"

  cluster_name       = var.cluster_name
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  environment        = var.environment

  depends_on = [module.mlflow, module.flyte]
}

output "cognito_user_pool_id" {
  value = module.ui_exposure.cognito_user_pool_id
}

# HF_TOKEN → Secrets Manager → K8s Secret (for gated dataset access)
resource "aws_secretsmanager_secret" "hf_token" {
  count                   = var.hf_token != "" ? 1 : 0
  name                    = "${var.cluster_name}/hf-token"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "hf_token" {
  count         = var.hf_token != "" ? 1 : 0
  secret_id     = aws_secretsmanager_secret.hf_token[0].id
  secret_string = var.hf_token
}

# K8s Secret is created by post-apply script (reads from Secrets Manager)
