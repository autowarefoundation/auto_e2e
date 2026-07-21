locals {
  # VPC spans 3 AZs for EKS HA; GPU NodePool is pinned to ODCR AZ only
  vpc_azs = ["us-west-2a", "us-west-2b", "us-west-2c"]
  gpu_azs = var.gpu_azs
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

  cluster_name        = var.cluster_name
  artifacts_bucket    = module.storage.bucket_names["artifacts"]
  checkpoints_bucket  = module.storage.bucket_names["checkpoints"]
  datasets_bucket     = module.storage.bucket_names["datasets"]
  region              = var.region
  rds_host            = module.rds.address
  rds_password        = module.rds.master_password
  flyte_s3_access_key = aws_iam_access_key.flyte_s3.id
  flyte_s3_secret_key = aws_iam_access_key.flyte_s3.secret
  oidc_provider_arn   = module.eks.oidc_provider_arn
  oidc_provider_url   = module.eks.oidc_provider_url

  depends_on = [module.rds, module.storage, module.training_operator, module.kueue]
}

# IAM user for Flyte S3 access (stow library doesn't support IRSA/Pod Identity)
resource "aws_iam_user" "flyte_s3" {
  name = "${var.cluster_name}-flyte-s3"
}

resource "aws_iam_user_policy" "flyte_s3" {
  name = "s3-access"
  user = aws_iam_user.flyte_s3.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "s3:*"
      Resource = ["arn:aws:s3:::${module.storage.bucket_names["artifacts"]}", "arn:aws:s3:::${module.storage.bucket_names["artifacts"]}/*"]
    }]
  })
}

resource "aws_iam_access_key" "flyte_s3" {
  user = aws_iam_user.flyte_s3.name
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

  cluster_name              = var.cluster_name
  environment               = var.environment
  vpc_id                    = module.vpc.vpc_id
  private_subnet_ids        = module.vpc.private_subnet_ids
  cluster_security_group_id = module.eks.cluster_security_group_id
}

output "codebuild_project" {
  value = module.codebuild.project_name
}

output "overlay_launch_project" {
  value = module.codebuild.overlay_launch_project
}

# --- UI Exposure: CloudFront + VPC Origin → Internal NLB (K8s managed) ---
# NLB ARNs/DNS are passed as variables since K8s Service creates them.
# After first deploy, run post-apply to create NLB Services, then set these vars.

module "cloudfront" {
  source = "./modules/cloudfront"

  cluster_name           = var.cluster_name
  services               = var.cloudfront_services
  lambda_edge_arn        = var.auth_user_password != "" ? module.auth_edge[0].lambda_arn : ""
  auth_excluded_services = ["mlflow"]
}

module "auth_edge" {
  count  = var.auth_user_password != "" ? 1 : 0
  source = "./modules/auth-edge"

  providers = {
    aws.us_east_1 = aws.us_east_1
  }

  cluster_name  = var.cluster_name
  user_email    = var.auth_user_email
  user_password = var.auth_user_password
  callback_urls = var.auth_callback_urls
}

output "ui_urls" {
  value = module.cloudfront.urls
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

# K8s Secret consumed by Flyte tasks via secret_requests (Secret(group="hf-token",
# key="HF_TOKEN")). Injected as env var by the Flyte pod webhook — never appears
# as a workflow input, so it is not visible in the Flyte/MLflow UI.
resource "kubernetes_secret" "hf_token" {
  count = var.hf_token != "" ? 1 : 0
  metadata {
    name      = "hf-token"
    namespace = "auto-e2e-development"
  }
  data = {
    HF_TOKEN = var.hf_token
  }
  type = "Opaque"

  depends_on = [module.flyte]
}
