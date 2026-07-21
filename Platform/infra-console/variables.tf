variable "expected_aws_account_id" {
  type        = string
  description = "Expected Platform AWS account ID; set via TF_VAR_expected_aws_account_id"

  validation {
    condition     = can(regex("^[0-9]{12}$", var.expected_aws_account_id))
    error_message = "expected_aws_account_id must be a 12-digit AWS account ID."
  }
}

variable "aws_region" {
  type    = string
  default = "us-west-2"

  validation {
    condition     = var.aws_region == "us-west-2"
    error_message = "DataModelConsole infrastructure may only be managed in us-west-2."
  }
}

variable "cluster_name" {
  type    = string
  default = "auto-e2e-platform"

  validation {
    condition     = var.cluster_name == "auto-e2e-platform"
    error_message = "DataModelConsole infrastructure may only target auto-e2e-platform."
  }
}

variable "vpc_id" {
  type        = string
  description = "Platform VPC ID; set via TF_VAR_vpc_id"

  validation {
    condition     = can(regex("^vpc-[0-9a-f]+$", var.vpc_id))
    error_message = "vpc_id must be a valid VPC ID."
  }
}

variable "vpc_cidr" {
  type    = string
  default = "10.100.0.0/16"

  validation {
    condition     = var.vpc_cidr == "10.100.0.0/16"
    error_message = "The current auto-e2e-platform VPC requires vpc_cidr 10.100.0.0/16."
  }
}

# ALB listens on this port; CloudFront's VPC origin talks to it over plain HTTP
# (the ALB is internal, so no ACM cert on the ALB — CloudFront terminates the
# viewer TLS with its default *.cloudfront.net cert).
variable "alb_port" {
  type    = number
  default = 80

  validation {
    condition     = var.alb_port == 80
    error_message = "The current CloudFront VPC origin and console Ingress require alb_port 80."
  }
}

# --- Two-phase apply (see README) ---
#
# Phase 1 (deployment_phase == "bootstrap"): creates the ALB security group with a BOOTSTRAP
#   ingress (alb_port from the VPC CIDR — the internal ALB is not
#   internet-reachable) plus the Pod Identity IAM role. Deploy the k8s Ingress
#   next; the ALB controller creates the internal ALB and attaches this SG.
#
# Phase 2 (deployment_phase == "locked", with alb_arn/alb_dns): creates the VPC
#   origin + CloudFront distribution, and SWAPS the SG ingress to admit alb_port
#   ONLY from CloudFront's managed "CloudFront-VPCOrigins-Service-SG" (which the
#   VPC origin provisions), looked up via data source. End state: nothing but
#   CloudFront's VPC-origin ENIs can reach the ALB.
variable "deployment_phase" {
  type        = string
  description = "Explicit deployment phase: bootstrap for initial ALB creation, locked for the CloudFront-only production state"

  validation {
    condition     = contains(["bootstrap", "locked"], var.deployment_phase)
    error_message = "deployment_phase must be explicitly set to bootstrap or locked."
  }
}

variable "alb_arn" {
  type    = string
  default = ""

  validation {
    condition     = var.alb_arn == "" || can(regex("^arn:aws:elasticloadbalancing:[a-z0-9-]+:[0-9]{12}:loadbalancer/app/", var.alb_arn))
    error_message = "alb_arn must be empty for bootstrap or a valid application load balancer ARN."
  }
}

variable "alb_dns" {
  type    = string
  default = ""

  validation {
    condition     = var.alb_dns == "" || can(regex("^[A-Za-z0-9.-]+\\.elb\\.amazonaws\\.com$", var.alb_dns))
    error_message = "alb_dns must be empty for bootstrap or a valid ELB DNS name."
  }
}
