variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "cluster_name" {
  type    = string
  default = "auto-e2e-platform"
}

variable "vpc_id" {
  type    = string
  default = "vpc-06e99faf8815982b7" # auto-e2e-platform VPC (us-west-2)
}

variable "vpc_cidr" {
  type    = string
  default = "10.100.0.0/16"
}

# ALB listens on this port; CloudFront's VPC origin talks to it over plain HTTP
# (the ALB is internal, so no ACM cert on the ALB — CloudFront terminates the
# viewer TLS with its default *.cloudfront.net cert).
variable "alb_port" {
  type    = number
  default = 80
}

# --- Two-phase apply (see README) ---
#
# Phase 1 (alb_arn == ""): creates the ALB security group with a BOOTSTRAP
#   ingress (alb_port from the VPC CIDR — the internal ALB is not
#   internet-reachable) plus the Pod Identity IAM role. Deploy the k8s Ingress
#   next; the ALB controller creates the internal ALB and attaches this SG.
#
# Phase 2 (alb_arn/alb_dns set from `kubectl get ingress`): creates the VPC
#   origin + CloudFront distribution, and SWAPS the SG ingress to admit alb_port
#   ONLY from CloudFront's managed "CloudFront-VPCOrigins-Service-SG" (which the
#   VPC origin provisions), looked up via data source. End state: nothing but
#   CloudFront's VPC-origin ENIs can reach the ALB.
variable "alb_arn" {
  type    = string
  default = ""
}
variable "alb_dns" {
  type    = string
  default = ""
}
