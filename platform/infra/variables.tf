variable "region" {
  type    = string
  default = "us-west-2"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "cluster_name" {
  type    = string
  default = "auto-e2e-platform"
}

variable "vpc_cidr" {
  type    = string
  default = "10.100.0.0/16"
}

variable "gpu_instance_types" {
  type    = list(string)
  default = ["g6e.4xlarge"]
}

variable "gpu_azs" {
  description = "AZ(s) where the GPU ODCR is confirmed. Capacity-dependent, set per environment."
  type        = list(string)
  default     = ["us-west-2b"]
}

# Supplied via secrets.auto.tfvars (gitignored) or TF_VAR_odcr_id.
# The ODCR is a capacity reservation that changes per provisioning attempt and is
# account-specific, so it is never committed.
variable "odcr_id" {
  description = "On-Demand Capacity Reservation ID for the GPU node (set in secrets.auto.tfvars)"
  type        = string
}

variable "hf_token" {
  description = "HuggingFace API token for gated dataset access (set in secrets.auto.tfvars)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "sim_odcr_id" {
  description = "ODCR ID for simulation GPU node (g5.xlarge). Empty = skip simulation NodePool."
  type        = string
  default     = ""
}
