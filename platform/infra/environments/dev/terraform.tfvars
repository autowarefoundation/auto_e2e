region       = "us-west-2"
environment  = "dev"
cluster_name = "auto-e2e-platform"
vpc_cidr     = "10.100.0.0/16"

# g6e.4xlarge in the AZ where the ODCR is held (capacity-constrained instance).
gpu_instance_types = ["g6e.4xlarge"]
gpu_azs            = ["us-west-2b"]

# odcr_id is set in secrets.auto.tfvars (gitignored) — it is account-specific
# and changes per capacity-reservation attempt. See secrets.auto.tfvars.example.
