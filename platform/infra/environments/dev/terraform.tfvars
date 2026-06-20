region       = "us-west-2"
environment  = "dev"
cluster_name = "auto-e2e-platform"
vpc_cidr     = "10.100.0.0/16"

# g6e.4xlarge in the AZ where the ODCR is held (capacity-constrained instance).
gpu_instance_types = ["g6e.4xlarge"]
gpu_azs            = ["us-west-2b"]

# odcr_id is set in secrets.auto.tfvars (gitignored) — it is account-specific
# and changes per capacity-reservation attempt. See secrets.auto.tfvars.example.

cloudfront_services = {
  mlflow = {
    nlb_arn = "arn:aws:elasticloadbalancing:us-west-2:381491877296:loadbalancer/net/k8s-mlflow-mlflownl-f571c2e62c/4a0df493deddf9b2"
    nlb_dns = "k8s-mlflow-mlflownl-f571c2e62c-4a0df493deddf9b2.elb.us-west-2.amazonaws.com"
  }
  flyte = {
    nlb_arn = "arn:aws:elasticloadbalancing:us-west-2:381491877296:loadbalancer/net/k8s-flyte-flytenlb-83d3b305a7/fe1ae752eae90b76"
    nlb_dns = "k8s-flyte-flytenlb-83d3b305a7-fe1ae752eae90b76.elb.us-west-2.amazonaws.com"
  }
}
