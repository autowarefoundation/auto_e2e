region       = "us-west-2"
environment  = "dev"
cluster_name = "auto-e2e-platform"
vpc_cidr     = "10.100.0.0/16"

cloudfront_services = {
  mlflow = {
    nlb_arn = "arn:aws:elasticloadbalancing:us-west-2:381491877296:loadbalancer/net/k8s-mlflow-mlflownl-f571c2e62c/4a0df493deddf9b2"
    nlb_dns = "k8s-mlflow-mlflownl-f571c2e62c-4a0df493deddf9b2.elb.us-west-2.amazonaws.com"
  }
  flyte = {
    nlb_arn = "arn:aws:elasticloadbalancing:us-west-2:381491877296:loadbalancer/app/k8s-flyte-flyteui-8b56b98bd1/f8964ae84a30220c"
    nlb_dns = "internal-k8s-flyte-flyteui-8b56b98bd1-493174987.us-west-2.elb.amazonaws.com"
  }
}
