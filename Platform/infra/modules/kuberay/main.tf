variable "cluster_name" { type = string }

resource "helm_release" "kuberay_operator" {
  name             = "kuberay-operator"
  repository       = "https://ray-project.github.io/kuberay-helm/"
  chart            = "kuberay-operator"
  version          = "1.4.2"
  namespace        = "kuberay-system"
  create_namespace = true
  timeout          = 600
  wait             = true

  set {
    name  = "leaderElectionEnabled"
    value = "true"
  }

  set {
    name  = "metrics.enabled"
    value = "true"
  }
}

