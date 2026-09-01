variable "cluster_name" { type = string }

resource "helm_release" "kueue" {
  name             = "kueue"
  repository       = "oci://registry.k8s.io/kueue/charts"
  chart            = "kueue"
  version          = "0.18.1"
  namespace        = "kueue-system"
  create_namespace = true

  values = [file("${path.module}/../../../helm-values/kueue.yaml")]
}
