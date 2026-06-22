variable "cluster_name" { type = string }

# Training Operator v1.9.3 — no maintained Helm chart for v1, so kustomize.
# The CRD (kubeflow.org/v1 PyTorchJob) must exist BEFORE Kueue is configured
# to manage pytorchjob workloads.
resource "null_resource" "training_operator" {
  triggers = {
    version = "v1.9.3"
  }

  provisioner "local-exec" {
    command = <<-EOT
      kubectl apply --server-side -k \
        "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.9.3"
    EOT
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<-EOT
      kubectl delete -k \
        "github.com/kubeflow/training-operator.git/manifests/overlays/standalone?ref=v1.9.3" \
        --ignore-not-found || true
    EOT
  }
}
