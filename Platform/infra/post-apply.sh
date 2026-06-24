#!/bin/bash
# Run after `terraform apply` completes successfully.
# Sets up kubeconfig, applies the GPU NodePool + warm-node keeper, builds and
# pushes the training image, and runs a GPU smoke-test Pod.
#
# All account/region specifics come from env or are resolved at runtime, so
# nothing account-specific is baked in:
#   AWS_PROFILE=myprofile AWS_REGION=us-west-2 ./post-apply.sh
#
# Uses Finch for the image build (Docker Desktop requires org sign-in here).

set -euo pipefail

PROFILE="${AWS_PROFILE:-autowarefoundation}"
REGION="${AWS_REGION:-us-west-2}"
CLUSTER="${EKS_CLUSTER:-auto-e2e-platform}"
CONTAINER_CLI="${CONTAINER_CLI:-finch}"   # finch or docker

echo "=== 1. Update kubeconfig ==="
aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION" --profile "$PROFILE"

echo "=== 2. Verify cluster access ==="
kubectl get nodepools

echo "=== 3. Apply GPU NodePool + warm-node keeper ==="
kubectl apply -f ../k8s/karpenter-nodepools/gpu-nodepool.yaml
kubectl apply -f ../k8s/gpu-node-keeper.yaml
echo "Waiting for a GPU node to register (Karpenter provisions g6e)..."
kubectl wait --for=condition=Ready node -l workload-type=gpu-training --timeout=600s

echo "=== 4. ECR login ==="
ACCOUNT=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
ECR_URL="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr get-login-password --region "$REGION" --profile "$PROFILE" | \
  "$CONTAINER_CLI" login --username AWS --password-stdin "$ECR_URL"

echo "=== 5. Build and push training image (linux/amd64) ==="
cd ../../..
"$CONTAINER_CLI" build \
  --platform linux/amd64 \
  --output type=image,name="${ECR_URL}/auto-e2e/training:latest",push=true \
  -f platform/docker/training/Dockerfile .

echo "=== 6. Run GPU smoke test Pod ==="
cd platform/k8s
sed "s|REPLACE_WITH_ECR_URL|${ECR_URL}|g" gpu-smoke-test.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/train-smoke-test --timeout=300s 2>/dev/null || true
kubectl logs -f train-smoke-test

echo ""
echo "=== Done ==="
echo "GPU node: kubectl get nodes -l workload-type=gpu-training"
echo "Cleanup smoke test: kubectl delete pod train-smoke-test"
