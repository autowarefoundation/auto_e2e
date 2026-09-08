#!/bin/bash
# Run after `terraform apply` completes successfully.
# Sets up kubeconfig, applies GPU capacity and queue manifests, builds and
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
ACCOUNT=$(aws sts get-caller-identity \
  --profile "$PROFILE" \
  --query Account \
  --output text)

echo "=== 1. Update kubeconfig ==="
aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION" --profile "$PROFILE"

echo "=== 2. Verify cluster access ==="
kubectl get nodepools

echo "=== 3. Verify Flyte project GPU quotas ==="
wait_for_gpu_quota() {
  local namespace="$1"
  local expected_gpu="$2"
  local deadline=$((SECONDS + 600))

  while (( SECONDS < deadline )); do
    if kubectl \
      --namespace "$namespace" \
      get resourcequota project-quota \
      --output json 2>/dev/null | \
      EXPECTED_GPU="$expected_gpu" python3 -c '
import json
import os
import sys

hard = json.load(sys.stdin)["spec"]["hard"]
expected = os.environ["EXPECTED_GPU"]
valid = (
    hard.get("requests.nvidia.com/gpu") == expected
    and "limits.nvidia.com/gpu" not in hard
)
raise SystemExit(0 if valid else 1)
'
    then
      return
    fi
    sleep 5
  done

  echo "Flyte GPU quota did not converge for ${namespace}" >&2
  kubectl \
    --namespace "$namespace" \
    get resourcequota project-quota \
    --output yaml >&2 || true
  return 1
}

for namespace in \
  auto-e2e-development \
  auto-e2e-staging \
  auto-e2e-production
do
  if kubectl \
    --namespace "$namespace" \
    get resourcequota project-quota \
    --output json | \
    python3 -c '
import json
import sys

hard = json.load(sys.stdin)["spec"]["hard"]
raise SystemExit(
    0 if "limits.nvidia.com/gpu" in hard else 1
)
'
  then
    kubectl \
      --namespace "$namespace" \
      patch resourcequota project-quota \
      --type json \
      --patch \
      '[{"op":"remove","path":"/spec/hard/limits.nvidia.com~1gpu"}]'
  fi
done

wait_for_gpu_quota auto-e2e-development 18
wait_for_gpu_quota auto-e2e-staging 0
wait_for_gpu_quota auto-e2e-production 0

echo "=== 4. Verify retained g6 validation reservation ==="
aws ec2 describe-capacity-reservations \
  --profile "$PROFILE" \
  --region "$REGION" \
  --filters \
    Name=tag:Name,Values=auto-e2e-gpu-validation \
    Name=instance-type,Values=g6.2xlarge \
    Name=state,Values=active \
  --output json > /tmp/auto-e2e-gpu-validation-reservations.json
python3 - <<'PY'
import json

with open(
    "/tmp/auto-e2e-gpu-validation-reservations.json",
    encoding="ascii",
) as stream:
    reservations = json.load(stream)["CapacityReservations"]
if len(reservations) != 1:
    raise ValueError(
        "exactly one active g6 validation reservation is required"
    )
reservation = reservations[0]
if (
    reservation["AvailabilityZone"] != "us-west-2a"
    or reservation["InstanceMatchCriteria"] != "targeted"
    or int(reservation["TotalInstanceCount"]) != 2
):
    raise ValueError(
        "g6 validation reservation must be targeted, us-west-2a, "
        "and contain exactly two slots"
    )
PY

echo "=== 5. Apply GPU NodeClasses, NodePools, and Kueue queues ==="
sed "s/REPLACE_WITH_AWS_ACCOUNT_ID/${ACCOUNT}/g" \
  ../k8s/karpenter-nodepools/gpu-nodeclass.yaml | kubectl apply -f -
kubectl apply -f ../k8s/karpenter-nodepools/gpu-nodepool.yaml
kubectl apply -f ../k8s/kueue-config/kueue-objects.yaml
for resource in \
  nodeclass/auto-e2e-gpu-validation \
  nodepool/gpu-validation \
  nodeclass/auto-e2e-p5en-capacity-block \
  nodepool/p5en-capacity-block
do
  kubectl wait --for=condition=Ready "$resource" --timeout=120s
done
kubectl --namespace kueue-system get configmap kueue-manager-config \
  --output jsonpath='{.data.controller_manager_config\.yaml}' \
  > /tmp/auto-e2e-kueue-manager-config.yaml
for framework in \
  batch/job \
  kubeflow.org/pytorchjob \
  ray.io/rayjob \
  pod
do
  grep -Eq \
    "^[[:space:]]*-[[:space:]]*\"?${framework}\"?[[:space:]]*$" \
    /tmp/auto-e2e-kueue-manager-config.yaml
done
kubectl rollout status \
  deployment/kueue-controller-manager \
  --namespace kueue-system \
  --timeout=300s
kubectl delete --ignore-not-found --wait=false \
  nodepool/gpu-canary \
  nodepool/gpu-performance-reserved \
  nodepool/gpu-performance-ondemand \
  nodepool/gpu-training \
  nodepool/gpu-burst \
  nodepool/gpu-smoke
kubectl delete --ignore-not-found --wait=false \
  nodeclass/auto-e2e-gpu-canary \
  nodeclass/auto-e2e-gpu-performance-reserved \
  nodeclass/auto-e2e-gpu-performance-ondemand \
  nodeclass/auto-e2e-gpu-training
kubectl delete --ignore-not-found --wait=false \
  --namespace auto-e2e-development \
  localqueue/gpu-canary \
  localqueue/gpu-performance \
  localqueue/training
kubectl delete --ignore-not-found --wait=false \
  --namespace auto-e2e-training \
  localqueue/training
kubectl delete --ignore-not-found --wait=false \
  clusterqueue/gpu-canary-queue \
  clusterqueue/gpu-performance-queue \
  clusterqueue/training-queue
kubectl delete --ignore-not-found --wait=false \
  resourceflavor/gpu-canary-flavor \
  resourceflavor/gpu-performance-flavor \
  resourceflavor/gpu-flavor

echo "p5en NodePool remains at zero nodes until a tagged block is active."

echo "=== 6. ECR login ==="
ECR_URL="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr get-login-password --region "$REGION" --profile "$PROFILE" | \
  "$CONTAINER_CLI" login --username AWS --password-stdin "$ECR_URL"

echo "=== 7. Build and push training image (linux/amd64) ==="
cd ../../..
"$CONTAINER_CLI" build \
  --platform linux/amd64 \
  --output type=image,name="${ECR_URL}/auto-e2e/training:latest",push=true \
  -f Platform/docker/training/Dockerfile .

echo "=== 8. Run GPU smoke test Pod ==="
cd Platform/k8s
sed "s|REPLACE_WITH_ECR_URL|${ECR_URL}|g" gpu-smoke-test.yaml | kubectl apply -f -
if kubectl wait \
  --namespace auto-e2e-development \
  --for=condition=Ready \
  pod/train-smoke-test \
  --timeout=900s
then
  kubectl logs \
    --namespace auto-e2e-development \
    -f train-smoke-test
else
  kubectl describe \
    --namespace auto-e2e-development \
    pod/train-smoke-test >&2
  exit 1
fi

echo ""
echo "=== Done ==="
echo "GPU capacity: kubectl get nodeclasses,nodepools"
echo "Cleanup smoke test: kubectl delete pod --namespace auto-e2e-development train-smoke-test"
