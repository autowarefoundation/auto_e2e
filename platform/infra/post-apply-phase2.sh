#!/bin/bash
# Run after `terraform apply` for Phase 2 completes.
# Applies K8s manifests that depend on Terraform-provisioned resources (RDS, etc.)
# and injects real DB credentials into placeholder Secrets.
#
# Usage: AWS_PROFILE=autowarefoundation ./post-apply-phase2.sh

set -euo pipefail

PROFILE="${AWS_PROFILE:-autowarefoundation}"
REGION="${AWS_REGION:-us-west-2}"
CLUSTER="${EKS_CLUSTER:-auto-e2e-platform}"

echo "=== 1. Update kubeconfig ==="
aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION" --profile "$PROFILE"

echo "=== 2. Apply StorageClass (must be first — PVC charts need it) ==="
kubectl apply -f ../k8s/storage-class.yaml

echo "=== 3. Apply Phase 2 namespaces + SA + placeholder Secrets ==="
kubectl apply -f ../k8s/phase2-namespaces.yaml

echo "=== 4. Inject real RDS credentials into K8s Secrets ==="
RDS_ENDPOINT=$(terraform output -raw rds_endpoint)
RDS_HOST="${RDS_ENDPOINT%%:*}"
SECRET_ARN=$(terraform output -raw -module=rds secret_arn 2>/dev/null || true)

if [ -n "$SECRET_ARN" ]; then
  CREDS_JSON=$(aws secretsmanager get-secret-value \
    --secret-id "$SECRET_ARN" --region "$REGION" --profile "$PROFILE" \
    --query SecretString --output text)
  DB_USER=$(echo "$CREDS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['username'])")
  DB_PASS=$(echo "$CREDS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['password'])")
else
  echo "WARNING: Could not retrieve secret ARN. Using terraform output."
  DB_USER="pgadmin"
  DB_PASS=$(terraform output -raw -module=rds master_password 2>/dev/null || echo "UNKNOWN")
fi

# Patch flyte-db-pass secret
kubectl create secret generic flyte-db-pass -n flyte \
  --from-literal=POSTGRES_HOST="$RDS_HOST" \
  --from-literal=POSTGRES_PORT="5432" \
  --from-literal=POSTGRES_DB="flyteadmin" \
  --from-literal=POSTGRES_USER="$DB_USER" \
  --from-literal=POSTGRES_PASSWORD="$DB_PASS" \
  --dry-run=client -o yaml | kubectl apply -f -

# Patch mlflow-db-secret
kubectl create secret generic mlflow-db-secret -n mlflow \
  --from-literal=POSTGRES_HOST="$RDS_HOST" \
  --from-literal=POSTGRES_PORT="5432" \
  --from-literal=POSTGRES_DB="mlflow" \
  --from-literal=POSTGRES_USER="$DB_USER" \
  --from-literal=POSTGRES_PASSWORD="$DB_PASS" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "=== 5. Create mlflow DB on RDS (if not exists) ==="
kubectl run pg-init --rm -i --restart=Never --namespace=flyte \
  --image=postgres:16-alpine \
  --env="PGPASSWORD=$DB_PASS" \
  -- psql -h "$RDS_HOST" -U "$DB_USER" -d flyteadmin \
  -c "SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\\gexec" \
  || echo "mlflow DB may already exist (OK)"

echo "=== 6. Apply Kueue objects (after Helm installs CRDs) ==="
kubectl apply -f ../k8s/kueue-config/kueue-objects.yaml

echo "=== 7. Build and push training image ==="
ACCOUNT=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
ECR_URL="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
CONTAINER_CLI="${CONTAINER_CLI:-finch}"

aws ecr get-login-password --region "$REGION" --profile "$PROFILE" | \
  "$CONTAINER_CLI" login --username AWS --password-stdin "$ECR_URL"

cd ../../..
"$CONTAINER_CLI" build \
  --platform linux/amd64 \
  --output type=image,name="${ECR_URL}/auto-e2e/training:latest",push=true \
  -f platform/docker/training/Dockerfile .

echo "=== 8. Register Flyte workflows ==="
cd platform/pipelines
pip install flytekit==1.16.23 flytekitplugins-kfpytorch==1.16.23 2>/dev/null || true
pyflyte register training/ \
  --project auto-e2e \
  --domain development \
  --image "${ECR_URL}/auto-e2e/training:latest" \
  || echo "WARNING: pyflyte register failed. Ensure flytekit is installed."

echo ""
echo "=== Done ==="
echo "Verify:"
echo "  kubectl get pods -n flyte"
echo "  kubectl get pods -n mlflow"
echo "  kubectl get pods -n kueue-system"
echo "  kubectl get clusterqueues"
echo "  kubectl get localqueues -n auto-e2e-training"
