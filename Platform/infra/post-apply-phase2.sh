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
kubectl apply -f ../k8s/ingress-class.yaml

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
  -f Platform/docker/training/Dockerfile .

echo "=== 8. Register Flyte workflows ==="
cd Platform/pipelines
pip install flytekit flytekitplugins-kfpytorch 2>/dev/null || true
export AWS_ACCOUNT_ID="$ACCOUNT"
export AWS_REGION="$REGION"
export MLFLOW_TRACKING_URI=$(kubectl get svc mlflow -n mlflow -o jsonpath='{.spec.clusterIP}' 2>/dev/null):5000
export MLFLOW_TRACKING_URI="http://${MLFLOW_TRACKING_URI}"
pyflyte register training/ data_ingest/ evaluation/ \
  --project auto-e2e \
  --domain development \
  --image "${ECR_URL}/auto-e2e/data-prep:latest" \
  || echo "WARNING: pyflyte register failed. Ensure flytekit is installed."

echo "=== 9. Register ALB Target Group Bindings (MLflow + Flyte) ==="
MLFLOW_TG_ARN=$(terraform output -json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('mlflow_tg_arn',{}).get('value',''))" 2>/dev/null)
FLYTE_TG_ARN=$(terraform output -json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('flyte_tg_arn',{}).get('value',''))" 2>/dev/null)

if [ -n "$MLFLOW_TG_ARN" ]; then
  MLFLOW_IP=$(kubectl get pod -n mlflow -l app.kubernetes.io/name=mlflow -o jsonpath='{.items[0].status.podIP}' 2>/dev/null)
  if [ -n "$MLFLOW_IP" ]; then
    aws elbv2 register-targets --target-group-arn "$MLFLOW_TG_ARN" \
      --targets Id=$MLFLOW_IP,Port=5000 --profile "$PROFILE" --region "$REGION" 2>/dev/null
    echo "  MLflow registered: $MLFLOW_IP:5000"
  fi
fi

if [ -n "$FLYTE_TG_ARN" ]; then
  FLYTE_IP=$(kubectl get pod -n flyte -l app.kubernetes.io/name=flyte-binary-console -o jsonpath='{.items[0].status.podIP}' 2>/dev/null)
  if [ -n "$FLYTE_IP" ]; then
    aws elbv2 register-targets --target-group-arn "$FLYTE_TG_ARN" \
      --targets Id=$FLYTE_IP,Port=80 --profile "$PROFILE" --region "$REGION" 2>/dev/null
    echo "  Flyte registered: $FLYTE_IP:80"
  fi
fi

echo "=== 10. Create HF_TOKEN K8s Secret (from Secrets Manager) ==="
HF_SECRET_ARN=$(aws secretsmanager list-secrets --profile "$PROFILE" --region "$REGION" \
  --query "SecretList[?contains(Name,'hf-token')].ARN" --output text 2>/dev/null)
if [ -n "$HF_SECRET_ARN" ]; then
  HF_TOKEN_VAL=$(aws secretsmanager get-secret-value --secret-id "$HF_SECRET_ARN" \
    --region "$REGION" --profile "$PROFILE" --query SecretString --output text)
  kubectl create secret generic hf-token -n auto-e2e-training \
    --from-literal=HF_TOKEN="$HF_TOKEN_VAL" \
    --dry-run=client -o yaml | kubectl apply -f -
  echo "  hf-token secret created/updated"
else
  echo "  No HF_TOKEN in Secrets Manager (optional, skip)"
fi

echo ""
echo "=== Done ==="
echo "Verify:"
echo "  kubectl get pods -n flyte"
echo "  kubectl get pods -n mlflow"
echo "  kubectl get pods -n kueue-system"
echo "  kubectl get clusterqueues"
echo "  kubectl get localqueues -n auto-e2e-training"
