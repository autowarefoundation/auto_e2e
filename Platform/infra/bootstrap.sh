#!/bin/bash
# Bootstrap Terraform backend (S3 bucket + DynamoDB lock table).
# Run ONCE before the first `terraform init`.
#
# Usage: ./bootstrap.sh

set -euo pipefail

# All overridable via env so no account/profile specifics are baked in.
#   AWS_PROFILE=myprofile TFSTATE_BUCKET=my-bucket ./bootstrap.sh
PROFILE="${AWS_PROFILE:-autowarefoundation}"
REGION="${TFSTATE_REGION:-us-east-1}"   # state bucket region (created once)
BUCKET="${TFSTATE_BUCKET:-auto-e2e-platform-tfstate}"
TABLE="${TFLOCK_TABLE:-auto-e2e-platform-tflock}"

echo "Creating S3 bucket for Terraform state..."
aws s3api create-bucket \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --profile "$PROFILE" 2>/dev/null || echo "Bucket already exists"

aws s3api put-bucket-versioning \
  --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled \
  --profile "$PROFILE"

aws s3api put-bucket-encryption \
  --bucket "$BUCKET" \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' \
  --profile "$PROFILE"

aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true \
  --profile "$PROFILE"

echo "Creating DynamoDB table for state locking..."
aws dynamodb create-table \
  --table-name "$TABLE" \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region "$REGION" \
  --profile "$PROFILE" 2>/dev/null || echo "Table already exists"

echo "Done. Run: cd platform/infra && terraform init -var-file=environments/dev/terraform.tfvars"
