variable "environment" { type = string }

locals {
  repositories = ["auto-e2e/training", "auto-e2e/data-prep", "auto-e2e/eval", "auto-e2e/offline-rl"]
}

resource "aws_ecr_repository" "this" {
  for_each = toset(local.repositories)
  name     = each.value

  image_scanning_configuration { scan_on_push = true }
  image_tag_mutability = "MUTABLE"

  tags = { Purpose = split("/", each.value)[1] }
}

resource "aws_ecr_lifecycle_policy" "this" {
  for_each   = aws_ecr_repository.this
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 20 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = { type = "expire" }
    }]
  })
}

output "repository_urls" {
  value = { for k, r in aws_ecr_repository.this : k => r.repository_url }
}
