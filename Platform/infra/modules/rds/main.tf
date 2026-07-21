variable "cluster_name" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "cluster_security_group_id" { type = string }
variable "eks_cluster_security_group_id" {
  description = "EKS managed cluster SG (from cluster.resourcesVpcConfig.clusterSecurityGroupId)"
  type        = string
}
variable "environment" { type = string }

resource "random_password" "master" {
  length  = 32
  special = false
}

resource "aws_secretsmanager_secret" "db" {
  name                    = "${var.cluster_name}/rds/master-${var.environment}"
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "db" {
  secret_id = aws_secretsmanager_secret.db.id
  secret_string = jsonencode({
    username = "pgadmin"
    password = random_password.master.result
  })
}

resource "aws_db_subnet_group" "this" {
  name       = "${var.cluster_name}-rds"
  subnet_ids = var.private_subnet_ids
  tags       = { Name = "${var.cluster_name}-rds" }
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.cluster_name}-rds-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.cluster_security_group_id, var.eks_cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.cluster_name}-rds-sg" }
}

resource "aws_db_parameter_group" "this" {
  name   = "${var.cluster_name}-pg16"
  family = "postgres16"

  # Flyte's stow library doesn't pass sslmode in connection string;
  # disable force_ssl so non-SSL connections from within VPC are allowed.
  parameter {
    name         = "rds.force_ssl"
    value        = "0"
    apply_method = "immediate"
  }
}

resource "aws_db_instance" "this" {
  identifier = "${var.cluster_name}-pg"
  engine     = "postgres"
  # Track the auto-applied minor version (AWS upgraded 16.9 → 16.13). Pinning the
  # old value would make every plan attempt an invalid downgrade.
  engine_version = "16.13"
  instance_class = "db.r6g.large"

  allocated_storage     = 100
  max_allocated_storage = 500
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "flyteadmin"
  username = "pgadmin"
  password = random_password.master.result

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  parameter_group_name   = aws_db_parameter_group.this.name

  multi_az                  = false
  publicly_accessible       = false
  deletion_protection       = true
  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.cluster_name}-pg-final"
  backup_retention_period   = 7

  tags = { Name = "${var.cluster_name}-pg" }
}

output "endpoint" {
  value = aws_db_instance.this.endpoint
}

output "address" {
  value = aws_db_instance.this.address
}

output "port" {
  value = aws_db_instance.this.port
}

output "master_username" {
  value = aws_db_instance.this.username
}

output "master_password" {
  value     = random_password.master.result
  sensitive = true
}

output "secret_arn" {
  value = aws_secretsmanager_secret.db.arn
}
