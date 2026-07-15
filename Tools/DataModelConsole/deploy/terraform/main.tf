terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# This Terraform root is retired. Platform/infra-console is the sole owner of
# the production DataModelConsole infrastructure. Requiring an explicit true
# value prevents an accidental apply in this directory; an intentional apply
# only detaches legacy local state and never destroys remote resources.
variable "acknowledge_state_detach" {
  type        = bool
  nullable    = false
  description = "Set true only to detach legacy state after verifying Platform/infra-console owns every resource"

  validation {
    condition     = var.acknowledge_state_detach
    error_message = "This root is retired; use Platform/infra-console. Set acknowledge_state_detach=true only for a state-only detach."
  }
}

removed {
  from = aws_security_group.console_alb

  lifecycle {
    destroy = false
  }
}

removed {
  from = aws_vpc_security_group_ingress_rule.console_alb_from_cloudfront

  lifecycle {
    destroy = false
  }
}

removed {
  from = aws_vpc_security_group_egress_rule.console_alb_all

  lifecycle {
    destroy = false
  }
}

removed {
  from = aws_cloudfront_vpc_origin.console

  lifecycle {
    destroy = false
  }
}

removed {
  from = aws_cloudfront_distribution.console

  lifecycle {
    destroy = false
  }
}

removed {
  from = aws_iam_role.console_api

  lifecycle {
    destroy = false
  }
}

removed {
  from = aws_iam_role_policy.console_api_s3_readonly

  lifecycle {
    destroy = false
  }
}

removed {
  from = aws_iam_role_policy.console_api_dynamo

  lifecycle {
    destroy = false
  }
}

removed {
  from = aws_eks_pod_identity_association.console_api

  lifecycle {
    destroy = false
  }
}
