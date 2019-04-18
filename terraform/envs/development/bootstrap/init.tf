terraform {
  required_version = "= 0.11.7"
}

provider "aws" {
  region = "${var.aws_region}"

  allowed_account_ids = [
    "${var.aws_account_id}",
  ]
}
