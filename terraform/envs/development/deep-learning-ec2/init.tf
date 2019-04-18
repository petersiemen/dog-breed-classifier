terraform {
  required_version = "= 0.11.7"

  backend "s3" {
    encrypt        = "true"
    bucket         = "acme-development-terraform-remote-state"
    key            = "deep-learning-ec2.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "terraform-lock"
  }
}

provider "aws" {
  region = "${var.aws_region}"

  allowed_account_ids = [
    "${var.aws_account_id}",
  ]
}
