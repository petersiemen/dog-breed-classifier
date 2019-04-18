variable "organization" {}
variable "env" {}
variable "aws_account_id" {}
variable "aws_region" {}

variable "tf-state-bucket" {
  default = "acme-development-terraform-remote-state"
}
