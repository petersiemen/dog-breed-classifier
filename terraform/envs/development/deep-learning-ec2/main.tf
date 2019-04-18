data "terraform_remote_state" "access" {
  backend = "s3"

  config {
    encrypt        = "true"
    bucket         = "acme-development-terraform-remote-state"
    key            = "access.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "terraform-lock"
  }
}

resource "aws_instance" "web" {
  ami           = "${var.deep_learning_aws_ami}"
  instance_type = "${var.deep_learning_aws_instance_type}"
  key_name      = "${data.terraform_remote_state.access.aws_key_pair__key_name__peter}"

  security_groups = [
    "${data.terraform_remote_state.access.security_group__name__ssh_access_from_home}",
    "${data.terraform_remote_state.access.security_group__name__internet_out}",
  ]

  tags = {
    Name        = "deep-learning"
    Hostname    = "deep-learning"
    Role        = "deep-learning"
    Environment = "${var.env}"
  }
}
