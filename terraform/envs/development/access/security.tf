resource "aws_security_group" "ssh-access-from-home" {
  name = "ssh-access-from-home"
}

resource "aws_security_group" "internet-out-all-ports" {
  name = "all-access-out-to-internet"
}

resource "aws_security_group_rule" "ssh-access-from-home" {
  from_port         = 22
  protocol          = "TCP"
  security_group_id = "${aws_security_group.ssh-access-from-home.id}"
  to_port           = 22
  type              = "ingress"

  cidr_blocks = [
    "${var.ip_address_home}/32",
  ]
}

resource "aws_security_group_rule" "internet-out-all-ports" {
  description       = "Internet access"
  from_port         = 0
  protocol          = "-1"
  security_group_id = "${aws_security_group.internet-out-all-ports.id}"
  to_port           = 0
  type              = "egress"

  cidr_blocks = [
    "0.0.0.0/0",
  ]
}
