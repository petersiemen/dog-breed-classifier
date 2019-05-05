output "aws_key_pair__key_name__peter" {
  value = "${aws_key_pair.peter.key_name}"
}

output "security_group__name__ssh_access_from_home" {
  value = "${aws_security_group.ssh-access-from-home.name}"
}

output "security_group__name__jupyter_access_from_home" {
  value = "${aws_security_group.jupyter-access-from-home.name}"
}

output "security_group__name__internet_out" {
  value = "${aws_security_group.internet-out-all-ports.name}"
}

output "security_group__id__ssh_access_from_home" {
  value = "${aws_security_group.ssh-access-from-home.id}"
}

output "security_group__id__internet_out" {
  value = "${aws_security_group.internet-out-all-ports.id}"
}
