output "aws_instance__public_ip" {
  value = "${aws_instance.web.public_ip}"
}
