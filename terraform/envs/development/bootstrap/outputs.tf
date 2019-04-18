output "tf-bucket" {
  value = "${aws_s3_bucket.terraform-state.bucket}"
}

output "tf-lock" {
  value = "${aws_dynamodb_table.terraform-lock.name}"
}
