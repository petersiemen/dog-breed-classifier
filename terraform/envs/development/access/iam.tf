resource "aws_iam_instance_profile" "deep-learning-ec2-profile" {
  name = "deep-learning-ec2-profile"
  role = "${aws_iam_role.deep-learning-ec2-role.name}"
}

resource "aws_iam_role" "deep-learning-ec2-role" {
  name = "deep-learning-ec2-role"
  path = "/"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Principal": {
       "Service": "ec2.amazonaws.com"
      },
      "Effect": "Allow",
      "Sid": ""
    }
  ]
}
EOF
}

resource "aws_iam_policy" "cloudwatch-put-metric-policy" {
  name        = "cloudwatch-policy"
  description = "cloudwatch policy"

  policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
         {
             "Action": [
                 "cloudwatch:PutMetricData"
              ],
              "Effect": "Allow",
              "Resource": "*"
         }
     ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "deep-learning-ec2-attach-cloudwatch-put-policy" {
  role       = "${aws_iam_role.deep-learning-ec2-role.name}"
  policy_arn = "${aws_iam_policy.cloudwatch-put-metric-policy.arn}"
}
