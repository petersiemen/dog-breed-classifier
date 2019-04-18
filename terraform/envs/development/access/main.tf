resource "aws_iam_user" "peter" {
  name = "peter"
  path = "/"
}

resource "aws_iam_user_ssh_key" "peter" {
  username   = "${aws_iam_user.peter.name}"
  encoding   = "SSH"
  public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC7u+xdOvvKxZYaG5sOREKYlwgH5Xfy0oIY8pIJ27d04kwoIWbbGuYXJ9C/1wOPq0URvOv6F5ztEyzAJ71KHREJymTwEqtrshIXqVokcAF76/QhEM+AtO0+7277WKtdsBPap4+jrwpJyUr2QaichfXHXBbDp56dpXCeXeool5d/ZuBg9RfFQM6V1aFwz7YroL3frJhRV2Eo/iSPPHspDQXlODcjYCVB/ECZy1dR56W8RsA8VivNGZJHtb1Py50tTgBRj1hFUIxEkFwQRB6cU8CFZw95T5j9xt1Adn1u6h2IBiv3EdLlM1F8YQdanwstWQT1ipgjtGJdGrZMfEoFrPcv peter@xps"
}

resource "aws_key_pair" "peter" {
  key_name   = "peter"
  public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC7u+xdOvvKxZYaG5sOREKYlwgH5Xfy0oIY8pIJ27d04kwoIWbbGuYXJ9C/1wOPq0URvOv6F5ztEyzAJ71KHREJymTwEqtrshIXqVokcAF76/QhEM+AtO0+7277WKtdsBPap4+jrwpJyUr2QaichfXHXBbDp56dpXCeXeool5d/ZuBg9RfFQM6V1aFwz7YroL3frJhRV2Eo/iSPPHspDQXlODcjYCVB/ECZy1dR56W8RsA8VivNGZJHtb1Py50tTgBRj1hFUIxEkFwQRB6cU8CFZw95T5j9xt1Adn1u6h2IBiv3EdLlM1F8YQdanwstWQT1ipgjtGJdGrZMfEoFrPcv peter@xps"
}
