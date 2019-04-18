# deep-learning-aws

This repository contains terraform modules and layers to bootstrap a development environment for deep learning. 



# Bootstrap the enviroment
```bash
cd terraform/envs/development/bootstrap
terraform init
terraform apply
```

# Configure Basic Security Access Rules
```bash
cd terraform/envs/development/access
terraform init
terraform apply
```

# Terraform the deep-learning-aws instance
```bash
cd terraform/envs/development/deep-learning-ec2
terraform init
terraform apply
```

Get the public IP address of the ec2-instance
# establish and ssh tunnel on port 8888 into the machine
```bash
ssh -L localhost:8888:localhost:8888  ec2-user@[PUBLIC_IP_ADDRESS_OF_EC2_INSTANCE]
```

# start a jupyter notebook
```bash
juypter notebook
```




