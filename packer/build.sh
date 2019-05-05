#!/usr/bin/env bash


set -euo pipefail
IFS=$'\n\t'

PACKER_FILE="$1"

AWS_PROFILE=acme-development packer build -var-file=variables.json  "${PACKER_FILE}"


#"sudo yum -y -q install cmake3",
#        "cd /home/ec2-user/deep-learning-aws/dog-breed-classifier && /home/ec2-user/.local/bin/pipenv install",
