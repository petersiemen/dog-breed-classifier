#!/usr/bin/env bash


set -euo pipefail
IFS=$'\n\t'

PACKER_FILE="$1"

AWS_PROFILE=acme-development packer build -var-file=variables.json  "${PACKER_FILE}"
