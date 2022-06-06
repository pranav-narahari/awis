#!/bin/bash

sh ./generate_config.sh

sudo apt-get install python-pip
python3 -m pip install --upgrade pip

python3 -m pip install pyyaml pyopenssl boto3 flask filterpy awsiotsdk AWSIoTPythonSDK

sh ./reqs.sh