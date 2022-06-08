#!/bin/bash

sudo echo "creating config.json"

echo "
Sl No: $1 
Name: $2
Site: $3
DD Location: $4"

#need to change the directory
cat > /etc/awis_device_config.json<< EOF 

{
"model": "AWIS-CORAL-v1.0",
"serial": "$1",
"application": "Dock Door VPM Detection",
"name": "$2",
"site": "$3",
"location": "$4",
"notes": "Hardware: Coral TPU Dev Board with Coral Camera"
}


EOF

echo "--- CONFIG CREATED ---"