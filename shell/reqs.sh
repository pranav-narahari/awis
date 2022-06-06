#!/bin/bash
# Copyright (c) Amazon Inc.
# All Rights Reserved
# AMAZON.COM CONFIDENTIAL

# error handling and reporting --- fail and report, debug all lines
set -eu -o pipefail

# run with sudo/admin privileges and don't ask for password
sudo -n true
test $? -eq 0 || exit 1 "this script requires admin privileges"

echo "installing required libraries"
while read -r requirement ;
do
  sudo apt-get install -y --quiet $requirement ;
done < <(cat << EOF 
  libglfw3-dev
  libgl1-mesa-dev
  libglu1-mesa-dev
  libusb-1.0-0-dev
  xorg-dev
  libssl-dev
  python3-dev
  python3-pip
  cmake
  libhdf5-dev
  pkg-config
  libpng-dev
  libtiff-dev
  libgtk-3-dev
  gfortran
  git
  libxvidcore-dev
  libx265-dev
  libatlas-base-dev
  unzip
EOF
)

echo -e "\n"
echo "--- COMPLETE ---"