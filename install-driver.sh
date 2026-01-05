#!/bin/bash -eu
# Modified from google cloouds install_driver.sh program that comes inside /opt/deeplearning/install_driver.sh
#    - Added section to remove current driver
#   - updated driver version to 550.127.08
# Copyright 2020 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Purpose: This script installs NVIDIA Drivers for GPU
#
# Refer the following links for NVIDIA driver installation.
# https://developer.nvidia.com/cuda-toolkit-archive
# https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/"
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

# Remove any current driver first
sudo nvidia-uninstall
sudo apt-get purge -y '^nvidia-.*'
sudo apt-get autoremove -y

export ENV_FILE="/etc/profile.d/env.sh"
# shellcheck source=/etc/profile.d/env.sh disable=SC1091
source "${ENV_FILE}" || exit 1

set -x

function get_metadata_value() {
  e_WAS_SET=${-//[^e]/}
  set +e
  curl --retry 5 \
    -s \
    -f \
    -H "Metadata-Flavor: Google" \
    "http://metadata/computeMetadata/v1/$1"
  if [[ -n $e_WAS_SET ]]; then
    set -e
  fi
  unset e_WAS_SET
}

function get_attribute_value() {
  get_metadata_value "instance/attributes/$1"
}

function install_linux_headers() {
  # Install linux headers. Note that the kernel version might be changed after
  # installing gvnic version. For example: 4.19.0-8-cloud-amd64 ->
  # 4.19.0-9-cloud-amd64. So we install the kernel headers for each driver
  # installation.
  if [[ "${OS_NAME}" == "UBUNTU"* ]]; then
    echo "install linux headers: linux-headers-gcp"
    # https://askubuntu.com/a/268710
    sudo apt install -y linux-headers-gcp linux-headers-"$(uname -r)"
  else
    echo "install linux headers: linux-headers-cloud-amd64"
    # https://askubuntu.com/a/268710
    sudo apt install -y linux-headers-cloud-amd64 linux-headers-"$(uname -r)"
  fi
}

# Try to download driver via Web if GCS failed (Example: VPC-SC/GCS failure)
function download_driver_via_http() {
  local driver_url_path=$1
  local downloaded_file=$2
  echo "Could not use Google Cloud Storage APIs to download driver. Attempting to download them directly from Nvidia."
  echo "Downloading driver from URL: ${driver_url_path}"
  wget -nv "${driver_url_path}" -O "${downloaded_file}" || {
    echo 'Download driver via Web failed!' &&
    rm -f "${downloaded_file}" &&
    echo "${downloaded_file} deleted"
  }
}

function wait_apt_locks_released() {
  # Wait for apt lock to be released
  # Source: https://askubuntu.com/a/373478
  echo "wait apt locks released"
  while sudo fuser /var/{lib/{dpkg,apt/lists},cache/apt/archives}/lock >/dev/null 2>&1 ||
     sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 ; do
     sleep 1
  done
}

function install_nvidia_linux_drivers() {
  echo "DRIVER_VERSION: ${DRIVER_VERSION}"
  local driver_installer_file_name="driver_installer.run"
  local nvidia_driver_file_name="NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
  custom_driver=false
  local driver_gcs_file_path
  if [[ -z "${DRIVER_GCS_PATH}" ]]; then
    DRIVER_GCS_PATH="gs://nvidia-drivers-us-public/tesla/${DRIVER_VERSION}"
    driver_gcs_file_path=${DRIVER_GCS_PATH}/${nvidia_driver_file_name}
  else
    # if custom driver gcs path is provided it must be the exact path to the
    # driver runfile. E.g. "gs://nvidia-drivers-us-public/tesla/510.47.03/NVIDIA-Linux-x86_64-510.47.03.run"
    custom_driver=true
    driver_gcs_file_path=${DRIVER_GCS_PATH}
  fi
  echo "Downloading driver from GCS location and install: ${driver_gcs_file_path}"
  set +e
  gsutil -q cp "${driver_gcs_file_path}" "${driver_installer_file_name}"
  set -e

  # Download driver via http if GCS failed.
  if [[ ! -f "${driver_installer_file_name}" ]]; then
    custom_driver=false
    driver_url_path="http://us.download.nvidia.com/tesla/${DRIVER_VERSION}/${nvidia_driver_file_name}"
    download_driver_via_http "${driver_url_path}" "${driver_installer_file_name}"
  fi

  if [[ ! -f "${driver_installer_file_name}" ]]; then
    echo "Failed to find drivers!"
    exit 1
  fi

  # open kernel modules argument, blank if installing proprietary kernel modules
  # "-m=kernel-open" if installing open kernel modules. Provided to the Runfile.
  local open_kernel_module_arg="-m=kernel-open"
  IFS=. read -r major minor patch <<< "${DRIVER_VERSION}"
  # full machine type string, example: projects/1098636466100/machineTypes/a2-highgpu-1g
  # -r for readonly as we will not be modifying this var, and this satisfies
  # https://github.com/koalaman/shellcheck/wiki/SC2155
  local -r machine_type_full=$(get_metadata_value instance/machine-type)
  # get last element of string split by '/' to get machine type
  # https://stackoverflow.com/questions/3162385/how-to-split-a-string-in-shell-and-get-the-last-field
  local machine_type=${machine_type_full##*/}
  # Install proprietary kernel modules if the instance is on an N-series
  # machine, if the driver version is on a major release older than R525, or if
  # a custom driver install runfile was provided as we don't know if it supports
  # open kernel drivers or not.
  # Nvidia R515 is the first major release that supports open kernel modules,
  # and R525 is the first major release that will be compatible with an upcoming
  # required kernel module needed for Kyubi.
  if [[ "${major}" -lt 525 ]] || [[ $machine_type =~ ^n ]] || [[ $custom_driver == true ]]; then
    open_kernel_module_arg=""
  fi

  chmod +x ${driver_installer_file_name}
  # --dkms registers the driver with DKMS, which is used to install the kernel
  # modules on kernel upgrade. --ui=none hides the UI, --no-questions skips the
  # interactive prompts, --no-drm skips the DRM module installation, and
  # --install-libglvnd installs the libglvnd package, open_kernel_module_arg
  # installs the open kernel modules if supported on the machine type.
  sudo ./${driver_installer_file_name} --dkms -a --ui=none --no-questions --no-drm --install-libglvnd "${open_kernel_module_arg}"
  rm -rf ${driver_installer_file_name}
}

main() {
  wait_apt_locks_released
  install_linux_headers
  # shellcheck source=/opt/deeplearning/driver-version.sh disable=SC1091
  export DRIVER_VERSION=550.127.08
  export DRIVER_GCS_PATH
  # Custom GCS driver location via instance metadata.
  DRIVER_GCS_PATH=$(get_attribute_value nvidia-driver-gcs-path)
  install_nvidia_linux_drivers
  exit 0
}

main
