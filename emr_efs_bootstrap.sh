#!/bin/bash

set -ex
# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

if [[ -z "$1" || -z "$2" ]]
  then
    echo "Missing mandatory arguments: File system ID, region"
    exit 1
fi
 
# get file system id from input argument
fs_id=$1
 
# get region from input argument
region_id=$2

# verify file system is ready
times=0
echo
while [ 5 -gt $times ] && ! aws efs describe-file-systems --file-system-id $fs_id --region $region_id --no-paginate | grep -Po "available"
do
  sleep 5
  times=$(( $times + 1 ))
  echo Attempt $times at verifying efs $fs_id is available...
done
 
# verify mount target is ready
times=0
echo
while [ 5 -gt $times ] && ! aws efs describe-mount-targets --file-system-id $fs_id --region $region_id --no-paginate | grep -Po "available"
do
  sleep 5
  times=$(( $times + 1 ))
  echo Attempt $times at verifying efs $fs_id mount target is available...
done
 
# create local path to mount efs
sudo mkdir -p /efs
 
# mount efs
sudo mount -t nfs4 \
           -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 \
           $fs_id.efs.$region_id.amazonaws.com:/ \
           /efs
 
cd /efs
 
# give hadoop user permission to efs directory
sudo chown -R hadoop:hadoop .
 
if grep  $fs_id /proc/mounts; then
  echo "File system is mounted successfully."
else
  echo "File system mounting is unsuccessful."
  exit 1
fi
