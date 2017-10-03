#!/bin/sh
 
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

set -e
 
worker_path=/home/hadoop/worker_ip_file
 
echo "Finding ip addresses of all the nodes in the cluster.."

#remove file if already exists
rm -rf /tmp/worker_metadata
 
# gets ip addresses of the nodes in the cluster and saves in temp file 
for LINE in `yarn node -list | grep RUNNING | cut -f1 -d:`
do
  nslookup $LINE | grep Add | grep -v '#' | cut -f 2 -d ' ' >> /tmp/worker_metadata
done
 
# ip address of master node saved in temp file
echo $(hostname -i) >> /tmp/worker_metadata
 
# sorting node ip addresses
sort -n -t . -k 1,1 -k 2,2 -k 3,3 -k 4,4 /tmp/worker_metadata > $worker_path
 
rm -rf /tmp/worker_metadata
 
echo "Setting DEEPLEARNING_WORKERS_PATH, DEEPLEARNING_WORKERS_COUNT, DEEPLEARNING_WORKER_GPU_COUNT as environment variables"
 
sed -i '/export DEEPLEARNING_WORKERS_PATH=/d' ~/.bashrc
echo "export DEEPLEARNING_WORKERS_PATH="$worker_path >> ~/.bashrc
 
# keeping worker_count=number of nodes in the cluster
worker_count="$(wc -l < $worker_path)"
 
sed -i '/export DEEPLEARNING_WORKERS_COUNT=/d' ~/.bashrc
echo "export DEEPLEARNING_WORKERS_COUNT="$worker_count >> ~/.bashrc
 
# setting up number of gpus as env var (Assuming uniform cluster)
gpu_count="$(nvidia-smi -L | grep ^GPU | wc -l)"
 
sed -i '/export DEEPLEARNING_WORKER_GPU_COUNT=/d' ~/.bashrc
echo "export DEEPLEARNING_WORKER_GPU_COUNT="$gpu_count >> ~/.bashrc

source ~/.bashrc

echo $DEEPLEARNING_WORKERS_COUNT
echo $DEEPLEARNING_WORKERS_PATH
echo $DEEPLEARNING_WORKER_GPU_COUNT

echo "Environment variables are set!"
