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

# keeping worker_count=number of nodes in the cluster
worker_count="$(wc -l < $worker_path)"

# setting up number of gpus as env var (Assuming uniform cluster)
gpu_count="$(nvidia-smi -L | grep ^GPU | wc -l)"

if [ -e ~/.bashrc ]; then
  sed -i '/export DEEPLEARNING_WORKERS_PATH=/d' ~/.bashrc
  echo "export DEEPLEARNING_WORKERS_PATH="$worker_path >> ~/.bashrc

  sed -i '/export DEEPLEARNING_WORKERS_COUNT=/d' ~/.bashrc
  echo "export DEEPLEARNING_WORKERS_COUNT="$worker_count >> ~/.bashrc

  sed -i '/export DEEPLEARNING_WORKER_GPU_COUNT=/d' ~/.bashrc
  echo "export DEEPLEARNING_WORKER_GPU_COUNT="$gpu_count >> ~/.bashrc

  source ~/.bashrc
fi

if [ -e ~/.zshrc ]; then
  sed -i '/export DEEPLEARNING_WORKERS_PATH=/d' ~/.zshrc
  echo "export DEEPLEARNING_WORKERS_PATH="$worker_path >> ~/.zshrc

  sed -i '/export DEEPLEARNING_WORKERS_COUNT=/d' ~/.zshrc
  echo "export DEEPLEARNING_WORKERS_COUNT="$worker_count >> ~/.zshrc

  sed -i '/export DEEPLEARNING_WORKER_GPU_COUNT=/d' ~/.zshrc
  echo "export DEEPLEARNING_WORKER_GPU_COUNT="$gpu_count >> ~/.zshrc

  source ~/.zshrc
fi

if [ -e ~/.config/fish/config.fish ]; then
  sed -i '/set DEEPLEARNING_WORKERS_PATH /d' ~/.config/fish/config.fish
  echo "set DEEPLEARNING_WORKERS_PATH "$worker_path >> ~/.config/fish/config.fish

  sed -i '/set DEEPLEARNING_WORKERS_COUNT /d' ~/.config/fish/config.fish
  echo "set DEEPLEARNING_WORKERS_COUNT "$worker_count >> ~/.config/fish/config.fish

  sed -i '/set DEEPLEARNING_WORKER_GPU_COUNT /d' ~/.config/fish/config.fish
  echo "set DEEPLEARNING_WORKER_GPU_COUNT "$gpu_count >> ~/.config/fish/config.fish

  source ~/.config/fish/config.fish
fi

echo "Setting Enviroment variables: "
echo "DEEPLEARNING_WORKERS_COUNT = $DEEPLEARNING_WORKERS_COUNT"
echo "DEEPLEARNING_WORKERS_PATH = $DEEPLEARNING_WORKERS_PATH"
echo "DEEPLEARNING_WORKER_GPU_COUNT = $DEEPLEARNING_WORKER_GPU_COUNT"
echo "Environment variables are set!"
