# Distributed Deep Learning on AWS EMR cluster with Apache MXNet and TensorFlow

Distributed training for deep learning models can be made easier and more scalable than ever before by using Apache MXNet in conjunction with Amazon Web Service's (AWS) Elastic MapReduce (EMR). The following information will help you create an EMR cluster with an AWS Deep Learning Amazon Machine Image (AMI). This package provides scripts to configure an EMR cluster with AWS DL AMI on a CPU/single-GPU/multi-GPU cluster of machines and manage Distributed Deep Learning training jobs.

Deep Learning is an area of Machine Learning that learns patterns and objectives using neural networks with deep hidden layers. Training a deep learning model involves many computationally intensive mathematical operations. These operations can be parallelized on AWS instances with GPUs, where each GPU has thousands of cores that excel at mathematical computations. A CPU-only instance's number of cores isn't directly comparable to a similar number of GPU cores. Nevertheless, for deep learning, GPUs will consistently outperform CPUs by orders of magnitude, and in some cases by over 100x. Even with this performance boost, with the vast amount of data now available, training a complex model on a single GPU or even a single instance with multiple GPUs can take days, or even weeks, to complete. Another problem can occur as well: the number of model parameters can exceed the available memory of a single machine.

## Deep Learning AMIs and Amazon EMR

To overcome the challenges in distributed training, one possible solution is to use a cluster of GPU instances. Deep learning frameworks such as Apache MXNet and TensorFlow readily support such a setup through **data parallelism**  and **model parallelism**. To manually setup and manage this kind of cluster is time-consuming and hard to manage. AWS provides a couple of solutions:

1) [Deep Learning Amazon Machine Images (DL AMIs)](https://aws.amazon.com/amazon-ai/amis/) and the [Cloud Formation (CFN) Deep Learning template](https://github.com/awslabs/deeplearning-cfn): The DL AMIs are pre-configured with popular deep learning frameworks, including MXNet, Caffe, Caffe2, TensorFlow, Theano, CNTK, Torch, and Keras, as well as launch configuration tools that enable easy integration with AWS. You can quickly and easily launch a cluster of DL AMIs with the CFN Deep Learning template.

2) [Amazon EMR](https://aws.amazon.com/emr/): A web service which uses Hadoop to quickly and cost-effectively process vast amounts of data across dynamically scalable Amazon EC2 instances. You can also run other popular distributed frameworks such as Hadoop, Apache Spark, Ganglia, etc. Amazon EMR has recently launched a new feature to launch clusters with a [Custom AMI](http://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-custom-ami.html). Most importantly, this feature gives you the ability to also use the previously mentioned [DL AMIs](https://aws.amazon.com/amazon-ai/amis/) on an EMR cluster.

The second option will be our focus here as it gives user the ability to use big data tools with deep learning frameworks easily. We will discuss how to use this solution and do distributed training with Apache MXNet and TensorFlow.

## Creating an EMR Cluster with a Deep Learning AMI:

This section will help you create an Elastic MapReduce (EMR) cluster with a deep learning Amazon Machine Image (AMI).

**Note:** Following steps only works for "Uniform Instance Groups" configuration option of AWS EMR. To know more about configuration options, refer to [EMR Instance Group Configurations](http://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-instance-group-configuration.html).  

#### Helper Scripts

We will facilitate this process by providing helper scripts, [metadata_setup.sh](metadata_setup.sh) to setup the metadata that is needed for distributed training using Apache MXNet and TensorFlow on the cluster. More detail about the metadata and how to run the script is described later in this document.

Additionally, you can also use [Amazon Elastic File System (Amazon EFS)](https://aws.amazon.com/efs/) attached to the cluster with EMR by using the [emr_efs_bootstrap.sh](emr_efs_bootstrap.sh) script provided in this repository. More info on how to use the script and how to setup EFS is described later in this document.


#### Amazon EFS Setup (Optional)

**You can safely skip this section if you don't want EFS support on EMR cluster.**

You can use EMR configured with S3 with the DL AMIs to do distributed training without using EFS. That being said, there are benefits and motivations for using EFS:

- EFS is automatically mounted on all worker instances during startup.
- EFS allows sharing of code, data, and results across worker instances.
- Using EFS doesn't degrade the performance for densely packed files (for example, .rec files containing image data used in MXNet).

**Prerequisites:**

1) **Create EFS:** Create a new Elastic File System using the [Amazon EFS Console](https://console.aws.amazon.com/efs/).

2) **Mount targets creation:** After you create a file system, you need to [create mount targets](http://docs.aws.amazon.com/efs/latest/ug/accessing-fs.html) in the **same Virtual Private Cloud (VPC)** that you wish to create the EMR cluster.
It is recommended that you create mount targets in all availability zones of the VPC.

**Warning! "If you haven't created mount targets in same VPC as the EMR cluster, cluster creation will fail."**

### Creating EMR cluster with DL AMI using Console

Refer to the [AWS EMR instructions](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs.html) to setup EMR cluster with a Custom AMI. There are several prerequisites such as having a ready-to-use S3 bucket and EC2 keys that you need to attend to prior to moving forward.

Aside from the AWS EMR instructions, you can follow the checklist below to launch the cluster for use with a DL AMI:

1) Open the [Amazon EMR console](https://console.aws.amazon.com/elasticmapreduce/)
2) Choose Create cluster, Go to advanced options.
3) Under **Step 1: Software and Steps**, for Release, choose emr-5.7.0 or later version. **Please select only required services. There is a known bug with launching Hue and Zeppelin which will be solved in next release.** Choose Next.
4) Under **Step 2: Hardware**, select "Instance group configuration" as "Uniform Instance Groups" and make sure to set "Root device EBS volume size" more than 50GB otherwise the cluster might fail.
5) Under **Step 3: General Cluster Settings**, give a cluster name. Under "Additional Options", provide the Amazon Linux DL AMI ID. Find the latest region specific DL AMI IDs on [AWS Marketplace](https://aws.amazon.com/marketplace/pp/B01M0AXXQB).

**(Optional) Mounting EFS via bootstrap action:**

6) Attach "AmazonElasticFileSystemReadOnlyAccess" managed policy to "EMR_EC2_DefaultRole" in [Identity and Access Management (IAM) tool](https://console.aws.amazon.com/iam/). This is needed for "DescribeFileSystems" and "DescribeMountTargets" commands to work.
7) Under **Step 3: General Cluster Settings:** Considering you have successfully completed prerequisites to mount EFS, "Under Bootstrap Actions", select "custom action" then click on "Configure and add" to specify the Script location as `s3://aws-dl-emr-bootstrap/emr_efs_bootstrap.sh`, and **provide space seperated EFS ID and region(e.g. us-east-1) in which EFS is created as arguments. You can find EFS ID and region on EFS console under your EFS name."**
This script will mount EFS on all instances of the cluster.
8) Under **Step 4: Security:** Click on "EC2 Security groups", provide the security group associated with your EFS in [Additional Security Groups](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-additional-sec-groups.html) for both Master and Core type of nodes.

### Creating EMR cluster with DL AMI using [AWS Command Line Interface (AWS CLI)](https://aws.amazon.com/cli/) (Optional)

* Install [AWS CLI](http://docs.aws.amazon.com/cli/latest/userguide/awscli-install-bundle.html).
* Create an IAM User if doesn't exist already.
* Attach following managed policies if not already attached
  * AmazonElasticFileSystemFullAccess
  * AmazonElasticMapReduceFullAccess
  * AmazonS3ReadOnlyAccess
  * AmazonEC2FullAccess
* Configure AWS CLI with Access Key, Secret Key and Region by using "aws configure" command.
* Create EFS from Console - Note down EFS ID and region.
* Create EMR cluster with Custom AMI using following command:

```
aws emr create-cluster \
--release-label emr-5.8.0 \
--instance-type g2.2xlarge \
--instance-count ["$instance-count"] \
--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,KeyName=["$your-ssh-key"] \
--service-role EMR_DefaultRole \
--custom-ami-id ["$ami-id"] \
--log-uri "s3://your-logging-bucket" \
--name ["$cluster-name"] \
--region ["$region-id"]
```

Note that you should set your own values for each of the parameters. The example above would spin up "$instance-count" number of GPU instances (g2.2xlarge), and is expecting an SSH key, an AMI ID, an S3 bucket URI, a cluster name, and a region ID.

Mounting EFS via bootstrap action:

- Attach "AmazonElasticFileSystemReadOnlyAccess" managed policy to "EMR_EC2_DefaultRole" in Identity and Access Management(IAM) tool. This is needed for "DescribeFileSystems" and "DescribeMountTargets" commands to work.
- Replace `sg-masterId` with your existing master security group, `sg-slaveId` with slave security group and `sg-EFS` with EFS mount target security group.
- If you don't have master and slave security groups already created, you can create empty security groups under same VPC where you want to launch the EMR cluster and give those IDs as `sg-masterId` and `sg-slaveId` as follows. EMR automatically updates the security groups as needed.
```
aws ec2 create-security-group --group-name ["$group-name"] --description "Master security group" --vpc-id ["$vpc-id"]
```

- Pass bootstrap script location as `s3://aws-dl-emr-bootstrap/emr_efs_bootstrap.sh` by adding the --bootstrap-action parameter then provide EFS id and region in which EFS is created as arguments to the script as follows:

```
aws emr create-cluster \
--release-label emr-5.8.0 \
--instance-type g2.2xlarge \
--instance-count ["$instance-count"] \
--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,KeyName=["$your-ssh-key"],\
EmrManagedMasterSecurityGroup=["$sg-masterId"],EmrManagedSlaveSecurityGroup=["$sg-slaveId"],\
AdditionalMasterSecurityGroups=["$sg-EFS"],AdditionalSlaveSecurityGroups=["$sg-EFS"] \
--service-role EMR_DefaultRole \
--custom-ami-id ["$ami-id"] \
--bootstrap-actions Path="s3://aws-dl-emr-bootstrap/emr_efs_bootstrap.sh",Args=["$fs-id","$region-id"] \
--log-uri "S3://your-logging-bucket" \
--name ["$cluster-name"] \
--region ["$region-id"]
```

After following above steps, EFS will be mounted on the instances of your cluster in the `/efs` folder. Now, you can easily share code, data, and results across all the instances.

### Troubleshooting

1) If the cluster is stuck in "Starting" stage for more than 20 minutes, check your configuration again. For example, make sure you have provided "ebs root size" to be greater than 50GB. Make sure you haven't selected "Hue" or "Zeppelin" applications while launching the cluster.
2) If the cluster fails to launch with error "bootstrap action 1 returned a non-zero return code", check the bootstrap logs on the S3 bucket. It will be in S3 bucket similar to this: `S3/aws-logs-us-east-1/elasticmapreduce/j-cluster-id/node/i-instance-id/bootstrap-actions/1/stderr`.
3) For other types of failures, check the logs saved in the S3 bucket.

## MXNet Distributed Training on EMR

### SSH to the master node with agent forwarding enabled:

**Before trying to ssh to master node of the cluster, make sure to enable inbound SSH traffic from your IP address to master instance. You can check the "ElasticMapReduce-master" security group to add SSH rule for inbound connections.**

SSH agent forwarding securely connects the instances within the VPC, which are connected to the private subnet. To set up and use SSH agent forwarding, see [Securely Connect to Linux Instances Running in a Private Amazon VPC](https://aws.amazon.com/blogs/security/securely-connect-to-linux-instances-running-in-a-private-amazon-vpc/).
When connecting on MacOS vs. Windows, note that with Mac you should use the -A switch.

On MacOS:
  ```
  ssh -A hadoop@<MASTER-WORKER-PUBLIC-DNS/IP>
  ```

On Windows:
  ```
  ssh hadoop@<MASTER-WORKER-PUBLIC-DNS/IP>
  ```  

To test the ssh-agent forwarding, you can try to ssh to worker nodes from the master node.

**OR if you have done above step you don't need to do following steps**

SSH agent forwarding makes communication between workers possible. Alternatively, you can also setup passwordless ssh between nodes as follows:
```
scp -i your-ssh-key.pem your-ssh-key.pem hadoop@worker-ip-address:~/
```
SSH to master node and run following commands:

```
ssh-agent bash
ssh-add your-ssh-key.pem
ssh-keygen [enter enter]
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod og-wx ~/.ssh/authorized_keys
ssh-copy-id -i ~/.ssh/id_rsa.pub hadoop@IP_OF_SLAVE1
ssh-copy-id -i ~/.ssh/id_rsa.pub hadoop@IP_OF_SLAVE2
```

### Running the Metadata Script

**Make sure you have all the nodes available by checking in “Hardware” tab of created EMR cluster on the console before moving further.**

Download the script in the "home" folder of the master node and run it as follows:

  ```
  wget https://s3.amazonaws.com/aws-dl-emr-bootstrap/metadata_setup.sh -P /home/hadoop/
  chmod +x metadata_setup.sh
  source metadata_setup.sh
  ```

This script sets up the environment variables on the master node:

**$DEEPLEARNING_WORKERS_COUNT:** The total number of workers.

**$DEEPLEARNING_WORKERS_PATH:** The file path that contains the list of workers in the cluster.

**$DEEPLEARNING_WORKER_GPU_COUNT:** The number of GPUs on the instance.

Apart from running above script, you should give "hadoop" user permissions to the "ec2-user" folder on all the nodes as follows:

```
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "sudo chown -R hadoop /home/ec2-user/" ; done 10<$DEEPLEARNING_WORKERS_PATH
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "sudo chmod -R u+rwx /home/ec2-user/" ; done 10<$DEEPLEARNING_WORKERS_PATH
```

### Running Distributed Training with MXNet

Now, we have the required setup to run distributed training. We will use [MXNet CIFAR-10 example](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/train_cifar10.py).
The following example shows how to run CIFAR-10 with data parallelism on MXNet. Note the use of the `DEEPLEARNING_*` environment variables.

```
#terminate all running Python processes across workers
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "pkill -f python" ; \
done 10<$DEEPLEARNING_WORKERS_PATH

#navigate to the MXNet image-classification example directory \
cd /home/ec2-user/src/mxnet/example/image-classification/

#run the CIFAR10 distributed training example \
../../tools/launch.py -n $DEEPLEARNING_WORKERS_COUNT -H $DEEPLEARNING_WORKERS_PATH \
python train_cifar10.py --gpus $(seq -s , 0 1 $(($DEEPLEARNING_WORKER_GPU_COUNT - 1))) \
--network resnet --num-layers 50 --kv-store dist_device_sync
```

You will see samples/sec, speed, and accuracy for each epoch as the output.

These steps summarize how to get started. For more information about running distributed training on MXNet, see [Run MXNet on Multiple Devices](http://mxnet.readthedocs.io/en/latest/how_to/multi_devices.html).

## Using MXNet with S3 on EMR

If you have EMR cluster, it is very natural to store your datasets on S3. Apache MXNet is deeply integrated with S3 for this purpose, and MXNet on DL AMI comes with S3 integration enabled. You need to follow these steps to use data in S3 bucket:

- **Configure S3 authentication tokens:**
MXNet requires the S3 environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to be set. You may set them as follows:

```
export AWS_ACCESS_KEY_ID=<your-access-key-id>
export AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
```

- **Upload data to S3:**
There are several ways to upload data to S3. One easy way is to use the AWS CLI.

```
aws s3 cp training-data s3://bucket-name/training-data
```

- **Train with data from S3:**
Once the data is in S3, it's pretty easy to make your program use training/validation data from S3. For example, you can change the data iterator, `train_dataiter`, in training script as follows:

```
train_dataiter = mx.io.MNISTIter(
             image="s3://bucket-name/training-data/train-images-idx3-ubyte",
             label="s3://bucket-name/training-data/train-labels-idx1-ubyte",
             data_shape=(1, 28, 28),
             label_name='sm_label',
             batch_size=batch_size, shuffle=True, flat=False, silent=False, seed=10)
```

Also, if you are running the training script on EMR cluster without EFS you need to make this change to the script on all of the nodes in the cluster.

To run the same [train_cifar10.py](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/train_cifar10.py) example with a dataset on S3, upload the [CIFAR_10 dataset rec files](http://data.mxnet.io/data/cifar10/) to an S3 bucket.
You can then make the following change to the script, while also updating the S3 bucket location, on all of the nodes:

```
[hadoop@ip image-classification]$ git diff train_cifar10.py
diff --git a/example/image-classification/train_cifar10.py b/example/image-classification/train_cifar10.py
old mode 100644
new mode 100755
index 0186233..edd26aa
--- a/example/image-classification/train_cifar10.py
+++ b/example/image-classification/train_cifar10.py
@@ -16,8 +16,13 @@ def download_cifar10():

 if __name__ == '__main__':
     # download data
-    (train_fname, val_fname) = download_cifar10()
+    train_fname='s3://bucket-name/training-data/cifar10-data/cifar10_train.rec'
+    val_fname='s3://bucket-name/training-data/cifar10-data/cifar10_val.rec'
```

## TensorFlow Distributed Training on EMR

If you run TensorFlow on an EMR-5.8.0 cluster with a DL AMI, you might see errors like "ImportError: numpy.core.multiarray failed to import". To solve this error, reinstall numpy on all the nodes in the cluster as follows:

```
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "sudo pip uninstall numpy" ; done 10<$DEEPLEARNING_WORKERS_PATH
while read -u 10 host; do ssh -o "StrictHostKeyChecking no" $host "sudo easy_install -U numpy" ; done 10<$DEEPLEARNING_WORKERS_PATH
```

This issue should be resolved in next version, EMR-5.9.0.

If you are not using shared file storage on the cluster like HDFS/EFS, make sure that the code and data are available to all of the machines in the cluster.

You should also copy `worker_ip_file` to all other nodes in the cluster. This file is created after running `metadata_setup.sh` on the master node; it contains a list of the IP address of the nodes in the cluster. This step is needed to run distributed training with TensorFlow.

```
while read -u 10 host; do scp -p $DEEPLEARNING_WORKERS_PATH hadoop@$host:$DEEPLEARNING_WORKERS_PATH ; done 10<$DEEPLEARNING_WORKERS_PATH
```

If you are using EFS on the EMR cluster, checkout the TensorFlow distributed training example in [deeplearning-cfn](https://github.com/awslabs/deeplearning-cfn/blob/master/README.md#running-distributed-training-on-tensorflow).
