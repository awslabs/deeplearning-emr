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

import pyspark
from pyspark import SparkConf
import logging
log_fmt = '%(asctime)s -  %(name)s - %(levelname)s %(process)d %(funcName)s:%(lineno)d %(message)s'
logging.basicConfig(format=log_fmt, level=logging.INFO)

logger = logging.getLogger(__name__)
logger.debug ('All modules imported')

def main():

    # Note: It is important to import the libraries needed within the function
    #      so Spark does not attempt serializing the libraries to all the workers,
    #      otherwise it could fail during Serialization/Deserialization
    #      using the pickle methods.

    from mxinfer import load_images
    from mxinfer import predict

    from utils import get_args
    from utils import get_s3client
    from utils import fetch_s3_keys
    from utils import download_objects
    from utils import upload_file

    args = get_args()
    logger.info('received arguments:{}'.format(args))

    conf = SparkConf().setAppName("Distributed Inference using MXNet and Spark")

    # we will set the number of cores per executor to 1 to force Spark to create
    # only one task per executor since MXNet efficiently uses all the cpus on the
    # system for inference
    conf.set('spark.executor.cores', '1')

    sc = pyspark.SparkContext(conf=conf)
    logger.info("Spark Context created")


    s3_client = get_s3client(args['access_key'], args['secret_key'])

    keys = fetch_s3_keys(args['bucket'], args['prefix'], s3_client)

    # filter out only png images.
    # you can also choose to check the content-type headers by doing
    # a head call against each S3-Key

    keys = filter(lambda x: x.endswith('.png'), keys)

    # number of keys
    n_keys = len(keys)
    if n_keys < args['batch']:
        args['batch'] = n_keys

    n_partitions = n_keys // args['batch']

    logger.info('number of keys from s3: {}'.format(n_keys))

    # if keys cannot be divided by args['batch'] .
    if (n_partitions * args['batch'] != n_keys):
        keys.extend(keys[: args['batch'] - (n_keys - n_partitions * args['batch'])])

    logger.debug('Keys:{}'.format(keys))

    n_partitions = len(keys) // args['batch']
    logger.info("number of keys:{}, n_partitions:{}".format(len(keys), n_partitions))

    # we will create partitions of args['batch']
    rdd = sc.parallelize(keys, numSlices=n_partitions)
    logger.info('created rdd with {} partitions'.format(rdd.getNumPartitions()))

    sc.broadcast(args['bucket'])

    rdd = rdd.mapPartitions(lambda k : download_objects(args['bucket'], k))

    rdd = rdd.mapPartitions(load_images)

    sc.broadcast(args)
    rdd = rdd.mapPartitions(lambda imgs: predict(imgs, args))

    output = rdd.collect()

    # drop the extra keys that we added to fill the last batch

    keys = keys[:n_keys]
    output = output[:n_keys]

    logger.info("predictions: {}".format(output))

    if args['output_s3_key'] and args['output_s3_bucket']:
        with open('/tmp/' + args['output_s3_key'] , 'w+') as f:
            for k, o in zip(keys, output):
                f.write("Key %s: Prediction: %s\n" % (k, o))
        upload_file(args['output_s3_bucket'], args['output_s3_key'], '/tmp/' + args['output_s3_key'], s3_client)

if __name__ == '__main__':
    main()
