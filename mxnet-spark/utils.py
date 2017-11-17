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

import logging
log_fmt = '%(asctime)s -  %(name)s - %(levelname)s %(process)d %(funcName)s:%(lineno)d %(message)s'
logging.basicConfig(format=log_fmt, level=logging.ERROR)
logger = logging.getLogger(__name__)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Inference using MXNet and Spark")
    parser.add_argument('--bucket', help='S3 bucket holding the input files', required=True)
    parser.add_argument('--prefix', help='S3 Key Prefix', required=False, default='')
    
    parser.add_argument('--sym_url', help='Url at which MXNet Model symbol is stored', required=True)
    parser.add_argument('--param_url', help='Url at which MXNet Model symbol is stored', required=True)
    parser.add_argument('--label_url', help='Url at which MXNet Model Synset file is stored', required=True)
    
    parser.add_argument('--batch', type=int, default=32, help='Number of images to process at a time', required=True)
    parser.add_argument('--output_s3_key', help='S3 Key in the bucket to which the predictions should be written',
                        required=False, default='mxinfer_output')

    parser.add_argument('--access_key', help='AWS access Key', required=False, default=None)
    parser.add_argument('--secret_key', help='AWS access secret Key', required=False, default=None)
    
    args = vars(parser.parse_args());
    return args

def get_s3client(access_key=None, secret_key=None):
    """
    Returns a boto3 instantiated s3 client

    If you do not pass the access key and secret key, this routine expects that the
    default profile in ~/.aws/config or the EC2 Instance role has the right set of
    credentials to access this bucket.
    
    """
    import  boto3
    if access_key and secret_key:
        return boto3.client('s3', access_key, secret_key)
    else:
        return boto3.client('s3')

def fetch_s3_keys(bucket, prefix, s3_client):
    """
    Fetch S3 Keys from the given S3 bucket and prefix, return the list.

    Parameters:
    ----------
    bucket: str, mandatory
        Bucket from which keys have to be fetched.
    prefix: str, optional
        Uses the prefix to fetch keys from the bucket.
    s3_client: boto3.client, mandatory
        boto3 s3 client

    Returns
    -------
    list of s3 keys

    """
    # Note: We import the libraries needed within the function so Spark does
    #      not have serialize the libraries to all the workers,
    #      otherwise it could fail during Serialization/Deserialization
    #      using the pickle methods.
    import boto3
    
    all_keys = []
    more_pages = True
    next_page_token = ''
 
    while more_pages:
        if next_page_token == '':
            objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        else:
            objects = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix,
                                                 ContinuationToken=next_page_token)
        if not objects:
            break

        next_page_token = ''

        keys = [object['Key'] for object in objects.get('Contents', [])]
        all_keys.extend(keys)

        more_pages = objects['IsTruncated']
        if 'NextContinuationToken' in objects:
            next_page_token = objects['NextContinuationToken']

    logger.info('fetched {} keys from s3 bucket: {} and prefix:{} '.format(len(all_keys), bucket, prefix))
    logger.debug('fetched keys \n {}'.format(all_keys))

    return all_keys


def download_object(key, bucket):
    """
    This routine downloads the s3 key into memory from the s3_bucket
    """
    s3_client = get_s3client()
    s3_obj = s3_client.get_object(Bucket=bucket, Key=key)
    if s3_obj['ResponseMetadata'] and s3_obj['ResponseMetadata']['HTTPStatusCode'] == 200:
        return bytearray(s3_obj['Body'].read())
    return None

def download_objects(bucket, list_of_keys):
    """
    This routine will download the list of s3 keys 
    in parallel by spawning threads
    """

    import multiprocessing
    from functools import partial
    
    s3_client = get_s3client()
    # spark application(using YARN) will run out of memory if we start to many threads.
    num_threads_to_use = min(100, multiprocessing.cpu_count())

    try:
        pool = multiprocessing.Pool(num_threads_to_use)
        s3_download_obj = partial(download_object, bucket=bucket)
        results = pool.map(s3_download_obj, list_of_keys)
        pool.close()
        pool.join()
        return results
    except Exception as e:
        logging.exception("Error downloading keys")

def upload_file(bucket, key, local_file, s3_client):
    """
    Uploads a given file to the s3 key in the bucket
    """
    import boto3
    s3_client.upload_file(local_file, bucket, key)

    return

