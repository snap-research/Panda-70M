from typing import List

import boto3


def list_bucket_keys(bucket_name: str, prefix: str=None) -> List[str]:
    """
    Lists the keys of all objects in an Amazon S3 bucket
    :param bucket_name: The name of the bucket to list
    :param prefix: only videos starting with the given prefix are returned
    :return:
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    all_bucket_contents = []
    for bucket_object in bucket.objects.all():
        bucket_object_key = bucket_object.key
        if prefix is None or bucket_object_key.startswith(prefix):
            all_bucket_contents.append(bucket_object_key)

    return list(sorted(all_bucket_contents))


def list_bucket_keys_v2(bucket_name: str, prefix: str=None) -> List[str]:
    """
    Lists the keys of all objects in an Amazon S3 bucket
    :param bucket_name: The name of the bucket to list
    :param prefix: only videos starting with the given prefix are returned
    :return:
    """
    if prefix is None:
        prefix = ""

    reached_end = False
    all_keys = []
    token = None
    first = True
    s3 = boto3.client('s3')

    while not reached_end:
        # will limit to 1000 objects
        if first:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            first = False
        else:
            response = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=token, Prefix=prefix)
        token = response.get('NextContinuationToken', None)
        contents = response.get('Contents', [])
        all_keys.extend([obj['Key'] for obj in contents])

        if not token and response.get('IsTruncated') is False:
            reached_end = True
    return list(sorted(all_keys))


def bucket_keys_to_file(bucket_name: str, output_filename: str, prefix: str="") -> List[str]:
    """
    Writes all the keys of a bucket to the given file
    param bucket_name: name of the bucket
    output_filename: name of the output file
    """
    with open(output_filename, "w") as outfile:

        reached_end = False
        all_keys = []
        token = None
        first = True
        s3 = boto3.client('s3')
        iteration = 0

        while not reached_end:
            if first:
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                first = False
            else:
                response = s3.list_objects_v2(Bucket=bucket_name, ContinuationToken=token, Prefix=prefix)
            token = response.get('NextContinuationToken', None)
            contents = response.get('Contents', [])

            for obj in contents:
                current_key = obj["Key"]
                outfile.write(current_key + "\n")
            if iteration % 100 == 0:
                outfile.flush()

            all_keys.extend([obj['Key'] for obj in contents])

            iteration += 1
            if not token and response.get('IsTruncated') is False:
                reached_end = True

        return list(sorted(all_keys))


def bucket_keys_from_file(filename: str, prefix: str=None, max_videos: int=None) -> List[str]:
    """
    Retrieves the object keys in a bucket from the entries of a file
    """
    all_object_keys = []
    with open(filename, "r") as infile:
        for current_object_key in infile:
            current_object_key = current_object_key.strip()
            if len(current_object_key) > 0 and (prefix is None or current_object_key.startswith(prefix)):
                all_object_keys.append(current_object_key)
            if max_videos is not None and len(all_object_keys) >= max_videos:
                break

    return list(sorted(all_object_keys))





