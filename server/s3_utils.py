import os
import boto3
import server.server_settings as settings
from botocore.exceptions import NoCredentialsError


def s3_client_info():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY,
        aws_secret_access_key=settings.AWS_SECRET_KEY,
        region_name=settings.AWS_REGION,
    )
    return s3


def upload_media_to_s3(file_path,file_name):
    print("Going to uploaded to S3")
    s3_client = s3_client_info()
    bucket_name = settings.AWS_BUCKET_NAME
    try:
        if file_path is None:
            return None
        with open(file_path, "rb") as data:
            s3_client.put_object(
                Bucket=bucket_name, Key=file_name, Body=data, ACL="private"
            )
        return file_name
    except NoCredentialsError:
        print("Credentials not available.")
        return None
    except Exception as e:
        print(e)
        return None

def get_uploaded_media_from_s3(file_path):
    s3_client = s3_client_info()
    bucket_name = settings.AWS_BUCKET_NAME
    try:
        if not file_path:
            return None
        signed_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket_name, "Key": file_path},
            ExpiresIn=3600,  # Number of seconds the presigned URL is valid for
        )
        if signed_url:
            return signed_url
    except Exception as e:
        print(e)
        return None