import boto3
import server.server_settings as settings



def upload_to_s3(save_path, object_path):
    """
    Uploads an image to Amazon S3 and returns the object URL

    :param file_obj: File object to upload.
    :param bucket_name: Name of the bucket to upload to.
    :param object_path: Full path (including file name) inside the bucket.
    :return: URL of the uploaded object if successful, else None
    """
    bucket_name = settings.AWS_BUCKET_NAME
    with open(save_path, "rb") as file_obj:

        session = boto3.session.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
            region_name=settings.AWS_REGION,
        )
        bucket_url = settings.AWS_URL

        s3 = session.resource("s3")
        try:
            s3.Bucket(bucket_name).upload_fileobj(file_obj, object_path)
            # Construct the object URL
            object_url = f"{bucket_url}{object_path}"
            return object_url
        except Exception as e:
            print(e)
            return None
