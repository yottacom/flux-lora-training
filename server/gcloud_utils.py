import uuid
from google.cloud import storage
import server.server_settings as settings


def upload(path, bucket_path):
    try:
        client = storage.Client.from_service_account_json(settings.GCLOUD_CREDENTIALS)
        bucket = client.bucket(settings.GCLOUD_BUCKET_NAME)
        image_name = f"{uuid.uuid4()}.png"
        full_image_path = f"{bucket_path}{image_name}"
        blob = bucket.blob(full_image_path)
        blob.upload_from_filename(path, content_type='image/png')
        return full_image_path
    except Exception as e:
        print(f"An error occurred while uploading the image: {e}")
        return None