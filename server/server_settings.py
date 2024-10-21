import os
from decouple import config

GCLOUD_BUCKET_NAME = config("GCLOUD_BUCKET_NAME")
GCLOUD_CREDENTIALS = config("GCLOUD_CREDENTIALS")

BASE_DIR =  config("BASE_DIR")
DATASET_DIR = os.path.join(BASE_DIR,"datasets")

