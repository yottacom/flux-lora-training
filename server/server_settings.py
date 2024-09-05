import os
from decouple import config

AWS_ACCESS_KEY=config("AWS_ACCESS_KEY")
AWS_SECRET_KEY=config("AWS_SECRET_KEY")
AWS_REGION=config("AWS_REGION")
AWS_BUCKET_NAME=config("AWS_BUCKET_NAME")

BASE_DIR = "/var/www/flux-lora-training"
DATASET_DIR = os.path.join(BASE_DIR,"datasets")