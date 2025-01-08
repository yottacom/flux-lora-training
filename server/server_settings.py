import os
# get base directory of , like the directory of project

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR,"datasets")

GCLOUD_PROJECT_ID = os.environ.get("GCLOUD_PROJECT_ID")
GCLOUD_BUCKET_NAME = os.environ.get("GCLOUD_BUCKET_NAME")
GCLOUD_PUB_SUB_SUBSCRIPTION = os.environ.get("GCLOUD_PUB_SUB_SUBSCRIPTION")
GCLOUD_STORAGE_CREDENTIALS = "gcloud-storage-credentials.json"
GCLOUD_PUBSUB_CREDENTIALS = "gcloud-pubsub-credentials.json"
HF_TOKEN=os.environ.get("HF_TOKEN")
IDLE_TIME_IN_SECONDS= int(os.environ.get("IDLE_TIME_IN_SECONDS",60))


RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")