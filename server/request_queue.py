import uuid
from datetime import datetime
from enum import Enum
from typing import List
from pydantic import BaseModel


class ModelTypes(Enum):
    DEV1 = "dev"
    SCHNELL = "schnell"


class TrainingRequest(BaseModel):
    lora_name: str = ""
    images_urls: list = []
    steps: int = 1000
    save_step:int = 200
    learning_rate: float = 5e-4
    rank: int | None = 16
    model: str = ModelTypes.DEV1.value
    low_vram: bool = True
    dataset_folder: str = ""
    config_file: str = ""
    example_prompts: list = []
    training_webhook_url: str | None = None

class JobStatus(Enum):
    WAITING = "waiting"
    PROCESSING = "processing"
    FINISHED = "finished"
    FAILED = "failed"


class TrainingResponse(BaseModel):
    total_epochs: int = 0
    current_epoch_number: int = 0
    current_epoch_id: str = None
    saved_checkout_path:str = ""
    cloud_storage_path:str = ""


class Job(BaseModel):
    job_id: str = None
    job_request: TrainingRequest = None
    # job_config:TrainingConfig = None
    job_number: int = None  # This will be set when added to the queue
    job_progress: int = 0
    job_status: str = JobStatus.WAITING.value  # Initial status
    job_epochs: int = 0
    job_s3_folder: str = None
    job_results: List[TrainingResponse] = []
    error_message: str = None

    def __init__(self, **data):
        super().__init__(**data)
        # Dynamically set job_s3_folder using job_id and current date
        if self.job_id:
            self.job_s3_folder = f"loras/{datetime.now().strftime('%Y-%m-%d')}/{self.job_id}/"


class JobQueue:
    def __init__(self):
        self.pending: List[Job] = []
        self.history: List[Job] = []
        self.last_job_id = None


job_queue = JobQueue()
