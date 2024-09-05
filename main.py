import os
from slugify import slugify
import requests as http_requests
from server.request_queue import TrainingRequest, Job
from server.utils import (
    save_images_and_generate_metadata,
    generate_config_file,
    webhook_response,
)
from server.request_processor import process_request


def train(training_request_dict: dict):
    job=None
    try:
        training_request_defaults = TrainingRequest()
        job_id = training_request_dict.get("job_id")
        lora_name = training_request_dict.get("lora_name")
        example_prompts = training_request_dict.get(
            "example_prompts", training_request_defaults.example_prompts
        )
        webhook_url = str(
            training_request_dict.get(
                "webhook_url", training_request_defaults.webhook_url
            )
        )
        images_urls = training_request_dict.get("images_urls", [])

        if not job_id:
            return webhook_response(webhook_url, False, 400, "No job id provided!")
        if not lora_name:
            return webhook_response(webhook_url, False, 400, "No lora name provided!")
        if len(images_urls) == 0:
            return webhook_response(webhook_url, False, 400, "No image urls provided!")

        dataset_path = save_images_and_generate_metadata(job_id, images_urls, lora_name)

        training_request = TrainingRequest()
        training_request.lora_name = lora_name
        training_request.images_urls = images_urls
        training_request.dataset_folder = dataset_path
        training_request.webhook_url = webhook_url
        training_request.example_prompts = example_prompts
        config_file_path = generate_config_file(training_request)
        training_request.config_file = config_file_path

        print("Config File generated successfully!", config_file_path)
        job = Job(job_id=job_id, job_request=training_request, job_epochs=10)
        process_request(job)
        webhook_response(
            job.job_request.webhook_url, True, 200, "Job Completed", job.dict()
        )
    except Exception as ex:
        print(ex)
        webhook_response(webhook_url, False, 500, str(ex),None if job is None else job.dict())
        raise Exception(ex)


training_request_dict = {
    "job_id": "alkjdiersdkjk",
    "lora_name": "Irfan",
    "webhook_url": "https://webhook.site/6292251e-6223-4091-8e32-8ca191b2ede6",
    "images_urls": [
        "https://i.ibb.co/gJnQY2P/5.jpg",
        "https://i.ibb.co/pRg3FXj/8.jpg",
        "https://i.ibb.co/ZYXfQjR/12.jpg",
        "https://i.ibb.co/7yzxRHd/13.jpg",
        "https://i.ibb.co/R6MqCcV/14.jpg",
        "https://i.ibb.co/prQwJ55/22.jpg",
        "https://i.ibb.co/TTQfVSY/25.jpg",
        "https://i.ibb.co/QFDsytx/30.jpg",
        "https://i.ibb.co/hRxDx5z/34.jpg",
        "https://i.ibb.co/mCGnTkX/48.jpg",
    ],
}
train(training_request_dict)
