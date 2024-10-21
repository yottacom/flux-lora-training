import json
import argparse
from server.request_queue import TrainingRequest, Job
from server.utils import (
    save_images_and_generate_metadata,
    generate_config_file,
    webhook_response,
)
from server.request_processor import process_request
from flux_inference import generate

def train(training_request_dict: dict):
    job=None
    try:
        training_request_defaults = TrainingRequest()
        job_id = training_request_dict.get("job_id")
        lora_name = training_request_dict.get("lora_name")
        example_prompts =[]
        training_webhook_url = str(
            training_request_dict.get(
                "training_webhook_url", training_request_defaults.training_webhook_url
            )
        )
        inference_webhook_url = str(
            training_request_dict.get(
                "inference_webhook_url", training_request_defaults.inference_webhook_url
            )
        )
        if not training_webhook_url:
            print("No training webhook url found!")
        if not inference_webhook_url:
            print("No inference webhook url found!")
        
        images_urls = training_request_dict.get("images_urls", [])

        if not job_id:
            return webhook_response(training_webhook_url, False, 400, "No job id provided!")
        if not lora_name:
            return webhook_response(training_webhook_url, False, 400, "No lora name provided!")
        if len(images_urls) == 0:
            return webhook_response(training_webhook_url, False, 400, "No image urls provided!")

        dataset_path = save_images_and_generate_metadata(job_id, images_urls, lora_name)

        training_request = TrainingRequest()
        training_request.lora_name = lora_name
        training_request.images_urls = images_urls
        training_request.dataset_folder = dataset_path
        training_request.training_webhook_url = training_webhook_url
        training_request.inference_webhook_url = inference_webhook_url
        training_request.example_prompts = example_prompts
        config_file_path = generate_config_file(training_request)
        training_request.config_file = config_file_path

        print("Config File generated successfully!", config_file_path)
        job = Job(job_id=job_id, job_request=training_request, job_epochs=10)
        process_request(job)
        webhook_response(
            job.job_request.training_webhook_url, True, 200, "Job Completed", job.dict()
        )
        job.job_request.example_prompts = training_request_dict.get(
            "example_prompts", training_request_defaults.example_prompts
        )
        return job
    except Exception as ex:
        print(ex)
        webhook_response(training_webhook_url, False, 500, str(ex),None if job is None else job.dict())
        raise Exception(ex)

def parse_args():
    parser = argparse.ArgumentParser(description="Training Job")
    parser.add_argument('--request', type=str, required=True, help='Training request JSON string')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    request_dict = json.loads(args.request)

    training_job = train(request_dict)
    inference_results = generate(training_job)

# job_id="818cfa9f-8a94-4123-b15c-a0ad097df7d6"
# training_request_dict = {
#     "job_id": job_id,
#     "lora_name": "Irfan",
#     "training_webhook_url": "https://webhook-test.com/d1fbd61b6e4a6c6a4269c3582384a2f2",
#     "images_urls": [
#         "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/5_nrFxoHy.jpg",
#         "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/8_gqxczDK.jpg",
#         "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/12_Ehg7LdQ.jpg",
#         "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/13_83IOj8I.jpg",
#         "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/14_8YdvYMl.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/15_acbmB79.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/22_6NwVFcf.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/25_MVz9VEl.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/26_RWXkqth.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/30_s1X8bgP.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/34_q44pwlG.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/36_S2nOmpF.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/41_gUwvrQy.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/43_izBe2Fn.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/46_3NYQWjf.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/47_nTcZgDy.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/48_68j55v4.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/49_L6Y9j7H.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/50_KlJ4G0k.jpg",
#         # "https://boothybooth.s3.amazonaws.com/lora_images/IrfanFlux/56_BVFsEb8.jpg"
#     ],
#     "example_prompts":[
#         "Irfan , young asian man in his 20s , wearing a blue tuxedo and white shirt, standing in a street with blury background, realistic, 4k hd."
#         "Irfan , young asian man in his 20s , wearing a black tuxedo and white shirt, standing in a street with blury background, realistic, 4k hd."
#         "Irfan , young asian man in his 20s , wearing a brown tuxedo and white shirt, standing in a street with blury background, realistic, 4k hd."
#         "Irfan , young asian man in his 20s , wearing a purple tuxedo and white shirt, standing in a street with blury background, realistic, 4k hd."
#     ]
# }
# job = train(training_request_dict)
# generate(job)
