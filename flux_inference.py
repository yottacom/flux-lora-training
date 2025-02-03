import os
import uuid
import torch
import requests
from diffusers import FluxPipeline
from server.request_queue import Job
from server.gcloud_utils import upload
from server.utils import webhook_response


def download_lora(
    url="https://huggingface.co/hamzamfarooqi/aidmaRealisticPeoplePhotograph/resolve/main/aidmaRealisticPeoplePhotograph.safetensors",
    save_path="/workspace/aidmaRealisticPeoplePhotograph.safetensors",
):
    """
    Checks if a file exists at the given path. If it doesn't, downloads the file from the specified URL.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path to save the file.
    """
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}. Skipping download.")
        return
    else:
        print(f"File does not exist at {save_path}. Downloading...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"File downloaded successfully and saved to {save_path}.")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")


def generate(job: Job,lora_url:str=None):
    lora_path="lora.safetensors"
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="/workspace/cache/",
    ).to("cuda")
    if not lora_url:
        lora_path = job.job_results[-1].saved_checkout_path
    else:
        download_lora(url=lora_url,save_path=lora_path)
    pipe.load_lora_weights(lora_path)

    results = []
    # pipe.fuse_lora(lora_scale=1.5)
    for prompt in job.job_request.example_prompts:
        print(prompt)
        image = pipe(
            prompt, num_inference_steps=50, guidance_scale=3.5, width=job.job_request.example_image_width, height=job.job_request.example_image_height
        ).images[0]
        image_path = f"{str(uuid.uuid4())}.png"
        image.save(image_path, compress_level=0)
        gcs_path = upload(image_path, f"inference/{job.job_id}/", ".png")
        webhook_response(
            job.job_request.inference_webhook_url,
            True,
            200,
            "Job Completed",
            {"job_id": job.job_id, "image_path": gcs_path},
        )
        results.append(gcs_path)
    return results
