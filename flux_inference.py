import uuid
import torch
from diffusers import FluxPipeline
from server.request_queue import Job
from server.gcloud_utils import upload
from server.utils import webhook_response


def generate(job: Job):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    lora_path = job.job_results[-1].saved_checkout_path
    pipe.load_lora_weights(lora_path)

    results = []
    # pipe.fuse_lora(lora_scale=1.5)
    for prompt in job.job_request.example_prompts:
        image = pipe(
            prompt, num_inference_steps=50, guidance_scale=3.5, width=1024, height=832
        ).images[0]
        image_path = f"{str(uuid.uuid4())}.png"
        image.save(image_path)
        gcs_path = upload(image_path, f"inference/{job.job_id}/")
        webhook_response(
            job.job_request.inference_webhook_url,
            True,
            200,
            "Job Completed",
            {"job_id": job.job_id, "image_path": gcs_path},
        )
        results.append(gcs_path)
    return results
