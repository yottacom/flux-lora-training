import os
import re
import copy
import select
import subprocess
import uuid
import requests
from threading import Thread
from PIL import Image
import server.server_settings as server_settings
from server.request_queue import (
    Job,
    TrainingRequest,
    ModelTypes,
    TrainingResponse,
    JobStatus,
)
from server.utils import webhook_response
from server.gcloud_utils import upload


def background_training(job: Job):
    yaml_path = job.job_request.config_file
    # command = f"bash -c 'cd {server_settings.BASE_DIR} && source venv/bin/activate && python -u run.py {yaml_path}'"
    command = f"bash -c 'export HF_TOKEN={server_settings.HF_TOKEN} && export HF_HOME=/workspace/cache/ && python3 -u run.py {yaml_path}'"
    webhook_response(
        job.job_request.training_webhook_url,
        True,
        200,
        f"Going to execute command {command}",
        job.dict(),
    )
    print(command)

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Use select to read from stdout and stderr without blocking
        logs_count = 0
        safetensors_files = set()
        stderr_output = []
        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [])
            for fd in ret[0]:
                if fd == process.stdout.fileno():
                    read = process.stdout.readline()
                    if read:
                        output = read.strip()
                        print(output)
                        percentage = 0
                        if f"/{job.job_request.steps}" in output:
                            percentage_value = get_progress_percentage(output)
                            percentage = percentage_value if percentage_value else 0
                            logs_count += 1
                        job.job_progress = percentage
                        if logs_count % 40 == 0:
                            webhook_response(
                                job.job_request.training_webhook_url,
                                True,
                                200,
                                "Job Progress",
                                job.dict(),
                            )
                            process_response(job, safetensors_files)
                if fd == process.stderr.fileno():
                    read = process.stderr.readline()
                    if read:
                        output = read.strip()
                        print(output)
                        stderr_output.append(output)
                        percentage = 0
                        if f"/{job.job_request.steps}" in output:
                            percentage_value = get_progress_percentage(output)
                            percentage = percentage_value if percentage_value else 0
                            logs_count += 1
                        job.job_progress = percentage
                        if logs_count % 40 == 0:
                            webhook_response(
                                job.job_request.training_webhook_url,
                                True,
                                200,
                                "Job Progress",
                                job.dict(),
                            )
                            process_response(job, safetensors_files)
                print(job.job_progress)
            if process.poll() is not None:
                break

        return_code = process.poll()
        if return_code != 0:
            stderr_combined = "\n".join(stderr_output)
            raise subprocess.CalledProcessError(
                return_code, command, output=stderr_combined
            )

        print("Job is Finished")
        job.job_progress = 100
        job.job_status = JobStatus.FINISHED.value
        process_response(job, safetensors_files)

    except subprocess.CalledProcessError as e:
        print(e)
        job.job_status = JobStatus.FAILED.value
        job.error_message = str(e)
        webhook_response(
            job.job_request.training_webhook_url, False, 500, str(e), job.dict()
        )
        raise Exception(str(e))

    except Exception as e:
        print(e)
        job.job_status = JobStatus.FAILED.value
        job.error_message = str(e)
        webhook_response(
            job.job_request.training_webhook_url, False, 500, str(e), job.dict()
        )
        raise Exception(str(e))


def process_request(job: Job):
    job.job_status = JobStatus.PROCESSING.value
    webhook_response(
        job.job_request.training_webhook_url, True, 200, "Job Started", job.dict()
    )
    background_training(job)


def get_progress_percentage(output_line):
    # Regex pattern to find the diffusion process progress
    pattern = re.compile(r"(\d+)/(\d+)\s+\[")
    match = pattern.search(output_line)
    if match:
        current, total = map(int, match.groups())
        percentage = (current / total) * 100
        return percentage
    return None


def process_response(job: Job, safetensors_files: set):
    safetensors_files_path = os.path.join(
        job.job_request.dataset_folder, job.job_request.lora_name
    )
    current_files = (
        {f for f in os.listdir(safetensors_files_path) if f.endswith(".safetensors")}
        if os.path.exists(safetensors_files_path)
        else set()
    )
    new_files = current_files - safetensors_files
    if new_files:
        for new_file in new_files:
            print(f"New file found: {new_file}")
            epoch_response = TrainingResponse(
                current_epoch_id=str(uuid.uuid4()),
                total_epochs=job.job_epochs,
                current_epoch_number=len(job.job_results) + 1,
            )
            saved_checkout_path = os.path.join(safetensors_files_path, new_file)
            epoch_response.saved_checkout_path = saved_checkout_path
            print("Local Path of uploaded model is ", saved_checkout_path)
            epoch_response.cloud_storage_path = upload(
                saved_checkout_path, job.job_s3_folder
            )
            job.job_results.append(epoch_response)
            webhook_response(
                job.job_request.training_webhook_url,
                True,
                200,
                "Epoch Completed",
                job.dict(),
            )
        safetensors_files.update(new_files)
