import os
import re
import copy
import select
import subprocess
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
from server.s3_utils import upload_media_to_s3
from server.utils import webhook_response


def background_training(job: Job):
    yaml_path = job.job_request.config_file
    # command = f"bash -c 'cd {server_settings.BASE_DIR} && source venv/bin/activate && python -u run.py {yaml_path}'"
    command = f"bash -c 'python3 -u run.py {yaml_path}'"
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
                        webhook_response(
                            job.job_request.webhook_url,
                            True,
                            200,
                            "Job Progress",
                            job.dict(),
                        )
                        if logs_count % 20 == 0:
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
                        webhook_response(
                            job.job_request.webhook_url,
                            True,
                            200,
                            "Job Progress",
                            job.dict(),
                        )
                        if logs_count % 20 == 0:
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
        raise Exception(str(e))

    except Exception as e:
        print(e)
        job.job_status = JobStatus.FAILED.value
        job.error_message = str(e)
        raise Exception(str(e))


def process_request(job: Job):
    job.job_status = JobStatus.PROCESSING.value
    webhook_response(job.job_request.webhook_url, True, 200, "Job Started", job.dict())
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
        job.job_request.dataset_folder,job.job_request.lora_name
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
                total_epochs=job.job_epochs,
                current_epoch_number=len(job.job_results) + 1,
            )
            epoch_model_s3_path = f"{job.job_s3_folder}{new_file}"
            saved_checkout_path = os.path.join(safetensors_files_path, new_file)
            print("Going to upload model in S3 at ", epoch_response)
            print("Local Path of uploaded model is ", saved_checkout_path)
            epoch_response.epoch_model_s3_path = epoch_model_s3_path
            Thread(
                target=upload_media_to_s3,
                args=(saved_checkout_path, epoch_model_s3_path),
            ).start()
            webhook_response(
                job.job_request.webhook_url, True, 200, "Epoch Completed", job.dict()
            )
            job.job_results.append(epoch_response)
        safetensors_files.update(new_files)
