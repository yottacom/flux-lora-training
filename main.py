import os
import sys
import time
import json
import torch
import argparse
import signal
import runpod
import threading
from google.oauth2 import service_account
from google.cloud import pubsub_v1
from server.request_queue import TrainingRequest, Job, JobStatus
from server.utils import (
    save_images_and_generate_metadata,
    generate_config_file,
    webhook_response,
    save_gcloud_keys,
)
from server.request_processor import process_request
from server import server_settings
from server.gcloud_utils import upload
from flux_inference import generate

runpod.api_key = server_settings.RUNPOD_API_KEY
save_gcloud_keys(
    "GCLOUD_STORAGE_CREDENTIALS", server_settings.GCLOUD_STORAGE_CREDENTIALS
)
save_gcloud_keys("GCLOUD_PUBSUB_CREDENTIALS", server_settings.GCLOUD_PUBSUB_CREDENTIALS)

if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)


def train(job: Job):
    try:
        saviour_thread = threading.Thread(target=saviour, args=(job,))
        saviour_thread.start()
        process_request(job)
        webhook_response(
            job.job_request.training_webhook_url, True, 200, "Job Completed", job.dict()
        )
        return job
    except Exception as ex:
        print(ex)
        webhook_response(
            training_webhook_url,
            False,
            500,
            str(ex),
            None if job is None else job.dict(),
        )
        raise Exception(ex)


credentials = service_account.Credentials.from_service_account_file(
    server_settings.GCLOUD_PUBSUB_CREDENTIALS
)

subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
subscription_path = subscriber.subscription_path(
    server_settings.GCLOUD_PROJECT_ID, server_settings.GCLOUD_PUB_SUB_SUBSCRIPTION
)

torch.cuda.empty_cache()

# Global stop events for graceful thread control
stop_event = threading.Event()  # Application-wide shutdown
ack_extension_stop_event = threading.Event()  # For acknowledgment thread control

# Thread-safe lock for global variables
lock = threading.Lock()

last_message_acknowledge_time = time.time()
is_last_message_acknowledged = True


def check_idle_timeout():
    global last_message_acknowledge_time
    global is_last_message_acknowledged
    idle_timeout = (
        server_settings.IDLE_TIME_IN_SECONDS or 60
    )  # Configurable idle timeout in seconds
    while not stop_event.is_set():
        try:
            with lock:
                if is_last_message_acknowledged:
                    current_time = time.time()
                    time_difference = None
                    if last_message_acknowledge_time:
                        time_difference = current_time - last_message_acknowledge_time
                    if time_difference and time_difference > idle_timeout:
                        print("Idle timeout reached. Stopping pod...")
                        runpod.terminate_pod(server_settings.RUNPOD_POD_ID)
                        break
                    else:
                        print(
                            f"Idle timeout not reached. Time difference: {time_difference}"
                        )
                else:
                    print("Message is being processed. Resetting idle timeout...")
        except Exception as e:
            print(f"Error during idle timeout check: {e}")
        time.sleep(5)


def extend_ack_deadline(message, stop_event, interval=30):
    while not stop_event.is_set():
        try:
            print(
                f"Extending acknowledgment deadline for message: {message.message_id}"
            )
            message.modify_ack_deadline(60)  # Extend the deadline by 60 seconds
            time.sleep(interval)  # Modify the deadline every 30 seconds
        except Exception as e:
            print(f"Error extending acknowledgment deadline: {e}")
            break


def acknowledge_message(message):
    global last_message_acknowledge_time
    global is_last_message_acknowledged
    with lock:
        last_message_acknowledge_time = time.time()
        is_last_message_acknowledged = True
    message.ack()
    print("Message acknowledged successfully!")


def saviour(job: Job):
    wait_time_for_saviour_interruption = 720
    start_time = time.time()
    while True:
        elapsed_time = (time.time()) - start_time
        if elapsed_time and elapsed_time < wait_time_for_saviour_interruption:
            print("Saviour Will not interrupt, Elapsed Time is : ", elapsed_time)
            time.sleep(10)
            if job.job_progress > 0:
                break
            continue

        if elapsed_time > wait_time_for_saviour_interruption and job.job_progress <= 0:
            job.job_status = JobStatus.FAILED.value
            job.error_message = "Try Again, It seems there was an issue with training!"
            webhook_response(
                job.job_request.training_webhook_url,
                False,
                500,
                job.error_message,
                job.dict(),
            )
            runpod.terminate_pod(server_settings.RUNPOD_POD_ID)

def process_request_payload(training_request_dict):
    training_request_defaults = TrainingRequest()
    job_id = training_request_dict.get("job_id")
    lora_name = training_request_dict.get("lora_name")
    quantize_model = training_request_dict.get("quantize_model", True)
    example_image_width = training_request_dict.get("example_image_width", None)
    example_image_height = training_request_dict.get("example_image_height", None)
    example_prompts = []
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
        webhook_response(
            training_webhook_url, False, 400, "No training webhook url found!"
        )
        return None
    if not inference_webhook_url:
        print("No inference webhook url found!")
        webhook_response(
            inference_webhook_url, False, 400, "No inference webhook url found!"
        )
        return None

    images_urls = training_request_dict.get("images_urls", [])

    if not job_id:
        webhook_response(
            training_webhook_url, False, 400, "No job id provided!"
        )
        return None
    if not lora_name:
        webhook_response(
            training_webhook_url, False, 400, "No lora name provided!"
        )
        return None
    if len(images_urls) == 0:
        webhook_response(
            training_webhook_url, False, 400, "No image urls provided!"
        )
        return None

    dataset_path = save_images_and_generate_metadata(job_id, images_urls, lora_name)

    training_request = TrainingRequest()
    # replace whitespace in lora name with _
    lora_name = lora_name.replace(" ", "_")
    training_request.lora_name = lora_name
    training_request.images_urls = images_urls
    training_request.dataset_folder = dataset_path
    training_request.training_webhook_url = training_webhook_url
    training_request.inference_webhook_url = inference_webhook_url
    training_request.example_prompts = example_prompts
    config_file_path = generate_config_file(training_request)
    training_request.config_file = config_file_path
    training_request.quantize_model = quantize_model
    if example_image_width is not None and example_image_height is not None:
        training_request.example_image_width=example_image_width
        training_request.example_image_height=example_image_height

    print("Config File generated successfully!", config_file_path)
    job = Job(job_id=job_id, job_request=training_request, job_epochs=10)

    return job

def callback(message):
    try:
        global is_last_message_acknowledged
        with lock:
            is_last_message_acknowledged = False
        print("message_id => ", message.message_id)
        message_data = message.data.decode("utf-8")
        parsed_message = json.loads(message_data)
        request_payload = None
        if "Field" in parsed_message:
            request_payload = json.loads(parsed_message["Field"])

        if not request_payload:
            print("No request payload found!")
            acknowledge_message(message)
            return
        print(f"Received message: {request_payload}")
        # Start a separate thread to keep extending the acknowledgment deadline during processing
        ack_extension_stop_event.clear()  # Ensure event is cleared before starting
        ack_extension_thread = threading.Thread(
            target=extend_ack_deadline, args=(message, ack_extension_stop_event)
        )
        ack_extension_thread.start()
        training_job = process_request_payload(request_payload)

        process_example_prompts = request_payload.get("process_example_prompts", True)
        perform_training_job = request_payload.get("perform_training_job", True)
        pretrained_lora_url = request_payload.get("pretrained_lora_url", None)
        if not perform_training_job and not pretrained_lora_url:
            webhook_response(
            training_job.job_request.training_webhook_url,
            False,
            400,
            "Pretrained LoRA URL is required when perform_training_job is False.",
            training_job.dict(),
        )

        if perform_training_job:
            training_job: Job = train(training_job)
        if process_example_prompts:
            training_job.job_request.example_prompts = training_request_dict.get(
                "example_prompts", training_request_defaults.example_prompts
            )
            inference_results = generate(training_job,pretrained_lora_url if not perform_training_job else None)
        training_job.job_logs_gcloud_path = upload(
            path=training_job.job_logs_gcloud_path,
            bucket_path="logs/",
            file_name=f"{training_job.job_id}.txt",
        )
        webhook_response(
            training_job.job_request.training_webhook_url,
            False,
            500,
            str(e),
            training_job.dict(),
        )
        acknowledge_message(message)
        print("Message acknowledged successfully!")

        # Stop the acknowledgment extension thread
        ack_extension_stop_event.set()
        ack_extension_thread.join()
        print("Exited Callback!")
    except Exception as e:
        print(f"Error processing message: {e}")
        acknowledge_message(message)


def listen_for_messages():
    # Flow control settings: only allow 1 message at a time
    flow_control = pubsub_v1.types.FlowControl(
        max_messages=1,  # Limit the number of messages being pulled concurrently
        max_bytes=10 * 1024 * 1024,  # Optionally limit the total message size
    )

    # Subscribe and listen for messages
    streaming_pull_future = subscriber.subscribe(
        subscription_path,
        callback=callback,
        flow_control=flow_control,
    )
    print(f"Listening for messages on {subscription_path}...")

    try:
        streaming_pull_future.result()  # Keeps the listener active
    except KeyboardInterrupt:
        print("Interrupt received, cancelling the subscription...")
        streaming_pull_future.cancel()
        stop_event.set()  # Set the global stop event to stop any ongoing threads


def handle_termination_signal(signum, frame):
    print("Received termination signal. Stopping listener...")
    stop_event.set()
    sys.exit(0)


# Register the signal handlers to gracefully stop the application
signal.signal(signal.SIGTERM, handle_termination_signal)
signal.signal(signal.SIGINT, handle_termination_signal)

idle_time_checker_thread = threading.Thread(target=check_idle_timeout)
idle_time_checker_thread.start()

listen_for_messages()

while True:
    time.sleep(5)
