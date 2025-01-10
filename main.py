import sys
import time
import json
import torch
import argparse
import signal
import threading
from google.oauth2 import service_account
from google.cloud import pubsub_v1
from server.request_queue import TrainingRequest, Job
from server.utils import (
    save_images_and_generate_metadata,
    generate_config_file,
    webhook_response,
    save_gcloud_keys,
)
from server.request_processor import process_request
from server import server_settings
from flux_inference import generate

save_gcloud_keys(
    "GCLOUD_STORAGE_CREDENTIALS", server_settings.GCLOUD_STORAGE_CREDENTIALS
)
save_gcloud_keys("GCLOUD_PUBSUB_CREDENTIALS", server_settings.GCLOUD_PUBSUB_CREDENTIALS)


def train(training_request_dict: dict):
    job = None
    try:
        training_request_defaults = TrainingRequest()
        job_id = training_request_dict.get("job_id")
        lora_name = training_request_dict.get("lora_name")
        quantize_model = training_request_dict.get("quantize_model",True)
        example_prompts = []
        training_webhook_url = str(
            training_request_dict.get(
                "training_webhook_url", training_request_defaults.training_webhook_url
            )
        )
        if not training_webhook_url:
            print("No training webhook url found!")
            return webhook_response(
                training_webhook_url, False, 400, "No training webhook url found!"
            )

        images_urls = training_request_dict.get("images_urls", [])

        if not job_id:
            return webhook_response(
                training_webhook_url, False, 400, "No job id provided!"
            )
        if not lora_name:
            return webhook_response(
                training_webhook_url, False, 400, "No lora name provided!"
            )
        if len(images_urls) == 0:
            return webhook_response(
                training_webhook_url, False, 400, "No image urls provided!"
            )

        dataset_path = save_images_and_generate_metadata(job_id, images_urls, lora_name)

        training_request = TrainingRequest()
        training_request.lora_name = lora_name
        training_request.images_urls = images_urls
        training_request.dataset_folder = dataset_path
        training_request.training_webhook_url = training_webhook_url
        training_request.example_prompts = example_prompts
        config_file_path = generate_config_file(training_request)
        training_request.config_file = config_file_path
        training_request.quantize_model=quantize_model

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
                        import runpod

                        runpod.api_key = server_settings.RUNPOD_API_KEY
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

        training_job=train(request_payload)
        inference_results = generate(training_job)

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