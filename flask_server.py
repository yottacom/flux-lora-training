from flask import Flask, request, jsonify
from test_response import webhook_response
from server.request_queue import JobStatus
import uuid
from typing import List # type: ignore
import threading
import time


app = Flask(__name__)


@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.get_json()

        lora_name = data.get('lora_name', None)
        example_prompts = data.get('example_prompts', [])
        training_webhook_url = data.get('training_webhook_url', None)
        image_urls = data.get('image_urls', []) 
        is_test = data.get('is_test', True)
        job_id = data.get('job_id', str(uuid.uuid4()))

        if not lora_name:
            return jsonify({"error": "No lora name provided"}), 400

        if not image_urls:
            return jsonify({"error": "No image urls provided"}), 400
        
        if not is_test:
            from main import train
            dict_to_pass = {
                "job_id" : job_id,
                "lora_name" : lora_name,
                "training_webhook_url" : training_webhook_url if training_webhook_url else None,
                "images_urls" : image_urls,
                "example_prompts" : example_prompts if example_prompts else None
            }
            threading.Thread(target=train, args=(dict_to_pass,)).start()
            return jsonify({"status": "success", "job_id": job_id, "status_code" : 200, "message" : "Job Started"}), 200

        threading.Thread(target=send_test_response, args=(job_id,lora_name,image_urls,training_webhook_url,example_prompts,)).start()

        return jsonify({"status": "success", "job_id": job_id, "status_code" : 200, "message" : "Job Started See Webhook For results"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def send_test_response(job_id: str,lora_name,image_urls,training_webhook_url,example_prompts):
    from server.request_queue import TrainingRequest,TrainingResponse,Job
    training_request = TrainingRequest()
    training_request.lora_name = lora_name
    training_request.images_urls = image_urls
    training_request.dataset_folder = ""
    training_request.training_webhook_url = training_webhook_url
    training_request.example_prompts = example_prompts
    training_request.config_file = ""
    job = Job(job_id=job_id, job_request=training_request, job_epochs=10)
    job.job_results.append(TrainingResponse(total_epochs=10))
    lora_url = "https://boothybooth.s3.eu-north-1.amazonaws.com/CustomFiles/phil.safetensors"

    for i in range(10):
        time.sleep(2)
        job.job_progress = i*10
        job.job_results.append(TrainingResponse(current_epoch_number=i+1,total_epochs=10))
        webhook_response(training_webhook_url, True, 200, "Job Started", job.dict())
    job.job_progress=100
    job.job_status = JobStatus.FINISHED.value
    job.job_results.append(TrainingResponse(epoch_model_s3_url=lora_url,current_epoch_number=10,total_epochs=10))
    webhook_response(training_webhook_url, True, 200, "Job Started", job.dict())
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
