import os
from threading import Thread
import uuid
import json
import yaml
import requests
import torch
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import server.server_settings as server_settings
from server.request_queue import TrainingRequest, ModelTypes


def save_images_and_generate_metadata(job_id, image_urls, lora_name):
    dataset_path = os.path.join(server_settings.DATASET_DIR, lora_name, job_id)
    os.makedirs(dataset_path, exist_ok=True)

    metadata_file_path = os.path.join(dataset_path, "metadata.jsonl")

    # Prepare the model and processor for captioning
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True
    )

    with open(metadata_file_path, "w") as metadata_file:
        # Download and process each image
        for idx, url in enumerate(image_urls):
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                image = Image.open(BytesIO(response.content))
                image = image.convert("RGB")
                image_path = os.path.join(dataset_path, f"{idx}.jpg")
                image.save(image_path, format="JPEG")
                print(f"Downloaded and saved image {idx} from {url} to {image_path}")

                # Generate caption
                prompt = "<DETAILED_CAPTION>"
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(
                    device, torch_dtype
                )

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                )

                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                parsed_answer = processor.post_process_generation(
                    generated_text, task=prompt, image_size=(image.width, image.height)
                )
                caption_text = parsed_answer["<DETAILED_CAPTION>"].replace(
                    "The image shows ", ""
                )

                # Add lora_name to the caption and save metadata
                caption_text = f"{lora_name}, {caption_text}"
                metadata = {"file_name": f"{idx}.jpg", "prompt": caption_text}

                metadata_file.write(json.dumps(metadata) + "\n")
                print(f"Image file Downloaded and Saved : ", image_path)

            except requests.RequestException as e:
                print(f"Failed to download {url}: {e}")
            except IOError as e:
                print(f"Failed to process image from {url}: {e}")

    # Clean up the model from memory
    model.to("cpu")
    del model
    del processor
    torch.cuda.empty_cache()  # Explicitly clear GPU memory
    print(
        f"Downloaded and saved {len(image_urls)} images with metadata to {metadata_file_path}"
    )
    return dataset_path


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def generate_config_file(training_request: TrainingRequest):
    push_to_hub = False
    # Load the default config
    with open(
        os.path.join(
            server_settings.BASE_DIR, "config/examples/train_lora_flux_24gb.yaml"
        ),
        "r",
    ) as f:
        config = yaml.safe_load(f)

    # Update the config with user inputs
    config["config"]["name"] = training_request.lora_name
    config["config"]["process"][0]["model"]["low_vram"] = training_request.low_vram
    config["config"]["process"][0]["train"]["skip_first_sample"] = True
    config["config"]["process"][0]["train"]["steps"] = int(training_request.steps)
    config["config"]["process"][0]["train"]["lr"] = float(
        training_request.learning_rate
    )
    config["config"]["process"][0]["network"]["linear"] = int(training_request.rank)
    config["config"]["process"][0]["network"]["linear_alpha"] = int(
        training_request.rank
    )
    config["config"]["process"][0]["datasets"][0][
        "folder_path"
    ] = training_request.dataset_folder
    config["config"]["process"][0]["training_folder"] = training_request.dataset_folder
    config["config"]["process"][0]["trigger_word"] = training_request.lora_name
    config["config"]["process"][0]["save"]["push_to_hub"] = push_to_hub
    config["config"]["process"][0]["save"]["save_every"] = training_request.save_step
    config["config"]["process"][0]["trigger_word"] = training_request.lora_name

    if len(training_request.example_prompts) > 0:
        config["config"]["process"][0]["train"]["disable_sampling"] = False
        config["config"]["process"][0]["sample"][
            "sample_every"
        ] = training_request.save_step
        config["config"]["process"][0]["sample"]["sample_steps"] = 50
        config["config"]["process"][0]["sample"]["prompts"] = []
        for prompt in training_request.example_prompts:
            config["config"]["process"][0]["sample"]["prompts"].append(prompt)

    else:
        config["config"]["process"][0]["train"]["disable_sampling"] = True

    if training_request.model == ModelTypes.SCHNELL.value:
        config["config"]["process"][0]["model"][
            "name_or_path"
        ] = "black-forest-labs/FLUX.1-schnell"
        config["config"]["process"][0]["model"][
            "assistant_lora_path"
        ] = "ostris/FLUX.1-schnell-training-adapter"
        config["config"]["process"][0]["sample"]["sample_steps"] = 10

    # Save the updated config
    # generate a random name for the config
    random_config_name = str(uuid.uuid4())
    os.makedirs("tmp", exist_ok=True)
    config_path = os.path.join(
        training_request.dataset_folder,
        f"{random_config_name}-{training_request.lora_name}.yaml",
    )
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print("Config File Generated & Saved at :", config_path)
    return config_path


def webhook_response(webhook_url, status, code, message, data=None):
    def send(webhook_url, status, code, message, data=None):
        response_data = {
            "status": status,
            "code": code,
            "message": message,
            "data": data,
        }
        if webhook_url and "http" in webhook_url:
            requests.post(webhook_url, json=response_data)

    Thread(target=send, args=(webhook_url, status, code, message, data)).start()
    return None
