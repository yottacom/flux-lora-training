from PIL import Image
import torch
import uuid
import os
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def run_captioning(images):
    # Load internally to not consume resources for training
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

    captions = []
    for i, image_path in enumerate(images):
        print(captions[i])
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")

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
        captions[i] = caption_text

        yield captions
    model.to("cpu")
    del model
    del processor


def start_training(
    lora_name,
    concept_sentence,
    steps,
    lr,
    rank,
    model_to_train,
    low_vram,
    dataset_folder,
    sample_1,
    sample_2,
    sample_3,
    use_more_advanced_options,
    more_advanced_options,
):
    push_to_hub = True
    if not lora_name:
        raise Exception("You forgot to insert your LoRA name! This name has to be unique.")
            
    print("Started training")
    slugged_lora_name = slugify(lora_name)

    # Load the default config
    with open("config/examples/train_lora_flux_24gb.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update the config with user inputs
    config["config"]["name"] = slugged_lora_name
    config["config"]["process"][0]["model"]["low_vram"] = low_vram
    config["config"]["process"][0]["train"]["skip_first_sample"] = True
    config["config"]["process"][0]["train"]["steps"] = int(steps)
    config["config"]["process"][0]["train"]["lr"] = float(lr)
    config["config"]["process"][0]["network"]["linear"] = int(rank)
    config["config"]["process"][0]["network"]["linear_alpha"] = int(rank)
    config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_folder
    config["config"]["process"][0]["save"]["push_to_hub"] = push_to_hub
    if concept_sentence:
        config["config"]["process"][0]["trigger_word"] = concept_sentence
    
    if sample_1 or sample_2 or sample_3:
        config["config"]["process"][0]["train"]["disable_sampling"] = False
        config["config"]["process"][0]["sample"]["sample_every"] = steps
        config["config"]["process"][0]["sample"]["sample_steps"] = 28
        config["config"]["process"][0]["sample"]["prompts"] = []
        if sample_1:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_1)
        if sample_2:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_2)
        if sample_3:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_3)
    else:
        config["config"]["process"][0]["train"]["disable_sampling"] = True
    if(model_to_train == "schnell"):
        config["config"]["process"][0]["model"]["name_or_path"] = "black-forest-labs/FLUX.1-schnell"
        config["config"]["process"][0]["model"]["assistant_lora_path"] = "ostris/FLUX.1-schnell-training-adapter"
        config["config"]["process"][0]["sample"]["sample_steps"] = 4
    if(use_more_advanced_options):
        more_advanced_options_dict = yaml.safe_load(more_advanced_options)
        config["config"]["process"][0] = recursive_update(config["config"]["process"][0], more_advanced_options_dict)
        print(config)
    
    # Save the updated config
    # generate a random name for the config
    random_config_name = str(uuid.uuid4())
    os.makedirs("tmp", exist_ok=True)
    config_path = f"tmp/{random_config_name}-{slugged_lora_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # run the job locally
    job = get_job(config_path)
    job.run()
    job.cleanup()

    return f"Training completed successfully. Model saved as {slugged_lora_name}"


