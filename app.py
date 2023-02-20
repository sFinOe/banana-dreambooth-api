import os
import json
import subprocess
from natsort import natsorted
from glob import glob
import time


def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    # model = StableDiffusionPipeline.from_pretrained("dreambooth_weights/",use_auth_token=HF_AUTH_TOKEN).to("cuda")


def training(model_inputs: dict) -> dict:
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    folder_name = os.path.expanduser("~/.huggingface")
    try:
        os.makedirs(folder_name, exist_ok=True)
    except FileExistsError:
        pass
    token = HF_AUTH_TOKEN
    token_file = os.path.join(folder_name, "token")
    with open(token_file, "w") as f:
        f.write(token)

    config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(config_dir, exist_ok=True)

    file_path = os.path.join(config_dir, "rclone.conf")
    with open(file_path, "w") as file:
        # Write any content you need to the file
        file.write(
            f"[cloudflare_r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = 4e57071dba2f0fb9f5a2af8037e95e82\nsecret_access_key = 0bdd57295ce8d381cb6a9ab486a0f02abbd9ab9eff8a7f521bea7cd03de8189c\nregion = auto\nendpoint = https://c5496cc41ca6d42c8358101ad551f1b4.r2.cloudflarestorage.com\n\n")

    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    OUTPUT_DIR = "stable_diffusion_weights/output"
    VAE_NAME = "stabilityai/sd-vae-ft-mse"
    REVISION = "fp16"
    LR_WARMUP_STEPS = "144"
    NUM_CLASS_IMAGES = "216"
    MAX_TRAIN_STEPS = "1440"
    SAVE_SAMPLE_PROMPT = "photo of 1676642713542"
    SAVE_MODEL = "models/1676642713542"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    concepts_list = [
        {
            "instance_prompt":      "1676642713542",
            "class_prompt":         "photo of a woman",
            "instance_data_dir":    "data/1676642713542",
            "class_data_dir":       "data/woman"
        },
    ]

    # `class_data_dir` contains regularization images

    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    subprocess.call(["accelerate", "launch", "train_dreambooth.py",
                     f"--pretrained_model_name_or_path={MODEL_NAME}",
                     f"--pretrained_vae_name_or_path={VAE_NAME}",
                     f"--output_dir={OUTPUT_DIR}",
                     f"--revision={REVISION}",
                     "--with_prior_preservation", "--prior_loss_weight=1.0",
                     "--seed=1337",
                     "--resolution=512",
                     "--train_batch_size=1",
                     "--train_text_encoder",
                     "--mixed_precision=fp16",
                     "--use_8bit_adam",
                     "--gradient_accumulation_steps=1",
                     "--learning_rate=1e-6",
                     "--lr_scheduler=constant",
                     f"--lr_warmup_steps={LR_WARMUP_STEPS}",
                     f"--num_class_images={NUM_CLASS_IMAGES}",
                     "--sample_batch_size=4",
                     f"--max_train_steps={MAX_TRAIN_STEPS}",
                     "--save_interval=10000",
                     f"--save_sample_prompt='{SAVE_SAMPLE_PROMPT}'",
                     "--concepts_list=concepts_list.json"])

    WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
    print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")

    subprocess.call(["rclone", "copy", WEIGHTS_DIR,
                     f"cloudflare_r2:/{SAVE_MODEL}"])

    return {"model": SAVE_MODEL}
