import os
import subprocess
import json
import argparse
import zipfile
import urllib.request
from natsort import natsorted
from glob import glob
import shutil
import time


def inference(model_inputs: dict) -> dict:
    MODEL_NAME = "SG161222/Realistic_Vision_V1.3"
    VAE_NAME = "stabilityai/sd-vae-ft-mse"
    LR_WARMUP_STEPS = "144"
    MAX_TRAIN_STEPS = "1440"
    SAVE_SAMPLE_PROMPT = "photo of 1676642713542"
    REVISION = "main"
    NUM_CLASS_IMAGES = "216"
    SAVE_MODEL = "models/1676642713542"
    OUTPUT_DIR = "/content/output"

    # s3 bucket config

    ACCESS_ID = "4e57071dba2f0fb9f5a2af8037e95e82"
    SECERT_KEY = "0bdd57295ce8d381cb6a9ab486a0f02abbd9ab9eff8a7f521bea7cd03de8189c"
    ENDPOINT_URL = "https://c5496cc41ca6d42c8358101ad551f1b4.r2.cloudflarestorage.com"

    # setup s3 bucket config

    config_dir = os.path.expanduser("~/.config/rclone")
    os.makedirs(config_dir, exist_ok=True)

    file_path = os.path.join(config_dir, "rclone.conf")
    with open(file_path, "w") as file:
        # Write any content you need to the file
        file.write(
            f"[cloudflare_r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = {ACCESS_ID}\nsecret_access_key = {SECERT_KEY}\nregion = auto\nendpoint = {ENDPOINT_URL}\n\n")

    folder_name = os.path.expanduser("~/.huggingface")
    try:
        os.makedirs(folder_name, exist_ok=True)
    except FileExistsError:
        pass
    token = "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"
    token_file = os.path.join(folder_name, "token")
    with open(token_file, "w") as f:
        f.write(token)

    concepts_list = [
        {
            "instance_prompt":      "1676642713542",
            "class_prompt":         "photo of a woman",
            "instance_data_dir":    "/content/data/1676642713542",
            "class_data_dir":       "/content/data/woman"
        },
    ]

    # `class_data_dir` contains regularization images
    import json
    import os
    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    subprocess.run(["accelerate", "launch", "train_dreambooth.py",
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
                    f"--save_sample_prompt={SAVE_SAMPLE_PROMPT}",
                    "--concepts_list=MyConfig.json"])

    WEIGHTS_DIR = natsorted(glob(OUTPUT_DIR + os.sep + "*"))[-1]
    print(f"[*] WEIGHTS_DIR={WEIGHTS_DIR}")

    subprocess.call(["rclone", "copy", WEIGHTS_DIR,
                     f"cloudflare_r2:/{SAVE_MODEL}"])
