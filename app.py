def training(model_inputs: dict) -> dict:

    import os
    import subprocess
    from natsort import natsorted
    from glob import glob
    import zipfile

    ID = model_inputs["id"]
    MODEL_NAME = model_inputs["model_name"]
    VAE_NAME = model_inputs["vae_name"]
    LR_WARMUP_STEPS = model_inputs["lr_warmup_steps"]
    MAX_TRAIN_STEPS = model_inputs["max_train_steps"]
    SAVE_SAMPLE_PROMPT = model_inputs["save_sample_prompt"]
    REVISION = model_inputs["revision"]
    NUM_CLASS_IMAGES = model_inputs["num_class_images"]
    SAVE_MODEL = model_inputs["save_model"]
    DATASET_PATH = model_inputs["dataset_path"]
    CLASS_TYPE = model_inputs["class_type"]
    OUTPUT_DIR = "output"

    # s3 bucket config

    ACCESS_ID = model_inputs["access_id"]
    SECERT_KEY = model_inputs["SECERT_KEY"]
    ENDPOINT_URL = model_inputs["ENDPOINT_URL"]

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
            "instance_prompt":      ID,
            "class_prompt":         f"photo of a {CLASS_TYPE}",
            "instance_data_dir":    "data/images",
            "class_data_dir":       "content/data/woman"
        },
    ]

    # `class_data_dir` contains regularization images
    import json
    import os
    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    # -----------------Download dataset-----------------#

    subprocess.call(
        [f"rclone", "copy", f"cloudflare_r2:/{DATASET_PATH}", "data/images"])

    with zipfile.ZipFile(f"data/images/{ID}.zip", "r") as zip_ref:
        zip_ref.extractall("data/images")
    os.remove(f"data/images/{ID}.zip")

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

    return {"status": "success"}
