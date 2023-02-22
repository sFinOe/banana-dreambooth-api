
def init():
    import os
    import subprocess
    from natsort import natsorted
    from glob import glob
    import zipfile

    global HF_AUTH_TOKEN
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")


def training(model_inputs: dict) -> dict:

    import os
    import subprocess
    from natsort import natsorted
    from glob import glob
    import zipfile

    ID = model_inputs.get("id", None)
    MODEL_NAME = model_inputs.get(
        "model_name", "runwayml/stable-diffusion-v1-5")
    VAE_NAME = model_inputs.get("vae_name", "stabilityai/sd-vae-ft-mse")
    LR_WARMUP_STEPS = model_inputs.get("lr_warmup_steps", "100")
    MAX_TRAIN_STEPS = model_inputs.get("max_train_steps", "1000")
    SAVE_SAMPLE_PROMPT = model_inputs.get(
        "save_sample_prompt", f"photo of {ID}")
    REVISION = model_inputs.get("revision", "main")
    NUM_CLASS_IMAGES = model_inputs.get("num_class_images", "100")
    SAVE_MODEL = model_inputs.get("save_model", None)
    DATASET_PATH = model_inputs.get("dataset_path", None)
    CLASS_TYPE = model_inputs.get("class_type", "person")
    OUTPUT_DIR = "output"

    # s3 bucket config

    ACCESS_ID = model_inputs.get("accessId", None)
    SECERT_KEY = model_inputs.get("accessSecret", None)
    ENDPOINT_URL = model_inputs.get("endpointUrl", None)

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
            "class_data_dir":       "class_images/person_ddim"
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
        [f"rclone", "copy", f"cloudflare_r2:{DATASET_PATH}", "data/images"])

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
    print(f"[*] SAVE_MODEL={SAVE_MODEL}")

    subprocess.call(["rclone", "copy", WEIGHTS_DIR,
                     f"cloudflare_r2:{SAVE_MODEL}"])

    return {"status": "success"}
