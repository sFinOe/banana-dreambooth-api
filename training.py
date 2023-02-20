import time
import wget
import os
from os import listdir
from os.path import isfile
import time
import shutil
import subprocess
import random


def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

# --------------------------------------------------------------------------#


def training(model_inputs: dict) -> dict:
    Model_Version = "1.5"
    Custom_Model_Version = "1.5"
    Path_to_HuggingFace = "SfinOe/stable-diffusion-v1.5"
    CKPT_Path = ""
    CKPT_Link = ""

    if os.path.exists('/gdrive/MyDrive/Fast-Dreambooth/token.txt'):
        with open("/gdrive/MyDrive/Fast-Dreambooth/token.txt") as f:
            token = f.read()
        auth = f"https://USER:{token}@"
    else:
        auth = "https://"

    if Path_to_HuggingFace:
        if os.path.exists('/stable-diffusion-custom'):
            os.system(f"rm -r /stable-diffusion-custom")
        os.chdir('/')
        os.makedirs('/stable-diffusion-custom/scheduler')
        os.mkdir
        os.chdir('/stable-diffusion-custom')
        subprocess.call(['git', 'init'])
        subprocess.call(['git', 'lfs', 'install', '--system', '--skip-repo'])
        os.system(
            f"git remote add -f origin  '{auth}huggingface.co/{Path_to_HuggingFace}'")
        os.system('git config core.sparsecheckout true')
        subprocess.run(['echo', '-e', 'scheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json\n!*.safetensors'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        os.system('git pull origin main')
        while not os.path.exists('/stable-diffusion-custom/unet/diffusion_pytorch_model.bin'):
            print('[1;31mCheck the link you provided')
            time.sleep(5)
        if os.path.exists('/stable-diffusion-custom/unet/diffusion_pytorch_model.bin'):
            os.system("rm -r /stable-diffusion-custom/.git")
            os.system("rm model_index.json")
            time.sleep(1)
            wget.download(
                'https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dreambooth/model_index.json')
            os.chdir('/')
            MODEL_NAME = "/stable-diffusion-custom"
        else:
            while not os.path.exists('/stable-diffusion-custom/unet/diffusion_pytorch_model.bin'):
                time.sleep(5)

    # --------------------------------------------------------------------------#

    try:
        MODEL_NAME
        pass
    except:
        MODEL_NAME = ""

    PT = "1676642713542"

    Session_Name = "1676642713542"
    while Session_Name == "":
        Session_Name = input('\033[1;31mInput the Session Name:\033[0m')
    Session_Name = Session_Name.replace(" ", "_")

    WORKSPACE = '/gdrive/MyDrive/Fast-Dreambooth'

    INSTANCE_NAME = Session_Name
    OUTPUT_DIR = "/models/" + Session_Name
    SESSION_DIR = WORKSPACE + '/Sessions/' + Session_Name
    INSTANCE_DIR = SESSION_DIR + '/instance_images'
    CONCEPT_DIR = SESSION_DIR + '/concept_images'
    CAPTIONS_DIR = SESSION_DIR + '/captions'
    MDLPTH = str(SESSION_DIR + "/" + Session_Name + '.ckpt')

    Model_Version = "1.5"

    os.makedirs(SESSION_DIR, exist_ok=True)
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    os.makedirs(CONCEPT_DIR, exist_ok=True)
    os.makedirs(CAPTIONS_DIR, exist_ok=True)

    # Change working directory
    os.chdir('')

    # Set resume flag to False
    resume = False

    Remove_existing_instance_images = False

    IMAGES_FOLDER_OPTIONAL = "/dataset"

    Smart_Crop_images = False
    Crop_size = 512

    if IMAGES_FOLDER_OPTIONAL != "":
        for file in os.listdir(IMAGES_FOLDER_OPTIONAL):
            if file.endswith(".txt"):
                shutil.move(os.path.join(
                    IMAGES_FOLDER_OPTIONAL, file), CAPTIONS_DIR)
            else:
                shutil.copy2(os.path.join(
                    IMAGES_FOLDER_OPTIONAL, file), INSTANCE_DIR)

        print('\n\033[1;32mDone, proceed to the next cell')

    # --------------------------------------------------------------------------#

    MODELT_NAME = MODEL_NAME

    UNet_Training_Steps = 1170
    UNet_Learning_Rate = 5e-6
    untlr = UNet_Learning_Rate

    Text_Encoder_Training_Steps = 450

    Text_Encoder_Concept_Training_Steps = 1500

    Text_Encoder_Learning_Rate = 1e-6
    txlr = Text_Encoder_Learning_Rate

    trnonltxt = ""
    if UNet_Training_Steps == 0:
        trnonltxt = "--train_only_text_encoder"

    Seed = ""

    External_Captions = False
    extrnlcptn = ""
    if External_Captions:
        extrnlcptn = "--external_captions"

    Style_Training = False
    Style = ""
    if Style_Training:
        Style = "--Style"

    Resolution = "512"
    Res = int(Resolution)

    fp16 = True

    if Seed == "" or Seed == "0":
        Seed = random.randint(1, 999999)
    else:
        Seed = int(Seed)

    GC = "--gradient_checkpointing"

    if fp16:
        prec = "fp16"
    else:
        prec = "no"

    GCUNET = GC
    if Res <= 640:
        GCUNET = ""

    precision = prec

    stpsv = 500
    stp = 0

    Enable_text_encoder_training = True
    Enable_Text_Encoder_Concept_Training = True

    if Text_Encoder_Training_Steps == 0:
        Enable_text_encoder_training = False
    else:
        stptxt = Text_Encoder_Training_Steps

    if Text_Encoder_Concept_Training_Steps == 0:
        Enable_Text_Encoder_Concept_Training = False
    else:
        stptxtc = Text_Encoder_Concept_Training_Steps

    def dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, precision, Training_Steps):
        os.system(f"accelerate launch /diffusers/examples/dreambooth/train_dreambooth.py {trnonltxt} {extrnlcptn} --image_captions_filename --train_text_encoder --dump_only_text_encoder --pretrained_model_name_or_path='{MODELT_NAME}' --instance_data_dir='{INSTANCE_DIR}' --output_dir='{OUTPUT_DIR}' --instance_prompt='{PT}' --seed={Seed} --resolution=512 --mixed_precision={precision} --train_batch_size=1 --gradient_accumulation_steps=1 {GC} --use_8bit_adam --learning_rate={txlr} --lr_scheduler='linear' --lr_warmup_steps=0 --max_train_steps={Training_Steps}")

    def train_only_unet(stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR, OUTPUT_DIR, PT, Seed, Res, precision, Training_Steps):
        os.system(f"accelerate launch /diffusers/examples/dreambooth/train_dreambooth.py {Style} {extrnlcptn} --image_captions_filename --train_only_unet --save_starting_step={stpsv} --save_n_steps={stp} --Session_dir={SESSION_DIR} --pretrained_model_name_or_path='{MODELT_NAME}' --instance_data_dir='{INSTANCE_DIR}' --output_dir='{OUTPUT_DIR}' --captions_dir='{CAPTIONS_DIR}' --instance_prompt='{PT}' --seed={Seed} --resolution={Res} --mixed_precision={precision} --train_batch_size=1 --gradient_accumulation_steps=1 {GCUNET} --use_8bit_adam --learning_rate={untlr} --lr_scheduler='linear' --lr_warmup_steps=0 --max_train_steps={Training_Steps}")

    if Enable_text_encoder_training:
        print('Training the text encoder...')
        if os.path.exists(OUTPUT_DIR + '/text_encoder_trained'):
            os.system(f"rm - r {OUTPUT_DIR}/text_encoder_trained")
        dump_only_textenc(trnonltxt, MODELT_NAME, INSTANCE_DIR,
                          OUTPUT_DIR, PT, Seed, precision, Training_Steps=stptxt)

    if Enable_Text_Encoder_Concept_Training:
        if os.path.exists(CONCEPT_DIR):
            if os.listdir(CONCEPT_DIR) != []:
                print('Training the text encoder on the concept...')
                dump_only_textenc(trnonltxt, MODELT_NAME, CONCEPT_DIR,
                                  OUTPUT_DIR, PT, Seed, precision, Training_Steps=stptxtc)
            else:
                print('No concept images found, skipping concept training...')
                Text_Encoder_Concept_Training_Steps = 0
                time.sleep(8)
        else:
            print('No concept images found, skipping concept training...')
            Text_Encoder_Concept_Training_Steps = 0
            time.sleep(8)

    if UNet_Training_Steps != 0:
        train_only_unet(stpsv, stp, SESSION_DIR, MODELT_NAME, INSTANCE_DIR,
                        OUTPUT_DIR, PT, Seed, Res, precision, Training_Steps=UNet_Training_Steps)

    if UNet_Training_Steps == 0 and Text_Encoder_Concept_Training_Steps == 0 and Text_Encoder_Training_Steps == 0:
        print('Nothing to do')

    if os.path.exists('/models/'+INSTANCE_NAME+'/unet/diffusion_pytorch_model.bin'):
        prc = "--fp16" if precision == "fp16" else ""
        os.system(
            f"python /diffusers/scripts/convertosdv2.py {prc} {OUTPUT_DIR} {SESSION_DIR}/{Session_Name}.ckpt")
        if os.path.exists(SESSION_DIR+"/"+INSTANCE_NAME+'.ckpt'):
            print(
                "\033[1;32mDONE, the CKPT model is in your Gdrive in the sessions folder")
    else:
        print("\033[1;31mSomething went wrong")
