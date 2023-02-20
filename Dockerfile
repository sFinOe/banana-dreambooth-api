# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install git python3 python3-pip wget curl unzip -y

# Install python packages
RUN pip3 install --upgrade pip
# ADD requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt

RUN wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py && \
	wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py

RUN pip3 install -qq git+https://github.com/ShivamShrirao/diffusers && \
	pip3 install -q accelerate==0.12.0 transformers ftfy bitsandbytes gradio natsort safetensors

RUN pip3 install git+https://github.com/facebookresearch/xformers@1d31a3a#egg=xformers && \
	pip3 install natsort

RUN pip3 install sanic==22.6.2 scipy boto3

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Add your huggingface auth key here
ENV HF_AUTH_TOKEN=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

# Add your model weight files 
# (in this case we have a python script)

RUN mkdir -p /content/dataset/1676642713542 && \
	mkdir -p /content/class_dir/woman && \
	mkdir -p /content/output

RUN curl https://rclone.org/install.sh | bash

COPY ./dataset /content/dataset/1676642713542

# ADD download.py .
# RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .

CMD python3 -u server.py
