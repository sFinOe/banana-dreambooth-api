# Must use a Cuda version 11+
FROM pytorch/pytorch:latest

WORKDIR /content

RUN apt update && apt install -y curl git wget unzip

RUN wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py && \
	wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py && \
	pip install -qq git+https://github.com/ShivamShrirao/diffusers && \
	pip install -q -U --pre triton && \
	pip install -q accelerate transformers ftfy gradio natsort safetensors xformers

RUN pip install bitsandbytes-cuda111

RUN pip install sanic==22.6.2 scipy boto3

RUN mkdir -p output

RUN curl https://rclone.org/install.sh | bash

RUN git clone https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-person_ddim.git class_images

ENV HF_AUTH_TOKEN=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

# We add the banana boilerplate here

ADD server.py .
EXPOSE 8000

# Add your custom app code, init() and inference()
ADD app.py .

CMD python3 -u server.py
