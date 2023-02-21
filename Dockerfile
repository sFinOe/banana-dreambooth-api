# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /content

# Install git
RUN apt-get update && apt-get install -y git unzip wget curl

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt /content/requirements.txt
RUN pip3 install -r /content/requirements.txt

RUN mkdir -p /content/data/1676642713542 && \
	mkdir -p /content/output

COPY ./dataset /content/data/1676642713542

RUN git clone https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-person_ddim.git /content/data/woman && \
	curl https://rclone.org/install.sh | bash

RUN wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py && \
	wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py && \
	pip3 install -q -U --pre triton

# We add the banana boilerplate here
ADD server.py /content/
EXPOSE 8000

# Add your huggingface auth key here
ENV HF_AUTH_TOKEN=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

# Add your custom app code, init() and inference()
ADD app.py /content/

CMD python3 -u server.py
