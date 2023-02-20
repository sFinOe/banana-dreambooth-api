# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install git wget curl unzip -y

RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN apt-get update && apt-get install -y \
	--yes \
	libunwind8 \
	liblzma-dev \
	libunwind8-dev \
	libgoogle-perftools4 \
	libtcmalloc-minimal4 \
	zstd \
	google-perftools

RUN pip install --no-cache-dir accelerate==0.12.0 

RUN wget -q -i https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/Dependencies/db.txt \
	&& dpkg -i *.deb \
	&& tar -C / --zstd -xf db_deps.tar.zst \
	&& rm *.deb *.zst *.txt


RUN git clone --depth 1 --branch updt https://github.com/TheLastBen/diffusers /diffusers

RUN pip install ipython tqdm google

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
	apt-get install git-lfs && \
	pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

RUN pip uninstall protobuf --yes && \
	pip install protobuf==3.19.6 absl-py grpcio markdown tensorboard-data-server tensorboard-plugin-wit werkzeug tb-nightly pytorch-lightning fsspec[http] lmdb scipy torchvision future Pillow google-auth-oauthlib && \
	pip install click promise filelock regex termcolor

# Set environment variable
ENV LD_PRELOAD=libtcmalloc.so

RUN pip install natsort

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Add your huggingface auth key here
ENV HF_AUTH_TOKEN=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

# Add your model weight files 
# (in this case we have a python script)

RUN mkdir -p /dataset && \
	mkdir -p /class_dir/woman && \
	mkdir -p /output

RUN curl https://rclone.org/install.sh | bash

COPY ./dataset /dataset


# ADD download.py .
# RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .
ADD training.py .

CMD python3 -u server.py
