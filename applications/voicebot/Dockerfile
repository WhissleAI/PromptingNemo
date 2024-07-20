FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get -y install python3.10 python3-pip python-is-python3 openmpi-bin libopenmpi-dev git git-lfs ffmpeg curl

RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
        | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
        && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
        | tee /etc/apt/sources.list.d/ngrok.list \
        && apt update \
        && apt install ngrok

COPY requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt
