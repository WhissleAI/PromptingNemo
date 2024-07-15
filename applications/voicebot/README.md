run and build cuda docker

```
docker-compose up --build --force-recreate

docker run --rm -v /external:/external -v /home/ksingla/workspace:/workspace --gpus all --entrypoint /bin/bash -it --name workbench_karan WhissleAI/workbench:latest

```