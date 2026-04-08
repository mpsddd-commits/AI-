## [Unsloth](https://unsloth.ai/docs/get-started/install/docker)

- Docker 설치
```bash
docker run -d -e JUPYTER_PASSWORD=1234 -p 8888:8888 -p 2222:22 -v ./work:/workspace/work --gpus all --name unsloth unsloth/unsloth
```

- Container 확인
```bash
docker ps
```

- [Jupyter 접속](http://localhost:8888)

- GPU 확인
```bash
nvidia-smi
```

- 실시간 로그 확인
```bash
docker inspect --format='{{json .State.Health}}' unsloth
````
