# Pytorch 練習

## Docker 環境構築

### イメージ作成

``` bash
docker build --no-cache --force-rm=true \
    --build-arg http_proxy=http://172.20.2.2:8080 \
    --build-arg https_proxy=http://172.20.2.2:8080 \
    -t study_pytorch_image:1.0 \
    build
```

### コンテナ作成

``` bash
docker run --gpus all -it --privileged \
    -v `pwd`:/home/ubuntu/workspace \
    --env HTTP_PROXY=http://172.20.2.2:8080 \
    --env HTTPS_PROXY=http://172.20.2.2:8080 \
    --env TZ=Asia/Tokyo \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    --env DISPLAY=$DISPLAY \
    --env QT_X11_NO_MITSHM=1 \
    --device /dev/video0:/dev/video0:mwr \
    --cpus=$(cat /proc/cpuinfo | grep processor | wc -l | perl -ne 'print $_ * 0.98') \
    --shm-size=2gb \
    --name study_pytorch_container study_pytorch_image:1.0 bash
```
