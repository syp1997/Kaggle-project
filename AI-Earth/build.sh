#!/bin/bash

export REPOBASE=registry.cn-shenzhen.aliyuncs.com
export REPO=registry.cn-shenzhen.aliyuncs.com/yinpei_su/syp_ai_earth_01
export USRNAME=苏小沛syp
export VER=4.5

docker  login --username=$USRNAME $REPOBASE
docker build  -t $REPO:$VER  .
docker push $REPO:$VER