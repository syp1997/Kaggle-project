# Base Images
## 从天池基础镜像构建
FROM registry.cn-shenzhen.aliyuncs.com/yinpei_su/syp_ai_earth_01:4.0

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

##安装依赖包,pip包请在requirements.txt添加
RUN pip install netCDF4
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simplet

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
