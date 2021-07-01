FROM tensorflow/tensorflow:1.15.4-gpu-py3
COPY * /cibi
WORKDIR /cibi
RUN pip install -r requirements.txt