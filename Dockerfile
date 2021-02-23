FROM ufoym/deepo:pytorch-cu101

RUN pip install pydicom

RUN pip install efficientnet_pytorch

RUN pip install opencv-python-headless

RUN pip install pytorch-lightning
