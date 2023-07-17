FROM python:3.11.4
WORKDIR /tbma-gnas

RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir pytest

CMD ["bash"]