FROM python:3.11.4

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch torchvision torchaudio
RUN pip3 install --no-cache-dir torch-geometric
RUN pip3 install --no-cache-dir scikit-fuzzy
RUN pip3 install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

RUN pip3 install --no-cache-dir pytest

RUN apt install git
RUN git clone https://github.com/Adrisui3/TBMA-GNAS.git

WORKDIR TBMA-GNAS/

CMD ["python3", "-m", "tbma_gnas.experiments.main"]
