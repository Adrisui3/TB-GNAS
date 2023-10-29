FROM python:3.11.4

RUN export OPENBLAS_NUM_THREADS = 1
RUN export OMP_NUM_THREADS = 1

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
RUN pip3 install --no-cache-dir torch-geometric
RUN pip3 install --no-cache-dir scikit-fuzzy
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

RUN pip3 install --no-cache-dir pytest

RUN git clone https://github.com/Adrisui3/TBMA-GNAS.git

WORKDIR TBMA-GNAS/

CMD ["python3", "-m", "tbma_gnas.experiments.main"]
