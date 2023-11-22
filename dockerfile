FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch-geometric
RUN pip3 install --no-cache-dir scikit-fuzzy
RUN pip3 install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

RUN pip3 install --no-cache-dir pytest

RUN apt update && apt install -y git
RUN git clone https://github.com/Adrisui3/TB-NAS.git

WORKDIR TB-NAS/

CMD ["python3", "-m", "tb_nas.experiments.main"]
