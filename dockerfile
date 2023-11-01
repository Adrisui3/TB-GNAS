FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir torch-geometric
RUN pip3 install --no-cache-dir scikit-fuzzy
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

RUN pip3 install --no-cache-dir pytest

RUN apt update && apt install -y git

COPY clone_and_run.sh clone_and_run.sh

CMD ["./clone_and_run.sh"]
