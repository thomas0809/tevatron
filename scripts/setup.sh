conda create -y -n tevatron python=3.9 numpy tqdm
conda activate tevatron
mamba install -y pytorch=1.13.1 pytorch-cuda=11.6 faiss-cpu rdkit pandas scipy \
  -c pytorch -c nvidia -c conda-forge
pip install datasets networkx==2.5
pip install --editable .
