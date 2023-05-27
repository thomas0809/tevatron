conda create -y -n tevatron python=3.9 numpy tqdm -c conda-forge
conda activate tevatron
#mamba install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
mamba install -y faiss-cpu rdkit -c conda-forge
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install accelerate datasets networkx==2.5
pip install --editable .
