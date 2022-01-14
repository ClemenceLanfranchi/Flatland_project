env2lmod
#module load gcc/6.3.0 python_gpu/3.8.5 cuda/10.1.243 eth_proxy

module load gcc/6.3.0 python_gpu/3.7.4 eth_proxy

if [ ! -d "venv/" ]; then
    python3 -m venv flatland-rl
    echo "Created virtual environment."
fi

source flatland-rl/bin/activate
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U flatland-rl
pip install pluggy==0.13.1
pip install pyparsing==2.4.7
pip install tensorboard
pip install psutil
pip install wandb 