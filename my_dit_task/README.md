

# 登录集群

ssh chenyidong@1.92.123.254 -i  C:\Users\Lenovo\.ssh\id_rsa_

# 环境配置实例
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh
source ~/.bashrc


pip install torch
pip install diffusers
pip install timm

# 运行实例
srun -p compute -N 1  python sample.py --ckpt /home/data/DiT-XL-2-256x256.pt >> /home/chenyidong/output.log


