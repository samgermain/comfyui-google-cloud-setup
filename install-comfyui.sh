sudo apt-get update
sudo apt-get install -y git wget
sudo apt update
sudo apt install -y \
  build-essential curl \
  libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev \
  libsqlite3-dev libffi-dev \
  liblzma-dev tk-dev
curl https://pyenv.run | bash
cat << 'EOF' >> ~/.bashrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF
source ~/.bashrc
pyenv install 3.12.1
pyenv global 3.12.1

wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run
sudo sh cuda_13.0.0_580.65.06_linux.run --silent --toolkit --no-opengl-libs --override
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
sudo apt-get install -y python3-venv
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
mkdir -p models/checkpoints