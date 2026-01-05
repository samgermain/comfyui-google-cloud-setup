# I just pushed all my custom nodes to my server from my local machine
# TODO: update to pull from github repo instead of scp
# Locally run  scp -i ~/.ssh/gcp_ssh_key  -r ~/Documents/ComfyUI/custom_nodes/* ${SERVER}:~/ComfyUI/custom_nodes

cd custom_nodes
for d in */; do
    if [ -f "$d/requirements.txt" ]; then
        pip install -r "$d/requirements.txt"
    fi
done
cd ..

# Below are the correct installations for these episodes

# Ep 33: How to Use Free & Local Text-to-Speech for AI Voiceovers
# List of voices https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
git clone https://github.com/stavsap/comfyui-kokoro.git custom_nodes/comfyui-kokoro
pip install pylatexenc
pip install custom_nodes/comfyui-kokoro/requirements.txt

pip install comfy-cli

# Ep 64: Nunchaku Qwen Image Edit 2509
git clone https://github.com/mit-han-lab/ComfyUI-nunchaku.git ~/ComfyUI/custom_nodes/ComfyUI-nunchaku
cd ~/ComfyUI/custom_nodes/ComfyUI-nunchaku
git fetch --tags
git checkout tags/v1.0.2 -b v1.0.2
cd ../../
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
TORCH_VER=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
# LATEST_RELEASE=$(grep '^version' ~/ComfyUI/custom_nodes/ComfyUI-nunchaku/pyproject.toml | head -n1 | cut -d '"' -f2)
LATEST_RELEASE="1.0.2"
WHEEL_URL=$(cat <<EOF
https://github.com/nunchaku-tech/nunchaku/releases/download/v${LATEST_RELEASE}/nunchaku-${LATEST_RELEASE}+torch${TORCH_VER}-cp3${PY_MINOR}-cp3${PY_MINOR}-linux_x86_64.whl
EOF
)
pip install --no-cache-dir "$WHEEL_URL"
# pip install --force-reinstall numpy
curl -L -o ~/ComfyUI/custom_nodes/ComfyUI-nunchaku/nunchaku_versions.json https://nunchaku.tech/cdn/nunchaku_versions.json
# SageAttention
TORCH_VER=$(python3 - <<'EOF'
import torch
v = torch.__version__.split("+")[0]
print(".".join(v.split(".")[:2]))
EOF
)

case "$TORCH_VER" in
  2.6)
    TRITON_VER="3.2.0"
    ;;
  2.7|2.8)
    TRITON_VER="3.3.1"
    ;;
  2.9|2.10|2.11|2.12)
    TRITON_VER="3.5.1"
    ;;
  *)
    echo "Unsupported Torch version: $TORCH_VER"
    exit 1
    ;;
esac

pip install --force-reinstall triton=="$TRITON_VER"

export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/$/+PTX/' | paste -sd ";")

pip uninstall -y sageattention
rm -rf ~/.cache/torch_extensions

pip install \
  --no-cache-dir \
  --force-reinstall \
  --no-build-isolation \
  git+https://github.com/thu-ml/SageAttention


# List of extensions to install:

# styles csv loader" created by theUpsider
# Was Node Suite
# ComfyUI Easy Use
# https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4 ; pip install -U bitsandbytes 
# comfyui_controlnet_aux
# Comfyroll Studio
# rgthree
# Crystools
# ID 97 Searge-LLM
# ComfyUI-Florence2 by kijai
# ComfyUI-Inspyrenet-Rembg
# ComfyUI-PixelResolutionCalculator
# ControlAltAI Nodes
# ComfyUI Ollama created by stavsap
# ComfyUI-GGUF
# comfyui itools
# ComfyUI-Inpaint-CropAndStitch
# canvas tab
# ComfyUI-OmniGen by the author 1038lab
# Advanced Reflux control
# ComfyUI-AdvancedLivePortrait
# ComfyUI-VideoHelperSuite
# ComfyUI-ToSVG
# https://github.com/stavsap/comfyui-kokoro.git
# Nodes Kokoro Speaker
# Kokoro Generator
# Save Audio
# ComfyUI-Janus-Pro created by the author cychenyue
# ComfyUI-VideoHelperSuite