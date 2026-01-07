for repo in \
    https://github.com/theUpsider/ComfyUI-Styles_CSV_Loader \
    https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4 \
    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes \
    https://github.com/crystian/ComfyUI-Crystools \
    https://github.com/SeargeDP/ComfyUI_Searge_LLM \
    https://github.com/kijai/ComfyUI-Florence2 \
    https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg \
    https://github.com/Ling-APE/ComfyUI-PixelResolutionCalculator \
    https://github.com/stavsap/comfyui-ollama \
    https://github.com/MohammadAboulEla/ComfyUI-iTools \
    https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch \
    https://github.com/Lerc/canvas_tab \
    https://github.com/1038lab/ComfyUI-OmniGen \
    https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl \
    https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait \
    https://github.com/Yanick112/ComfyUI-ToSVG \
    https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro \
    https://github.com/stavsap/comfyui-kokoro.git \
    https://github.com/Comfy-Org/ComfyUI-Manager \
    https://github.com/yolain/ComfyUI-Easy-Use \
    https://github.com/Fannovel16/comfyui_controlnet_aux \
    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes \
    https://github.com/rgthree/rgthree-comfy \
    https://github.com/city96/ComfyUI-GGUF \
    https://github.com/MohammadAboulEla/ComfyUI-iTools \
    https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch \
    https://github.com/1038lab/ComfyUI-RMBG \
    https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
    https://github.com/welltop-cn/ComfyUI-TeaCache \
    https://github.com/shiimizu/ComfyUI-TiledDiffusion \
    https://github.com/kijai/ComfyUI-KJNodes \
    https://github.com/kijai/ComfyUI-WanVideoWrapper \
    https://github.com/1038lab/ComfyUI-QwenVL \
    https://github.com/nunchaku-tech/ComfyUI-nunchaku \
    https://github.com/thu-ml/SageAttention \
    https://github.com/Dao-AILab/flash-attention \
    https://github.com/visualbruno/ComfyUI-Trellis2 \
    https://github.com/deepinsight/insightface \
    https://github.com/ussoewwin/ComfyUI-QwenImageLoraLoader
    # https://github.com/WASasquatch/was-node-suite-comfyui \ Archived
do
    name=$(basename "$repo" .git)
    git clone "$repo" "$HOME/ComfyUI/custom_nodes/$name"
    if [ -f "$HOME/ComfyUI/custom_nodes/$name/requirements.txt" ]; then
        pip install -r "$HOME/ComfyUI/custom_nodes/$name/requirements.txt"
    fi
done
# one by one command
# git clone https://github.com/rgthree/rgthree-comfy ~/ComfyUI/custom_nodes/rgthree-comfy; pip install -r ~/ComfyUI/custom_nodes/rgthree-comfy/requirements.txt


# ComfyUI-ToSVG
# Nodes Kokoro Speaker
# Kokoro Generator
# Save Audio

# Ep 13: Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull gemma3 # or whatever ollama model you want

# Ep 64: Nunchaku Qwen Image Edit 2509
git clone https://github.com/mit-han-lab/ComfyUI-nunchaku.git ~/ComfyUI/custom_nodes/ComfyUI-nunchaku
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
TORCH_VER=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
LATEST_RELEASE=$(grep '^version' ~/ComfyUI/custom_nodes/ComfyUI-nunchaku/pyproject.toml | head -n1 | cut -d '"' -f2)
WHEEL_URL=$(cat <<EOF
https://github.com/nunchaku-tech/nunchaku/releases/download/v${LATEST_RELEASE}/nunchaku-${LATEST_RELEASE}+torch${TORCH_VER}-cp3${PY_MINOR}-cp3${PY_MINOR}-linux_x86_64.whl
EOF
)
pip install --no-cache-dir "$WHEEL_URL"
pip install -r ~/ComfyUI/custom_nodes/ComfyUI-nunchaku/requirements.txt
# pip install --force-reinstall numpy
curl -L -o ~/ComfyUI/custom_nodes/ComfyUI-nunchaku/nunchaku_versions.json https://nunchaku.tech/cdn/nunchaku_versions.json

# SageAttention
pip install --force-reinstall triton==3.5.1
# export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/$/+PTX/' | paste -sd ";")
pip uninstall -y sageattention
rm -rf ~/.cache/torch_extensions
pip install  --no-cache-dir  --force-reinstall  --no-build-isolation git+https://github.com/thu-ml/SageAttention

