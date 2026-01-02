# --------------------------------------------- SETUP VM ---------------------------------------------
# Make VM with spot, GPU, set OS to CUDA w 500gb harddrive, 8 vCPU, 45GB RAM, firewall turn http traffic on
# ssh-keygen -t rsa -f ~/.ssh/gcp_ssh_key -C your_gmail@gmail.com
# cat ~/.ssh/gcp_ssh_key.pub | pbcopy
# Compute Engine -> VM Instances * Select Instance * edit -> ssh keys -> Add item *paste your ssh key* -> Save
# VPC Network -> Firewall -> Create Firewall Rule
    # Targets: All instances in the network
    # Source IPv4 ranges: *enter ip*/32 *or 0.0.0.0/0 for all ips*
    # Specified protocols and ports: tcp:8188
# VPC Network -> IP Addresses
    # Set external IP to static address

# --------------------------------------------- INSTALL COMFYUI ---------------------------------------------
sudo apt-get update
sudo apt-get install -y git wget
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
sudo apt-get install -y python3-venv
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
mkdir -p models/checkpoints

# --------------------------------------------- MODELS ---------------------------------------------

source .ENV

# Ep 1 - Introduction and Installation
curl -L \
  -H "Authorization: Bearer $CIVITAI_API_KEY" \
  -o models/checkpoints/juggernautXL_ragnarokBy.safetensors \
  "https://civitai.com/api/download/models/1759168?type=Model&format=SafeTensor"

# Ep 6 - 300+ Free Art Styles
# pip install gdown
# gdown --id 1A_4-FbTyJA8TtURr7wNxtNwZI0KnKInW
wget --content-disposition "https://drive.google.com/uc?export=download&id=1A_4-FbTyJA8TtURr7wNxtNwZI0KnKInW"

# Ep 8 - Flux 1: Schnell and Dev Installation Guide
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors -P models/checkpoints
# wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors -P models/checkpoints
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors -P models/vae
wget -c https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors -P models/clip
wget -c https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors -P models/clip

# Ep 9 - How to Use SDXL ControlNet Union
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors -O models/controlnet/controlnet-union-sdxl.safetensors
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors -O models/controlnet/controlnet-openpose-sdxl.safetensors

# Ep 10 - Flux GGUF and Custom Nodes
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf  -P models/unet
# wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q8_0.gguf  -P models/unet
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q8_0.gguf -P models/clip

# Ep 11 
mkdir -p models/llm_gguf
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf  -P models/llm_gguf

# Ep 12 - Upscale Models
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth  -P models/upscale_models

# Ep 13 - Ollamas
# https://ollama.com/

# Ep 14 - Flux ControlNet Union Pro
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors -O models/controlnet/flux-dev-controlnet-union.safetensors 

# Ep 17
mkdir -p models/loras/flux
mkdir -p models/loras/sdxl
mkdir -p models/loras/sd1.5
mkdir -p models/loras/pony
mkdir -p models/loras/ponyxl
mkdir -p models/loras/wan2.2
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/aleksa-codes/flux-ghibsky-illustration/resolve/main/lora_v2.safetensors -O models/loras/flux/flux-ghibsky-illustration.safetensors
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/diabolic6045/Flux_Sticker_Lora/resolve/main/tost-2024-10-05-16-42-00-5t1cker-5ty1e.safetensors -P models/loras/flux
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/resolve/main/diffusion_pytorch_model.safetensors -P models/loras/flux
# wget https://www.shakker.ai/modelinfo/3cf25bb29e0144e4849064b122150054/Flux-Fantasy-Hide?from=models -P models/loras/flux
# https://www.shakker.ai/modelinfo/33815c53e3024899bde957fa012e1f43/TQ-Sketchy-Pastel-Anime-Flux?from=models
wget -P models/loras/flux "https://civitai.com/api/download/models/1026423?type=Model&format=SafeTensor"

# Ep 18 - Easy Photo to Cartoon Transformation!
mkdir -p models/checkpoints/sdxl
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" -o models/checkpoints/sdxl/WildCardX-XL-ANIMATION.safetensors "https://civitai.com/api/download/models/357959?type=Model&format=SafeTensor&size=full&fp=fp16"
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" -o models/checkpoints/sdxl/WildCardX-XL-ANIMATION.safetensors "https://civitai.com/api/download/models/686204?type=Model&format=SafeTensor&size=pruned&fp=fp16"

wget https://civitai.com/models/677725/cute-3d-cartoon-flux -P models/loras/flux

# Ep 19 - SDXL & Flux Inpainting Tips with ComfyUI
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" -o models/checkpoints/sdxl/WildCardX-XL-ANIMATION.safetensors "https://civitai.com/api/download/models/456538?type=Model&format=SafeTensor&size=pruned&fp=fp16"


# Ep 21 - OmniGen
mkdir -p models/LLM/OmniGen-v1
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Shitao/OmniGen-v1/resolve/main/model.safetensors -P models/LLM/OmniGen-v1

# Ep 23 - Flux Tools, Fill, Redux, Depth, Canny
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors -P models/unet
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors -P models/style_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors -P models/clip_vision
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev/resolve/main/flux1-canny-dev.safetensors -P models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev/resolve/main/flux1-depth-dev.safetensors -P models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/resolve/main/flux1-canny-dev-lora.safetensors -P models/loras/flux
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/resolve/main/flux1-depth-dev-lora.safetensors -P models/loras/flux

# Ep 24 - Unlock Flux Redux & Inpainting with LoRA
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/895148?type=Model&format=SafeTensor" -o models/loras/flux/real-mona-lisa.safetensors

# Ep 25 - LTX Video – Fast AI Video Generator Model
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.safetensors -P models/checkpoints

# Ep 27 -  Photo to Watercolor, Oil & Digital Paintings 
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/539071?type=Model&format=SafeTensor" -o models/loras/sdxl/watercolor.safetensors
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/242087?type=Model&format=SafeTensor" -o models/loras/sdxl/retro-illustration.safetensors
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/360775?type=Model&format=SafeTensor" -o models/loras/sdxl/eldritch-impressionism.safetensors
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/348077?type=Model&format=SafeTensor" -o models/loras/sdxl/eldritch-digital-art.safetensors
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/356771?type=Model&format=SafeTensor" -o models/loras/sdxl/classipeint-xl.safetensors

# Ep 28 -  Create Consistent Characters with Flux + Train Loras Online
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/951461?type=Model&format=SafeTensor" -o models/loras/flux/eyes.safetensors

# Ep 34:  Turn Images into Prompts Using DeepSeek Janus Pro
mkdir models/Janus-Pro
git clone https://huggingface.co/deepseek-ai/Janus-Pro-1B models/Janus-Pro

# Ep 36 - WAN 2.1 Installation – Turn Text & Images into Video!
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors models/diffusion_models
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors models/text_encoders
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors models/vae

# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q8_0.gguf models/diffusion_models
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors models/text_encoders
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors models/vae

wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/city96/Wan2.1-I2V-14B-720P-gguf/resolve/main/wan2.1-i2v-14b-720p-Q4_0.gguf models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors models/text_encoders
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors models/clip_vision
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors models/vae


# --------------------------------------------- WORKFLOWS ---------------------------------------------
git clone https://github.com/samgermain/comfyui-workflows
mkdir -p user/default/workflows
mv comfyui-workflows/* user/default/workflows
rm comfyui-workflows -rf

# --------------------------------------------- EXTENSIONS ---------------------------------------------
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

# Ep 33: How to Use Free & Local Text-to-Speech for AI Voiceovers
# List of voices https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
git clone https://github.com/stavsap/comfyui-kokoro.git custom_nodes/comfyui-kokoro
pip install pylatexenc
pip install custom_nodes/comfyui-kokoro/requirements.txt

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

# --------------------------------------------- RUN COMFYUI ---------------------------------------------
cd ~/ComfyUI
source venv/bin/activate
python main.py --listen 0.0.0.0 --port $PORT