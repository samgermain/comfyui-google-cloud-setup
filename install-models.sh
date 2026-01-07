# --------------------------------------------- MODELS ---------------------------------------------

source ~/.ENV
source ~/comfyui-google-cloud-setup/.ENV

# Default placeholders
# DEFAULT_SERVER="gmail_username@00.00.00.000"
# DEFAULT_PORT="9999"

# Check if variables inside .ENV are set
if [ -z "$HF_TOKEN" ] || [ -z "$CIVITAI_API_KEY" ]; then
    echo "Error: One or more required environment variables (HF_TOKEN, CIVITAI_API_KEY) are not set or are using default placeholders. Please define them all in your ~/.ENV file."
    exit 1
fi

cd ~/ComfyUI

# Ep 1 - Introduction and Installation
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" -o ~/ComfyUI/models/checkpoints/juggernautXL_ragnarokBy.safetensors "https://civitai.com/api/download/models/1759168?type=Model&format=SafeTensor"

# Ep 6 - 300+ Free Art Styles
# pip install gdown
# gdown --id 1A_4-FbTyJA8TtURr7wNxtNwZI0KnKInW
wget --content-disposition "https://drive.google.com/uc?export=download&id=1A_4-FbTyJA8TtURr7wNxtNwZI0KnKInW"

# Ep 8 - Flux 1: Schnell and Dev Installation Guide
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors -P ~/ComfyUI/models/checkpoints
# wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors -P ~/ComfyUI/models/checkpoints
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors -P ~/ComfyUI/models/vae
wget -c https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors -P ~/ComfyUI/models/clip
wget -c https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors -P ~/ComfyUI/models/clip

# Ep 9 - How to Use SDXL ControlNet Union
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors -O ~/ComfyUI/models/controlnet/controlnet-union-sdxl.safetensors
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors -O ~/ComfyUI/models/controlnet/controlnet-openpose-sdxl.safetensors

# Ep 10 - Flux GGUF and Custom Nodes
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf  -P ~/ComfyUI/models/unet
# wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q8_0.gguf  -P ~/ComfyUI/models/unet
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q8_0.gguf -P ~/ComfyUI/models/clip

# Ep 11 
mkdir -p ~/ComfyUI/models/llm_gguf
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf  -P ~/ComfyUI/models/llm_gguf

# Ep 12 - Upscale Models
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth  -P ~/ComfyUI/models/upscale_models


# Ep 14 - Flux ControlNet Union Pro
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors -O ~/ComfyUI/models/controlnet/flux-dev-controlnet-union.safetensors 

# Ep 17
mkdir -p ~/ComfyUI/models/loras/flux
mkdir -p ~/ComfyUI/models/loras/sdxl
mkdir -p ~/ComfyUI/models/loras/sd1.5
mkdir -p ~/ComfyUI/models/loras/pony
mkdir -p ~/ComfyUI/models/loras/ponyxl
mkdir -p ~/ComfyUI/models/loras/wan2.2
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/aleksa-codes/flux-ghibsky-illustration/resolve/main/lora_v2.safetensors -O ~/ComfyUI/models/loras/flux/flux-ghibsky-illustration.safetensors
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/diabolic6045/Flux_Sticker_Lora/resolve/main/tost-2024-10-05-16-42-00-5t1cker-5ty1e.safetensors -P ~/ComfyUI/models/loras/flux
wget --header="Authorization: Bearer ${HF_TOKEN}" -c https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/resolve/main/diffusion_pytorch_model.safetensors -P ~/ComfyUI/models/loras/flux
# wget https://www.shakker.ai/modelinfo/3cf25bb29e0144e4849064b122150054/Flux-Fantasy-Hide?from=models -P ~/ComfyUI/models/loras/flux
# https://www.shakker.ai/modelinfo/33815c53e3024899bde957fa012e1f43/TQ-Sketchy-Pastel-Anime-Flux?from=models
wget -P ~/ComfyUI/models/loras/flux "https://civitai.com/api/download/models/1026423?type=Model&format=SafeTensor"

# Ep 18 - Easy Photo to Cartoon Transformation!
mkdir -p ~/ComfyUI/models/checkpoints/sdxl
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" -o ~/ComfyUI/models/checkpoints/sdxl/WildCardX-XL-ANIMATION.safetensors "https://civitai.com/api/download/models/357959?type=Model&format=SafeTensor&size=full&fp=fp16"
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" -o ~/ComfyUI/models/checkpoints/sdxl/WildCardX-XL-ANIMATION.safetensors "https://civitai.com/api/download/models/686204?type=Model&format=SafeTensor&size=pruned&fp=fp16"

wget https://civitai.com/models/677725/cute-3d-cartoon-flux -P ~/ComfyUI/models/loras/flux

# Ep 19 - SDXL & Flux Inpainting Tips with ComfyUI
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" -o ~/ComfyUI/models/checkpoints/sdxl/WildCardX-XL-ANIMATION.safetensors "https://civitai.com/api/download/models/456538?type=Model&format=SafeTensor&size=pruned&fp=fp16"

# Ep 21 - OmniGen
mkdir -p ~/ComfyUI/models/LLM/OmniGen-v1
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Shitao/OmniGen-v1/resolve/main/model.safetensors -P ~/ComfyUI/models/LLM/OmniGen-v1

# Ep 23 - Flux Tools, Fill, Redux, Depth, Canny
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/resolve/main/flux1-fill-dev.safetensors -P ~/ComfyUI/models/unet
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev/resolve/main/flux1-redux-dev.safetensors -P ~/ComfyUI/models/style_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors -P ~/ComfyUI/models/clip_vision
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev/resolve/main/flux1-canny-dev.safetensors -P ~/ComfyUI/models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev/resolve/main/flux1-depth-dev.safetensors -P ~/ComfyUI/models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/resolve/main/flux1-canny-dev-lora.safetensors -P ~/ComfyUI/models/loras/flux
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora/resolve/main/flux1-depth-dev-lora.safetensors -P ~/ComfyUI/models/loras/flux

# Ep 24 - Unlock Flux Redux & Inpainting with LoRA
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/895148?type=Model&format=SafeTensor" -o ~/ComfyUI/models/loras/flux/real-mona-lisa.safetensors

# Ep 25 - LTX Video – Fast AI Video Generator Model
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.safetensors -P ~/ComfyUI/models/checkpoints

# Ep 27 -  Photo to Watercolor, Oil & Digital Paintings 
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/539071?type=Model&format=SafeTensor" -o ~/ComfyUI/models/loras/sdxl/watercolor.safetensors
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/242087?type=Model&format=SafeTensor" -o ~/ComfyUI/models/loras/sdxl/retro-illustration.safetensors
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/360775?type=Model&format=SafeTensor" -o ~/ComfyUI/models/loras/sdxl/eldritch-impressionism.safetensors
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/348077?type=Model&format=SafeTensor" -o ~/ComfyUI/models/loras/sdxl/eldritch-digital-art.safetensors
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/356771?type=Model&format=SafeTensor" -o ~/ComfyUI/models/loras/sdxl/classipeint-xl.safetensors

# Ep 28 -  Create Consistent Characters with Flux + Train Loras Online
curl -L -H "Authorization: Bearer $CIVITAI_API_KEY" "https://civitai.com/api/download/models/951461?type=Model&format=SafeTensor" -o ~/ComfyUI/models/loras/flux/eyes.safetensors

# Ep 34:  Turn Images into Prompts Using DeepSeek Janus Pro
mkdir models/Janus-Pro
git clone https://huggingface.co/deepseek-ai/Janus-Pro-1B models/Janus-Pro

# Ep 36 - WAN 2.1 Installation – Turn Text & Images into Video!
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors -P ~/ComfyUI/models/diffusion_models
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors -P ~/ComfyUI/models/text_encoders
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors -P ~/ComfyUI/models/vae

# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q8_0.gguf -P ~/ComfyUI/models/diffusion_models
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors -P ~/ComfyUI/models/text_encoders
# wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors -P ~/ComfyUI/models/vae

wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/city96/Wan2.1-I2V-14B-720P-gguf/resolve/main/wan2.1-i2v-14b-720p-Q4_0.gguf -P ~/ComfyUI/models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors -P ~/ComfyUI/models/text_encoders
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors -P ~/ComfyUI/models/clip_vision
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors -P ~/ComfyUI/models/vae

# Ep 55:  Sage Attention, Wan Fusion X, Wan 2.2 & Video Upscale Tips
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q4_K_M.gguf -P ~/ComfyUI/models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors -P ~/ComfyUI/models/text_encoders
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors -P ~/ComfyUI/models/vae

# Ep 61:
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors -P ~/ComfyUI/models/checkpoints
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors -P ~/ComfyUI/models/clip_visions
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/USO_1.0_Repackaged/resolve/main/split_files/model_patches/uso-flux1-projector-v1.safetensors -P  ~/ComfyUI/models/model_patches
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/USO_1.0_Repackaged/resolve/main/split_files/loras/uso-flux1-dit-lora-v1.safetensors -P ~/ComfyUI/models/loras/flux

# Ep 62: Nunchaku Update | Qwen Control Net, Qwen Edit & Inpaint
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/nunchaku-tech/nunchaku-qwen-image/resolve/main/svdq-fp4_r128-qwen-image.safetensors -P ~/ComfyUI/models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors -P ~/ComfyUI/models/diffusion_models
# vae
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors -P ~/ComfyUI/models/vae
# control net
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/diffusion_pytorch_model.safetensors -O ~/ComfyUI/models/controlnet/qwen-image-instantx-controlnet-union.safetensors
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/diffusion_pytorch_model.safetensors -O ~/ComfyUI/models/controlnet/qwen-image-instantx-controlnet-inpainting.safetensors
# qwen edit
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/nunchaku-tech/nunchaku-qwen-image-edit/resolve/main/svdq-fp4_r128-qwen-image-edit.safetensors -P ~/ComfyUI/models/diffusion_models

# Ep 64: Nunchaku Qwen Image Edit 2509
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/nunchaku-tech/nunchaku-qwen-image-edit/resolve/main/svdq-int4_r32-qwen-image-edit.safetensors -P ~/ComfyUI/models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors -P ~/ComfyUI/models/clip
# qwen_image_vae.safetensors

# Ep 65: VibeVoice Free Text to Speech Workflow
mkdir ~/ComfyUI/models/vibevoice/
mkdir ~/ComfyUI/models/vibevoice/tokenizer/
mkdir ~/ComfyUI/models/vibevoice/VibeVoice-1.5B/

wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/tokenizer_config.json -P ~/ComfyUI/models/vibevoice/tokenizer/
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/merges.txt -P ~/ComfyUI/models/vibevoice/tokenizer/
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/tokenizer.json -P ~/ComfyUI/models/vibevoice/tokenizer/
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main/vocab.json -P ~/ComfyUI/models/vibevoice/tokenizer/

wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/config.json -P ~/ComfyUI/models/vibevoice/VibeVoice-1.5B/
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/model.safetensors.index.json -P ~/ComfyUI/models/vibevoice/VibeVoice-1.5B/
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/model-00001-of-00003.safetensors -P ~/ComfyUI/models/vibevoice/VibeVoice-1.5B/
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/model-00002-of-00003.safetensors -P ~/ComfyUI/models/vibevoice/VibeVoice-1.5B/
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/model-00003-of-00003.safetensors -P ~/ComfyUI/models/vibevoice/VibeVoice-1.5B/
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/microsoft/VibeVoice-1.5B/resolve/main/preprocessor_config.json -P ~/ComfyUI/models/vibevoice/VibeVoice-1.5B/

# Ep 66:  Qwen Outpainting Workflow + Subgraph Tips
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Qwen-Image-InstantX-ControlNets/resolve/main/split_files/controlnet/Qwen-Image-InstantX-ControlNet-Inpainting.safetensors -P ~/ComfyUI/models/controlnet

# Ep 67: Fluxmania Nunchaku + Wan 2.2 and Rapid AIO Workflows
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q4_K_M.GGUF -P ~/ComfyUI/models/diffusion_models # https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/tree/main
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q4_K_M.gguf -P ~/ComfyUI/models/diffusion_models # https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/tree/main
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors -P ~/ComfyUI/models/text_encoders
# wan_2.1_vae.safetensors
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors -P ~/ComfyUI/models/loras/wan22
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors -P ~/ComfyUI/models/loras/wan22

# Ep 68: How to Create Anime Illustrations - NetaYume v3.5
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0/resolve/main/NetaYumev35_pretrained_all_in_one.safetensors -P ~/ComfyUI/models/checkpoints

# Ep 69:  Create Long Videos with LongCat Video Model
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Kijai/LongCat-Video_comfy/resolve/main/LongCat_TI2V_comfy_bf16.safetensors -P ~/ComfyUI/models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Kijai/LongCat-Video_comfy/resolve/main/LongCat_distill_lora_alpha64_bf16.safetensors -P ~/ComfyUI/models/loras/wan
# wan_2.1_vae.safetensors
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/ALGOTECH/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors -P ~/ComfyUI/models/clip

# Ep 70: Nunchaku Qwen Loras - Relight, Camera Angle & Scene Change 
# See Ep 64 and 62 models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/ostris/qwen_image_edit_2509_shirt_design/resolve/main/qwen_image_edit_2509_shirt_design.safetensors -P ~/ComfyUI/models/loras/qwen

# Ep 72: Z-Image Turbo Workflows, ControlNet Essentials & LoRA Training
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors -P ~/ComfyUI/models/diffusion_models
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors -P ~/ComfyUI/models/text_encoders

# Ep 73: Final Episode & Z-Image ControlNet 2.0 
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/SeeSee21/Z-Image-Turbo-AIO/resolve/main/z-image-turbo-bf16-aio.safetensors -P ~/ComfyUI/models/checkpoints
wget --header="Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors -P ~/ComfyUI/models/model_patches