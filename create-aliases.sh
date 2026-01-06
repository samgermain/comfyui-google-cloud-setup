
source ~/.ENV
source ~/comfyui-google-cloud-setup/.ENV

# DEFAULT_SERVER="gmail_username@00.00.00.000"
DEFAULT_PORT="9999"

if [ -z "$PORT" ] || [ "$PORT" = "$DEFAULT_PORT" ]; then
    echo "Error: PORT is not set or is using default placeholders. Please define it in your ~/.ENV file."
    exit 1
fi

echo 'alias comfyui="python ~/ComfyUI/main.py --listen 0.0.0.0 --port '${PORT}'"' > ~/.bash_aliases  # create shortcut
