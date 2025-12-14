# Complete Setup Guide

## Step-by-Step Installation

### 1. Verify Python Installation

```bash
python --version
```

Should show Python 3.10 or higher.

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA Support

**IMPORTANT**: Install PyTorch FIRST before other dependencies.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify CUDA Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:

```
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

### 5. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 6. Login to Hugging Face

```bash
huggingface-cli login
```

When prompted, paste your token from: https://huggingface.co/settings/tokens

### 7. Test the Installation

```bash
python generate_image.py
```

## Terminal Commands Cheat Sheet

### Running the Generator

```bash
# Basic run
python generate_image.py

# With custom prompt (edit the script first)
python generate_image.py
```

### Checking System Info

```bash
# Check GPU
nvidia-smi

# Check Python packages
pip list

# Check Hugging Face login
huggingface-cli whoami
```

### Troubleshooting Commands

```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Clear Hugging Face cache (if model is corrupted)
# Windows: Delete C:\Users\<username>\.cache\huggingface\
```

## Common Issues & Solutions

### Issue 1: "CUDA not available"

**Solution**:

1. Install/update NVIDIA drivers: https://www.nvidia.com/download/index.aspx
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue 2: "OutOfMemoryError"

**Solution**: Edit `generate_image.py`:

```python
# Reduce resolution
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384

# OR uncomment this line in the script:
pipe.enable_sequential_cpu_offload()
```

### Issue 3: "401 Unauthorized" when downloading model

**Solution**:

1. Login again: `huggingface-cli login`
2. Verify: `huggingface-cli whoami`
3. Check token has read permissions

### Issue 4: Slow generation

**Causes**:

- First run downloads model (~4GB) - this is normal
- CPU offloading enabled - trades speed for VRAM
- High inference steps - reduce to 20-25

**Check**:

```python
# In generate_image.py, ensure these are NOT uncommented:
# pipe.enable_sequential_cpu_offload()
# pipe.enable_model_cpu_offload()
```

## Performance Expectations

### RTX 4050 (8GB VRAM)

- **First run**: 5-10 minutes (downloading model)
- **Subsequent runs**: 15-30 seconds per image
- **Resolution**: 512x512 works perfectly
- **Max safe resolution**: 768x768 (with optimizations)

### Settings for Best Quality

```python
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 8.5
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
```

### Settings for Fastest Generation

```python
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
```

## Next Steps After Setup

1. **Experiment with prompts**: Edit the `PROMPT` variable
2. **Try negative prompts**: Specify what you DON'T want
3. **Adjust parameters**: Play with steps, guidance scale
4. **Batch generation**: Modify script to generate multiple images
5. **Explore advanced features**: Img2img, inpainting, LoRA

## Useful Resources

- **Prompt Guide**: https://huggingface.co/docs/diffusers/using-diffusers/write_prompts
- **Diffusers Docs**: https://huggingface.co/docs/diffusers
- **Model Card**: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- **Community**: https://huggingface.co/spaces

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review this guide's troubleshooting section
3. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
4. Verify Hugging Face login: `huggingface-cli whoami`
5. Check GPU usage: `nvidia-smi`
