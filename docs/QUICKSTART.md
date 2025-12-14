# 🚀 QUICK START GUIDE

## Installation (5 minutes)

### 1️⃣ Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Login to Hugging Face

```bash
huggingface-cli login
```

Paste your token from: https://huggingface.co/settings/tokens

### 4️⃣ Generate Your First Image

```bash
# Text-to-Image (Create new images)
python generate_image.py

# Image-to-Image (Modify existing images) - NEW!
python simple_img2img.py
```

---

## 🎨 Customizing Your Prompts

Edit `generate_image.py` and change these lines:

```python
# Line 20-21: Your creative prompt
PROMPT = "your amazing prompt here"
NEGATIVE_PROMPT = "what you don't want"

# Line 22-25: Quality settings
NUM_INFERENCE_STEPS = 30    # 20-50 (higher = better)
GUIDANCE_SCALE = 7.5        # 7-12 (how closely to follow prompt)
IMAGE_HEIGHT = 512          # 384, 512, or 768
IMAGE_WIDTH = 512           # 384, 512, or 768
```

---

## ⚡ Performance Tips

### For Best Quality

```python
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 8.5
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
```

### For Fastest Speed

```python
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
```

---

## 🔧 If You Get "Out of Memory" Error

Open `generate_image.py` and uncomment line 95:

```python
pipe.enable_sequential_cpu_offload()
```

OR reduce resolution:

```python
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
```

---

## 📊 What to Expect

| Metric                 | Value         |
| ---------------------- | ------------- |
| First run (download)   | 5-10 minutes  |
| Generation time        | 15-30 seconds |
| Model size             | ~4 GB         |
| Recommended resolution | 512x512       |
| Max safe resolution    | 768x768       |

---

## ✅ Verify Installation

```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check HF login
huggingface-cli whoami
```

---

## 🎯 Example Prompts to Try

```python
# Photorealistic
"a professional photo of a cat wearing sunglasses, studio lighting, 8k, highly detailed"

# Artistic
"oil painting of a sunset over the ocean, vibrant colors, impressionist style"

# Fantasy
"a magical forest with glowing mushrooms, fantasy art, detailed, ethereal lighting"

# Architecture
"modern minimalist house in the mountains, architectural photography, golden hour"
```

---

## 📁 Project Structure

```
Image Model/
├── generate_image.py    ← Text-to-Image script
├── simple_img2img.py    ← Image-to-Image script (NEW!)
├── requirements.txt     ← Dependencies
├── outputs/            ← Generated images
├── img2img_outputs/    ← Modified images
├── README.md           ← Full documentation
└── docs/               ← Detailed guides
```

---

## 🆘 Quick Troubleshooting

| Problem              | Solution                                |
| -------------------- | --------------------------------------- |
| "CUDA not available" | Update NVIDIA drivers                   |
| "401 Unauthorized"   | Run `huggingface-cli login`             |
| "Out of memory"      | Reduce resolution or enable CPU offload |
| Slow generation      | Normal on first run (downloading model) |

---

## 🎓 Next Steps

1. ✅ Generate your first image
2. 🎨 Experiment with different prompts
3. ⚙️ Adjust quality settings
4. 🔍 Try negative prompts
5. 📚 Read SETUP_GUIDE.md for advanced features

---

**Need help?** Check `SETUP_GUIDE.md` for detailed troubleshooting!
