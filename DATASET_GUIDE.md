# How to Create Your Training Dataset

To fine-tune the model with your own images, follow these simple steps.

## 1. Create the Folder

Create a folder named `dataset` in the root of this project (where `train_lora.py` is).

## 2. Add Your Images

Place your training images into the `dataset` folder.

- **Formats**: JPG, PNG, WEBP.
- **Quantity**:
  - For a specific style/person: 10-20 images is often enough.
  - For a general concept: 50+ images is better.
- **Quality**: High quality, clear images work best.

## 3. Add Captions (Prompts)

For **each** image, you must provide a text file with the _exact same filename_ but with a `.txt` extension.

### Example Structure:

```
dataset/
    cat_01.jpg
    cat_01.txt      <-- Contains: "a photo of a funny cat wearing a hat"
    dog_playing.png
    dog_playing.txt <-- Contains: "a dog playing fetch in the park"
    ...
```

### What to write in the text file?

Describe the image as if you were prompting the AI to generate it. Include the specific keyword you want the model to learn if you are teaching it a new concept.

**Example for a specific person named "Ohwx":**

- `photo of Ohwx man standing in a garden`
- `close up portrait of Ohwx man, highly detailed`

## 4. Run Training

Once your `dataset` folder is ready, run the training script:

```bash
python train_lora.py
```

## 5. Use Your Model

After training, the model weights will be saved in `lora_output`.
To use them, update your generation script to load these weights:

```python
pipe.load_lora_weights("lora_output")
```
