# Week 3: Where does a CXR model look?

Tired of working with radiology reports? Back with a small vision task! 

Use XAI/Saliency method on three already-trained, binary DenseNet-121 models: **Cardiomegaly, Edema and Atelectasis**. 

Important: No training needed. `images.csv` identifies each image, its task, selection reason, original labels, age, sex and acquisition view. This is subset just for this exercise, do not redistribute and not clinically verified.

## Start

(CPU is sufficient):

```bash
python load_model.py
```

To help you out load models:

```python
from PIL import Image
from load_model import load_model, preprocess

model = load_model('Cardiomegaly')  # also 'Edema' or 'Atelectasis'
image = Image.open('images/02_positive_negative/02_01_original.jpg').convert('RGB')
x = preprocess(image)
logit = model(x)                    # shape [1, 1]: positive-finding logit
score = logit.sigmoid().item()      # score in [0, 1]; not calibrated confidence
```

Choose a Grad-CAM target layer and explain the **positive logit** `model(x)[0, 0]`. Do not use softmax (there is only one output), `no_grad()` or `inference_mode()` while computing Grad-CAM to keep the model in evaluation mode. Use RGB, a 256×256 resize and ImageNet normalization; apply image edits before normalization.

## Four tests

| Test | Images to use | What to do |
|---|---|---|
| 1. When does a model look | `01_compare_models/`: all three images | Run all three models on each image. Compare prediction/logits and Grad-CAM. Hint: do highlighted regions differ with the predicted pathology? Which regions appear consistent with it? |
| 2. Positive vs negative | `02_positive_negative/`: 01–03 cardiomegaly-positive; 04–06 low-score comparisons | Use the Cardiomegaly model. Compare the cardiac region and other areas. Images 04 and 06 have explicit **No Finding** labels; 05 does not. |
| 3. Text shortcut | `03_text_shortcut/`: four original/text pairs | Use the Atelectasis model. Compare each original with its `PORTABLE AP` version. Add unrelated text of similar size at the same position as a control. Does the effect depend on the words or simply added pixels or s....c..? |
| 4. Heart vs lung occlusion | `02_positive_negative/02_01_original.jpg` and `02_02_original.jpg` | Resize to 256×256. **Draw your own.** Occlude the heart-region rectangle and an lung-region control avoiding the heart. Compare original/heart-hidden/lung-hidden Cardiomegaly scores and maps. Interpretation on what it does and what you onbserved. |

What we will like to see in poster for this part ?
A small panel of images/heatmaps, scores and a few sentences interpreting each test (or whatever you find interesting). Make sure it is readable and easy to follow.

**Be careful:** Make sure to note the limitations of heatmaps - they show sensitivity for an output, not proof of a diagnosis or causal reasoning.

**Optional information on labels:** original columns use `1` positive, `0` negative, `-1` uncertain, blank unmentioned. The original study treated uncertain as positive and blanks as zero; neither is a confirmed positive/negative diagnosis. In `01_01`, edema is uncertain; `01_02` and `01_03` are explicitly positive for all three targets. Synthetic text does not change labels or the original PA acquisition view.

The folder contains 13 original images, four synthetic text variants and three portable checkpoints. `checkpoint_provenance.json` records the source models. 
