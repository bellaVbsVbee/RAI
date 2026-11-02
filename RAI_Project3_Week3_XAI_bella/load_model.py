from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision.models import densenet121

ROOT = Path(__file__).resolve().parent
TARGETS = ('Cardiomegaly', 'Edema', 'Atelectasis')


def load_model(pathology='Cardiomegaly', device='cpu'):
    if pathology not in TARGETS:
        raise ValueError(f'Choose one of {TARGETS}')
    model = densenet121(weights=None)  # No download or ImageNet checkpoint needed.
    model.classifier = torch.nn.Linear(1024, 1)
    state = torch.load(ROOT / 'checkpoints' / f'{pathology}.pth',
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    # Avoid in-place operations on hooked activations when computing Grad-CAM.
    for layer in model.modules():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False
    return model.to(device).eval()


def preprocess(image):
    """PIL image -> [1,3,256,256]; apply edits BEFORE this normalization."""
    image = image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    # Match the original teaching script, including float64 normalization.
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float().unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=TARGETS, default='Cardiomegaly')
    parser.add_argument('--image', type=Path,
                        default=ROOT / 'images/02_positive_negative/02_01_original.jpg')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_model(args.model, args.device)
    with Image.open(args.image) as im:
        x = preprocess(im).to(args.device)
    with torch.no_grad():
        score = model(x).sigmoid().item()
    print(f'{args.model}: {score:.6f}')
