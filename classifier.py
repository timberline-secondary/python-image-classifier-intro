import ast
from pathlib import Path

import torch
from PIL import Image
import torchvision.models as tv_models

# Supported models: name -> (model builder, pretrained ImageNet weights)
MODEL_BUILDERS = {
    'resnet': (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1),
    'alexnet': (tv_models.alexnet, tv_models.AlexNet_Weights.IMAGENET1K_V1),
    'vgg': (tv_models.vgg16, tv_models.VGG16_Weights.IMAGENET1K_V1),
}

# Models are loaded (and their weights downloaded) only the first time
# they are used, then kept here so later calls are fast.
_loaded_models = {}

# obtain ImageNet labels (path is relative to this file, so it works
# no matter which folder you run Python from)
_labels_path = Path(__file__).resolve().parent / 'imagenet1000_clsid_to_human.txt'
with open(_labels_path) as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())


def _get_model(model_name):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of: {', '.join(MODEL_BUILDERS)}"
        )
    if model_name not in _loaded_models:
        builder, weights = MODEL_BUILDERS[model_name]
        model = builder(weights=weights)
        # puts model in evaluation mode
        # instead of training mode (the default)
        model.eval()
        _loaded_models[model_name] = (model, weights.transforms())
    return _loaded_models[model_name]


def classifier(img_path, model_name):
    model, preprocess = _get_model(model_name)

    # load the image, converting it to 3-channel RGB
    # (handles PNGs with transparency, grayscale images, etc.)
    img_pil = Image.open(img_path).convert('RGB')

    # preprocess the image (resize, crop, convert to tensor, normalize)
    img_tensor = preprocess(img_pil)

    # resize the tensor (add dimension for batch)
    img_tensor = img_tensor.unsqueeze(0)

    # apply model to input (no_grad: we're only predicting, not training)
    with torch.no_grad():
        output = model(img_tensor)

    # return index corresponding to predicted class
    pred_idx = output.argmax().item()

    return imagenet_classes_dict[pred_idx]
