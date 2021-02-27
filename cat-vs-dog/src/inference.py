import argparse
from PIL import Image
import model_dispatcher
import config
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms


def inference(img_path):
    # classes for inference
    classes = {
        0: "cat",
        1: "dog"
    }
    # initialize model
    model = model_dispatcher.CatDogNN(config.MODEL_IN_FEATURES, config.MODEL_OUT_FEATURES)
    # load parameters of the model
    model.load_state_dict(torch.load(config.BEST_MODEL, map_location=config.INFERENCE_DEVICE))
    # send model to cpu for inference
    model = model.to(config.INFERENCE_DEVICE)
    # load image for inference
    img = Image.open(img_path)
    # specify transformation
    transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    # transform the image
    img = transform(img)
    # unsqueeze the image to add additional dimension
    img = torch.unsqueeze(img, dim=0)
    # predict probability of each class
    with torch.no_grad():
        model.eval()
        out = model(img)
    # obtain best probability
    softmax = nn.Softmax(dim=1)
    out_y = softmax(out)
    prob = torch.max(out_y)
    prob = np.float(prob.detach().cpu().numpy()) * 100
    # obtain predicted class
    out = torch.max(out, dim=1)[1]
    out = out.detach().cpu().numpy()[0]

    print(f"This is a {classes[out]} with {round(prob, 2)}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_path",
        type=str
    )
    args = parser.parse_args()
    inference(args.img_path)
