from rembg.bg import remove
import mediapipe as mp
import torchvision.transforms as T
from torchvision import models
import torch
import cv2 as cv
import io
import os
import random

import numpy as np


# import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


mp_drawing = mp.solutions.drawing_utils


# input_path = 'input.png'
# output_path = 'out.png'

def create_blank_image(image):  # rgb_color=(0, 0, 255)):
    # bg_image = np.zeros(image.shape, dtype=np.uint8)
    # bg_image.fill(255)
    bg_image = 255 * np.ones_like(image).astype(np.uint8)
    return bg_image


def bgRemoval(image):
    img = np.fromfile(image)
    # img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    # img = Image.fromarray(img)
    result = remove(img)
    out = Image.open(io.BytesIO(result)).convert("RGBA")
    return np.array(out)


def mpSegment(img):
    image = cv.imread(img)
    # image = cv.resize(image, (512,512), interpolation = cv.INTER_AREA)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    background = create_blank_image(image)
    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=0) as model:
        res = model.process(image)
        mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.5
        return np.where(mask, image, background)


def decode_segmap(image, source, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128,
                                                        0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64,
                                                              0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0,
                                                           128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)

    foreground = cv.imread(source)

    foreground = cv.cvtColor(foreground, cv.COLOR_BGR2RGB)
    foreground = cv.resize(foreground, (r.shape[1], r.shape[0]))

    background = 255 * np.ones_like(rgb).astype(np.uint8)

    foreground = foreground.astype(float)
    background = background.astype(float)

    th, alpha = cv.threshold(np.array(rgb), 0, 255, cv.THRESH_BINARY)

    alpha = cv.GaussianBlur(alpha, (7, 7), 0)

    alpha = alpha.astype(float)/255

    foreground = cv.multiply(alpha, foreground)

    background = cv.multiply(1.0 - alpha, background)

    outImage = cv.add(foreground, background)

    return outImage/255


def segment(net, path, show_orig=True, dev='cpu'):
    img = Image.open(path)
    trf = T.Compose([T.Resize(450),
                     # T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    rgb = decode_segmap(om, path)
    return rgb


dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
