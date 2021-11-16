# pylint: disable=missing-module-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gradio as gr
from mlfunc import *

title = "CarScan: BG Removal & Replacement"
description = '''Gradio demo CarScan Background Removal & Replacement.
To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.'''

examples = [['images/car1.jpeg'],['images/car2.jpeg'],['images/car3.jpeg'],['images/car4.jpeg'],['images/car5.jpg']]

# input_image = gr.inputs.Image(type="pil"),

def replacer(image, operation):

    if operation == "MediaPipe":
        return mpSegment(image)
    elif operation == "DeepLabV3":
        return  segment(dlab, image, show_orig=False)
    elif operation == "RemBG":
        return bgRemoval(image)



webapp = gr.Interface(replacer,
    [gr.inputs.Image(type='filepath',shape=(640, 480)), gr.inputs.Radio(["MediaPipe", "DeepLabV3", "RemBG"])],
    outputs="image",
    title = title,
    description = description,
    enable_queue = True,
    examples = examples,
    # live=True,
    # share=True,
)

webapp.launch()
