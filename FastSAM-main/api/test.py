from fastsam import FastSAM, FastSAMPrompt
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormaps

model = FastSAM('./weights/FastSAM-x.pt')
DEVICE = 'cpu'
image_path = "./image6.png"  
image_actl = Image.open(image_path)  
everything_results = model(image_actl, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
prompt_process = FastSAMPrompt(image_actl, everything_results, device=DEVICE)
ann = prompt_process.text_prompt_old(text='roof')

#deep dive
print(len(ann))

if len(ann) == 0:
    print("No masks found.")
else:
    heatmap = np.zeros((image_actl.height, image_actl.width))
    for mask in ann:
        heatmap += mask  
    heatmap = heatmap.clip(0, 1) 

    plt.imshow(heatmap, alpha=0.5, cmap='hot') 
    plt.colorbar()  
    plt.show()  
