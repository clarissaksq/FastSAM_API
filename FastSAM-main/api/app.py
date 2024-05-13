
import matplotlib
matplotlib.use('Agg') 
from fastsam import FastSAM, FastSAMPrompt
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import io 
import matplotlib.pyplot as plt
from datetime import datetime



app=Flask(__name__)
app.config['ALLOWED_EXTENSIONS']={'png','jpg','jpeg'}
model = FastSAM('./weights/FastSAM-x.pt')
DEVICE = 'cpu'
log_dir = "logs/"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/predictLogs',methods=['POST'])
def predictLogs():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename =='':
        return jsonify({'error': 'No selected file'}), 400
    if not(file and allowed_file(file.filename)):
        return jsonify({'error': 'File type not allowed'}), 400
    xyz = float(request.args.get('xyz'))  
    type = request.args.get('type')
    image_stream = io.BytesIO()
    file.save(image_stream)
    image_stream.seek(0) 
    image = Image.open(image_stream)
    everything_results = model(image, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)
    if type=="top":
        ann = prompt_process.text_prompt_old(text='solar panels',rank=xyz)
    elif type=="threshold":
        ann = prompt_process.text_prompt_threshold(text='solar panels',threshold=xyz)

    mask_coordinates = []
    num = len(ann)
    for mask in ann:
        mask = np.array(mask)  
        coords = np.argwhere(mask == True) 
        mask_coordinates.append(coords.tolist())  
    
    heatmap = np.zeros((image.height, image.width))
    for mask in ann:
        heatmap += mask
    heatmap = heatmap.clip(0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title('Input Image')
    ax[1].imshow(heatmap, alpha=0.5, cmap='hot')
    ax[1].set_title('Heatmap')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[2].text(0.5, 0.5, f'Type: {type}\nTopX/Threshold: {xyz}', fontsize=12, ha='center', va='center')
    plt.tight_layout()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_filename = f"{log_dir}/{current_time}.png"
    plt.savefig(image_filename)
    plt.close()

    return jsonify(
        {
            'number of masks': num,
            'coordinates': mask_coordinates
        }
        ), 200

@app.route('/predict',methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename =='':
        return jsonify({'error': 'No selected file'}), 400
    if not(file and allowed_file(file.filename)):
        return jsonify({'error': 'File type not allowed'}), 400
    xyz = float(request.args.get('xyz'))  
    type = request.args.get('type')
    image_stream = io.BytesIO()
    file.save(image_stream)
    image_stream.seek(0) 
    image = Image.open(image_stream)
    everything_results = model(image, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)
    if type=="top":
        ann = prompt_process.text_prompt_old(text='solar panels',rank=xyz)
    elif type=="threshold":
        ann = prompt_process.text_prompt_threshold(text='solar panels',threshold=xyz)

    mask_coordinates = []
    num = len(ann)
    for mask in ann:
        mask = np.array(mask)  
        coords = np.argwhere(mask == True) 
        mask_coordinates.append(coords.tolist())  

    return jsonify(
        {
            'number of masks': num,
            'coordinates': mask_coordinates
        }
        ), 200


if __name__ == '__main__':
    app.run(debug=True)