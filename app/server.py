import pandas as pd
import glob
import os
import aiohttp
import asyncio
import uvicorn
import base64
import json
import logging
from random import randint
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from fastai.vision import *

# Based on https://github.com/render-examples/fastai-v3/blob/master/app/server.py

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)

async def setup_learners():
    await download_file(MODEL_FILE_URL, path / MODEL_FILE_NAME)
    await download_file(BRAND_FILE_URL, path / BRAND_FILE_NAME)
    global brand_learner, model_learner
    model_learner = load_learner(path, MODEL_FILE_NAME)
    brand_learner = load_learner(path, BRAND_FILE_NAME)

def save_image(img_data, learn, model):
    img_bytes = base64.b64decode(img_data)
    img = open_image(BytesIO(img_bytes))
    transformed_img, pred_class, pred_idx, outputs = learn.predict(img, return_x=True)
    transformed_img.save(f'app/static/{model}_{randint(1,1000)}.jpg')

def get_prediction(img_data, learn):
    img_bytes = base64.b64decode(img_data)
    img = open_image(BytesIO(img_bytes))
    pred_class, pred_idx, outputs = learn.predict(img)
    top3_probs, top3_idxs = torch.topk(outputs, k=3)
    classes = np.array(learn.data.classes)
    top3_classes = list(classes[top3_idxs])
    top3_probs = [str(x) for x in list(np.array(top3_probs))]
    return top3_classes, top3_probs

MODEL_FILE_URL = 'https://www.dropbox.com/s/myqut34h04mhguk/export_model.pkl?dl=1'
BRAND_FILE_URL = 'https://www.dropbox.com/s/tbd3v1zw9t07rfw/export_brand.pkl?dl=1'
MODEL_FILE_NAME = 'export_model.pkl'
BRAND_FILE_NAME = 'export_brand.pkl'

path = Path(__file__).parent
app = Starlette(on_startup=[setup_learners])
app.mount('/app/static', StaticFiles(directory='app/static'))

@app.route('/')
async def homepage(request):
    return HTMLResponse('<h1>Welcome to the awesome car api!</h1>')

@app.route('/debug')
async def debug(request):
    image_files = glob.glob('app/static/*.jpg')
    models = [image.split('/')[-1].split('_')[0] for image in image_files]
    response_string = "".join([f'<img src="{image}"> <br> <p>{models[i]}</p> <br>' for i, image in enumerate(image_files)])
    return HTMLResponse(response_string)

@app.route('/predict', methods=['POST'])
async def predict(request):
    img_data = await request.body()
    top3_brand_classes, top3_brand_probs = get_prediction(img_data, brand_learner)
    top3_model_classes, top3_model_probs = get_prediction(img_data, model_learner)
    save_image(img_data, model_learner, top3_brand_classes[0])
    return JSONResponse({'brand_classes': top3_brand_classes,
			 'brand_probs': top3_brand_probs,
			 'model_classes': top3_model_classes,
			 'model_probs': top3_model_probs})

if __name__== '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level='info')

