import pandas as pd
import os
import aiohttp
import asyncio
import uvicorn
import base64
import json
import logging
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route
from fastai.vision import *

# Based on https://github.com/render-examples/fastai-v3/blob/master/app/server.py

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)

async def setup_learner():
    await download_file(EXPORT_FILE_URL, path / EXPORT_FILE_NAME)
    global learn
    learn = load_learner(path, EXPORT_FILE_NAME)
    return learn

EXPORT_FILE_URL = 'https://www.dropbox.com/s/6f2v6vtccrq7fsb/export.pkl?dl=1'
EXPORT_FILE_NAME = 'export.pkl'

path = Path(__file__).parent

app = Starlette(on_startup=[setup_learner])

@app.route('/')
async def homepage(request):
    logging.info('')
    return HTMLResponse('<h1>Welcome to the awesome car api!</h1>')

@app.route('/predict', methods=['POST'])
async def predict(request):
    img_data = await request.form()
    logging.info('1.')
    logging.info(img_data)
    img_bytes = base64.b64decode(str(img_data['file']))
    logging.info('2.')
    img = open_image(BytesIO(img_bytes))
    logging.info('3.')
    pred_class, pred_idx, outputs = learn.predict(img)
    logging.info('4.')
    top3_probs, top3_idxs = torch.topk(outputs, k=3)
    logging.info('5.')
    classes = np.array(learn.data.classes)
    logging.info('6.')
    top3_classes = list(classes[top3_idxs])
    logging.info('7.')
    top3_probs = [str(x) for x in list(np.array(top3_probs))]
    logging.info('8.')
    return JSONResponse({'prediction_classes': top3_classes,
                 'prediction_probs': top3_probs})

if __name__== '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level='info')

