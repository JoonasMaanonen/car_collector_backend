import pandas as pd
import os
import aiohttp
import asyncio
import uvicorn
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
    return HTMLResponse('<h1>Welcome to the awesome car api!</h1>')

@app.route('/predict', methods=['POST'])
async def predict(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    pred_class, pred_idx, outputs = learn.predict(img)
    top3_probs, top3_idxs = torch.topk(outputs, k=3)
    classes = np.array(learn.data.classes)
    top3_classes = list(classes[top3_idxs])
    top3_probs = list(np.array(top3_probs))
    return JSONResponse({'prediction_classes': top3_classes,
                         'prediction_probs': top3_probs})

if __name__== '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level='info')

