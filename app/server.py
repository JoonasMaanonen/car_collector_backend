import glob
import os
import aiohttp
import asyncio
import uvicorn
import base64
import binascii
import logging
from random import randint
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.authentication import (
    AuthenticationBackend, AuthenticationError, SimpleUser, UnauthenticatedUser,
    AuthCredentials, requires
)

from fastai.vision import *

# READ env variables
SECRET_KEY = os.getenv(key='SECRET_KEY')
MODEL_FILE_URL = os.getenv(key='MODEL_FILE_URL')
BRAND_FILE_URL = os.getenv(key='BRAND_FILE_URL')
MODEL_FILE_NAME = os.getenv(key='MODEL_FILE_NAME')
BRAND_FILE_NAME = os.getenv(key='BRAND_FILE_NAME')

class BasicAuthBackend(AuthenticationBackend):
    async def authenticate(self, request):
        if "Authorization" not in request.headers:
            return
        access_key = request.headers["Authorization"]
        decoded = base64.b64decode(access_key).decode("ascii")
        if decoded == SECRET_KEY:
            return AuthCredentials(["authenticated"]), SimpleUser('DummyUser')
        return

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

async def save_image(img_data, learn, model):
    img_bytes = base64.b64decode(img_data)
    img = open_image(BytesIO(img_bytes))
    transformed_img, pred_class, pred_idx, outputs = learn.predict(img, return_x=True)
    transformed_img.save(f'app/static/{model}_{randint(1,1000)}.jpg')

async def get_prediction(img_data, learn):
    img_bytes = base64.b64decode(img_data)
    img = open_image(BytesIO(img_bytes))
    pred_class, pred_idx, outputs = learn.predict(img)
    top3_probs, top3_idxs = torch.topk(outputs, k=3)
    classes = np.array(learn.data.classes)
    top3_classes = list(classes[top3_idxs])
    top3_probs = [str(x) for x in list(np.array(top3_probs))]
    return top3_classes, top3_probs

middleware = [
        Middleware(AuthenticationMiddleware, backend=BasicAuthBackend())
        ]

path = Path(__file__).parent
app = Starlette(on_startup=[setup_learners], middleware=middleware)
app.mount('/app/static', StaticFiles(directory='app/static'))

@app.route('/')
async def homepage(request):
    if request.user.is_authenticated:
        return HTMLResponse('<h1>Welcome to the awesome car api!</h1>')
    else:
        return HTMLResponse('<h1>Remember to authenticate to use the API!</h1>')

@app.route('/debug')
@requires('authenticated')
async def debug(request):
    image_files = glob.glob('app/static/*.jpg')
    models = [image.split('/')[-1].split('_')[0] for image in image_files]
    response_string = "".join([f'<img src="{image}"> <br> <p>{models[i]}</p> <br>'
                              for i, image in enumerate(image_files)])
    return HTMLResponse(response_string)

@app.route('/predict', methods=['POST'])
@requires('authenticated')
async def predict(request):
    img_data = await request.body()
    top3_brand_classes, top3_brand_probs = await get_prediction(img_data, brand_learner)
    top3_model_classes, top3_model_probs = await get_prediction(img_data, model_learner)
    await save_image(img_data, model_learner, top3_brand_classes[0])
    return JSONResponse({'brand_classes': top3_brand_classes,
			 'brand_probs': top3_brand_probs,
			 'model_classes': top3_model_classes,
			 'model_probs': top3_model_probs})

if __name__== '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level='info')

