# Car collector Backend

## How to run

Install Docker and run the following commands:

```
docker build -t car-collector . 
docker run --rm -it -p 5000:5000 car-collector
```

Now the app should be running in localhost:5000

## Used technologies
- [Starlette](https://www.starlette.io/) for asynchronous REST API.
- [Fastai](https://github.com/fastai/fastai) for the deep learning.
- [Docker](https://www.docker.com/) for deployment.
- [Render](https://render.com/) for hosting.
