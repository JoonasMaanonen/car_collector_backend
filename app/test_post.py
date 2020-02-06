import base64
import requests

url = 'http://localhost:5000/predict'
with open("000001.jpg", "rb") as image_file:
    encoded_img = base64.b64encode(image_file.read())

data = {'file': encoded_img}
r = requests.post(url=url, data=data)

print(r.json())

