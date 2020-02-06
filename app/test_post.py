import base64
import requests
import json

url = 'http://localhost:5000/predict'
with open("000001.jpg", "rb") as image_file:
    encoded_img = base64.b64encode(image_file.read())

print(encoded_img)
data = {'file': str(encoded_img)}
r = requests.post(url=url, data=encoded_img)

print(r.json())

