import base64
import requests
import json

url = 'http://localhost:5000/predict'
with open("000001.jpg", "rb") as image_file:
    encoded_img = base64.b64encode(image_file.read())

data = {'file': str(encoded_img)}



data_bytes = 'MySecret'.encode("utf-8")
encoded = base64.b64encode(data_bytes)
print(f"Encoded: {encoded}")
r = requests.post(url=url, data=encoded_img, headers={'Authorization': encoded})

print(r.json())

