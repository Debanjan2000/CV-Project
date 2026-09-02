import json
import os

with open('data/Valid.json', 'r') as f:
    data = json.load(f)

print(list(data['images'])[:2])
