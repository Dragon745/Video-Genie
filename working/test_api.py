import urllib.request
import json

url = 'https://huggingface.co/api/models?author=ai4bharat&search=indicxlit'
try:
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode('utf-8'))
    for m in data: 
        print(m['id'])
except Exception as e:
    print('Error:', e)
