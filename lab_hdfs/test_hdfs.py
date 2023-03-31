from json import dump, dumps

# https://hdfscli.readthedocs.io/en/latest/quickstart.html 
from hdfs import InsecureClient
client = InsecureClient('http://localhost:9870', user='hadoop')

client.content('//')
client.list('/')
client.status('//')

with client.read('tracking/#LATEST') as reader:
    data = reader.read()

client.makedirs('/data')

# client.write('samples')
records = [
    {'name': 'foo', 'weight': 1},
    {'name': 'bar', 'weight': 2},
]

import urllib3
try:
    client.write('/data/records.jsonl', data=dumps(records), encoding='utf-8')
# except urllib3.exceptions.New as e:
except urllib3.exceptions.HTTPError as e:
    print(e)
    ex = e
    print(e.pool.host,
    e.pool.port)
