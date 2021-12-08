import json, re

def openJSON(path):
    res = None
    with open(path, 'r') as f:
        res = json.load(f)
    return res

def saveJSON(path, dataset):
    with open('json/{}.json'.format(path), 'w') as f:
        json.dump(dataset, f, indent=2)
    return True