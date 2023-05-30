import os
import json
import pandas

root = os.path.join('data', 'yelp')
filenames = [f for f in os.listdir(root) if f.endswith('.json')]

for filename in filenames:
    print('Processing {}'.format(filename))
    with open(os.path.join(root, filename), 'r', encoding='utf-8') as data_file:
        data = []
        for line in data_file:
            data.append(json.loads(line))
    df = pandas.DataFrame(data)
    df.to_feather(os.path.join(root, filename.replace('.json', '.feather')))
    print('Saved {}'.format(filename.replace('.json', '.feather')))