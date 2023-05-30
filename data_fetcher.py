import os
import json
import requests


# set up API key and endpoint
path_to_api_file = os.path.join('../google-maps-api.txt')

with open(path_to_api_file, 'r', encoding='utf-8') as f:
    api_key = f.read()
    
endpoint = "https://maps.googleapis.com/maps/api/place/textsearch/json"

# set up search parameters
params = {
    "query": "restaurants near Kraków center",
    "key": api_key
}

# make the search request
response = requests.get(endpoint, params=params, timeout=10000)
data = json.loads(response.text)

with open(os.path.join('data', 'google_maps', 'restaurants.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f)

# get the place IDs for each restaurant
place_ids = []
for result in data["results"]:
    place_ids.append(result["place_id"])

# retrieve the details for each restaurant and print the user reviews
for place_id in place_ids:
    details_params = {
        "place_id": place_id,
        "key": api_key
    }
    details_response = requests.get("https://maps.googleapis.com/maps/api/place/details/json", params=details_params, timeout=10000)
    details_data = json.loads(details_response.text)
    with open(os.path.join('data', 'google_maps', f'restaurant_{place_id}.json'), 'w', encoding='utf-8') as jf:
        json.dump(details_data, jf)
    # reviews = details_data["result"]["reviews"]
    # for review in reviews:
    #     print(review["text"])
