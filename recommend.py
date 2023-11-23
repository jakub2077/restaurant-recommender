import argparse
from typing import Dict, List
import json
import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero



class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['restaurant'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class RecoModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


def recommend(user_id: int, model_path: str, channels: int = 64, device: str = 'cpu', n: int = 10):
    reverse_restaurant_mapping = dict(zip(restaurant_mapping.values(),restaurant_mapping.keys()))
    reverse_user_mapping = dict(zip(user_mapping.values(),user_mapping.keys()))
    
    model = RecoModel(channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))) 
    model.eval()
    
    row = torch.tensor([user_id] * num_restaurants)
    col = torch.arange(num_restaurants)
    edge_label_index = torch.stack([row, col], dim=0)

    data.to(device)
    
    pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
    # pred = pred.clamp(min=0, max=5)
    
    mapping = {restaurant_id: score for restaurant_id, score in zip(restaurant_mapping.values(), pred.tolist())}
    sorted_mapping = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[1], reverse=True)}
    top_n = list(sorted_mapping.keys())[:n]
    scores = list(sorted_mapping.values())[:n]
    return [reverse_restaurant_mapping[el] for el in top_n], scores
    mask = (pred == 5).nonzero(as_tuple=True)
    
    user_id_reverse = reverse_user_mapping[user_id]
    predictions = [reverse_restaurant_mapping[el] for el in  mask[0].tolist()[:n]]
    scores = pred[mask].tolist()[:n]
    return {'user_id': user_id_reverse, 'recommendations': predictions}, scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, default='output/recommender-yelp-final.pth')
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--graph_data_path", type=str, default="data/yelp-hetero.pt")
    parser.add_argument("--metadata_path", type=str, default="data/yelp-hetero-meta.json")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    
    data = torch.load(args.graph_data_path)
    
    with open('data/yelp-hetero-meta.json', 'r') as f:
        meta_data = json.load(f)
        
    num_users = meta_data['num_users']
    num_restaurants = meta_data['num_restaurants']
    user_mapping = meta_data['user_mapping']
    restaurant_mapping = meta_data['restaurant_mapping']
    
    # recommendations, scores = recommend(args.user_id, args.model_path, args.device, args.n)
    top_n, scores = recommend(args.user_id, args.model_path, args.channels, args.device, args.n)
    
    restaurants = pd.read_feather('data/yelp_restaurants.feather')
    
    restaurants_sub = restaurants[restaurants.business_id.isin(top_n)]
    restaurants_sub = restaurants_sub[['business_id', 'name', 'city', 'stars', 'review_count', 'categories']]
    restaurants_sub['categories'] = restaurants_sub.categories.str.split(', ')[:5]
    restaurants_sub['Recommendation Score'] = scores
    print(f'Recommendations for user {args.user_id}:')
    print(restaurants_sub)