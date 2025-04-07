import argparse
import json
import pandas as pd
import torch
from models.link_prediction import SingleEdgeModel


# def recommend(model: torch.nn.Module, user_id: int, model_path: str, channels: int = 64, device: str = 'cpu', n: int = 10):
#     reverse_restaurant_mapping = dict(zip(restaurant_mapping.values(),restaurant_mapping.keys()))
#     reverse_user_mapping = dict(zip(user_mapping.values(),user_mapping.keys()))
    
#     model = SingleEdgeModel(channels)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device(device))) 
#     model.eval()
    
#     row = torch.tensor([user_id] * num_restaurants)
#     col = torch.arange(num_restaurants)
#     edge_label_index = torch.stack([row, col], dim=0)

#     data.to(device)
    
#     pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
#     # pred = pred.clamp(min=0, max=5)
    
#     mapping = {restaurant_id: score for restaurant_id, score in zip(restaurant_mapping.values(), pred.tolist())}
#     sorted_mapping = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[1], reverse=True)}
#     top_n = list(sorted_mapping.keys())[:n]
#     scores = list(sorted_mapping.values())[:n]
#     return [reverse_restaurant_mapping[el] for el in top_n], scores
#     mask = (pred == 5).nonzero(as_tuple=True)
    
#     user_id_reverse = reverse_user_mapping[user_id]
#     predictions = [reverse_restaurant_mapping[el] for el in  mask[0].tolist()[:n]]
#     scores = pred[mask].tolist()[:n]
#     return {'user_id': user_id_reverse, 'recommendations': predictions}, scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, default='output/recommender-yelp-final.pth')
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--graph_data_path", type=str, default="data/base/yelp-hetero.pt")
    parser.add_argument("--metadata_path", type=str, default="data/base/yelp-hetero-meta.json")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    
    data = torch.load(args.graph_data_path)
    
    with open(args.metadata_path, 'r') as f:
        meta_data = json.load(f)
        
    num_users = meta_data['num_users']
    num_restaurants = meta_data['num_restaurants']
    user_mapping = meta_data['user_mapping']
    restaurant_mapping = meta_data['restaurant_mapping']

    reverse_restaurant_mapping = dict(zip(restaurant_mapping.values(),restaurant_mapping.keys()))
    reverse_user_mapping = dict(zip(user_mapping.values(),user_mapping.keys()))
    
    model = SingleEdgeModel(args.channels, data.metadata())
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device))) 
    model.eval()
    
    row = torch.tensor([args.user_id] * num_restaurants)
    col = torch.arange(num_restaurants)
    edge_label_index = torch.stack([row, col], dim=0)

    data.to(args.device)
    
    pred = model(data.x_dict, data.edge_index_dict, edge_label_index)
    # pred = pred.clamp(min=0, max=5)
    
    mapping = {restaurant_id: score for restaurant_id, score in zip(restaurant_mapping.values(), pred.tolist())}
    sorted_mapping = {k: v for k, v in sorted(mapping.items(), key=lambda item: item[1], reverse=True)}
    top_n = list(sorted_mapping.keys())[:args.n]
    scores = list(sorted_mapping.values())[:args.n]
    top_n = [reverse_restaurant_mapping[el] for el in top_n]

    restaurants = pd.read_feather('data/base/yelp_restaurants.feather')
    
    restaurants_sub = restaurants[restaurants.business_id.isin(top_n)]
    restaurants_sub = restaurants_sub[['business_id', 'name', 'city', 'stars', 'review_count', 'categories']]
    restaurants_sub['categories'] = restaurants_sub.categories.str.split(', ')[:5]
    restaurants_sub['Recommendation Score'] = scores

    print(f'Recommendations for user {args.user_id}:')
    print(restaurants_sub)