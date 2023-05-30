import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn as nn


# Load the Yelp businesses and reviews datasets
businesses_df = pd.read_feather(os.path.join('data', 'yelp', 'yelp_academic_dataset_business.feather'),)
print('buinesses_df loaded')

reviews_df = pd.read_feather(os.path.join('data', 'yelp', 'yelp_academic_dataset_review.feather'))
print('reviews_df loaded')

# Filter businesses by category (e.g., restaurants)
print('Filter businesses by category (e.g., restaurants)')
businesses_df = businesses_df.where(businesses_df['categories'].str.contains('Restaurants', na=False)).dropna()

# Extract unique restaurant IDs and create a dictionary mapping them to indices
print('Extract unique restaurant IDs and create a dictionary mapping them to indices')
restaurant_ids = businesses_df['business_id'].unique().tolist()
num_restaurants = len(restaurant_ids)

# Encode the restaurant IDs 
restaurant_indices = dict(zip(restaurant_ids, range(num_restaurants)))
restaurant_indices_inv = {v: k for k, v in restaurant_indices.items()}

####### MODIFIED ###########
reviews_df = reviews_df[['review_id', 'user_id', 'business_id', 'stars', 'text']]
top20k = reviews_df['user_id'].value_counts()[:20000].index.tolist()
reviews_subdf = reviews_df[reviews_df['user_id'].isin(top20k)]
businesses_subdf = businesses_df[businesses_df['business_id'].isin(top20k)]

# Convert categories to string type
businesses_subdf['categories'] = businesses_subdf['categories'].astype(str)

# Encode the categorical features using label encoding
label_encoder = LabelEncoder()
businesses_subdf['categories'] = label_encoder.fit_transform(businesses_subdf['categories'])


# Convert the node_features array to a suitable data type
node_features = businesses_subdf[['categories', 'review_count', 'stars']].values.astype(np.float32)
node_features = torch.FloatTensor(node_features)
reviews_subdf = reviews_subdf.merge(businesses_subdf[['business_id']], on='business_id')
restaurant_ratings = reviews_subdf.groupby(['business_id', 'user_id'])['stars'].mean().reset_index()

# reviews_df = reviews_df.merge(businesses_df[['business_id']], on='business_id')
# restaurant_ratings = reviews_df.groupby(['business_id', 'user_id'])['stars'].mean().reset_index()

# Create a sparse tensor representing the restaurant ratings

restaurant_ratings = restaurant_ratings.dropna(subset=['business_id', 'user_id'])
user_ratings_df = restaurant_ratings.pivot(index='business_id', columns='user_id', values='stars').fillna(0)
user_ratings_df = user_ratings_df.rename(index=restaurant_indices_inv)
user_ratings = torch.FloatTensor(user_ratings_df.values)

# Create adjacency matrix and node features
adjacency_matrix = torch.zeros((num_restaurants, num_restaurants))
for _, row in businesses_subdf.iterrows():
    i = restaurant_indices[row['business_id']]
    j = restaurant_indices[row['business_id']]
    adjacency_matrix[i, j] = 1
    adjacency_matrix[j, i] = 1

node_features = businesses_subdf[['categories', 'review_count', 'stars']].values
node_features = torch.FloatTensor(node_features)

# Define the GNN model
class YelpGNN(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(YelpGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

# Create the YelpGNN model instance
hidden_channels = 64
model = YelpGNN(node_features.shape[1], hidden_channels)

# Set up the data for the YelpGNN model
data = Data(x=node_features, edge_index=adjacency_matrix.nonzero().t().long())

# Make a forward pass with the YelpGNN model
output = model(data.x, data.edge_index)

# Define the model, optimizer, and loss function
model = YelpGNN(num_restaurants, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Train the model
for epoch in range(1):
    model.train()
    optimizer.zero_grad()
    output = model(node_features, adjacency_matrix.nonzero().t())
    loss = criterion(output)
    
    # Add regularization terms to the loss function to encourage diversity in the recommendations
    user_embeddings = model.restaurants_conv2(model.restaurants_conv1(node_features[:num_restaurants], adjacency_matrix.nonzero().t()))
    item_embeddings = node_features[:num_restaurants]
    user_embeddings = user_embeddings / torch.norm(user_embeddings, dim=1, keepdim=True)
    item_embeddings = item_embeddings
    item_embeddings / torch.norm(item_embeddings, dim=1, keepdim=True)
    diversity_loss = -torch.mean(torch.matmul(user_embeddings, item_embeddings.t()))
    loss += 0.01 * diversity_loss

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch {:03d}, Loss: {:.4f}".format(epoch, loss.item()))
