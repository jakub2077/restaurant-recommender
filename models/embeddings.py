from torch import nn
from  torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData


class GNN(nn.Module):
    """
    The `GNN` class is a PyTorch module that implements a Graph Neural Network (GNN) with 
    three `SAGEConv` Graph Convolutional Network layers. The GNN takes node features `x` and edge indices `edge_index` 
    as input, and applies a series of GCN layers to produce the final node embeddings.
    
    The `__init__` method initializes the GCN layers and a dropout layer. The `forward` method applies the GCN layers, 
    with ReLU activation and dropout between the layers, to produce the final node embeddings.
    """
    def __init__(self, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class Classifier(nn.Module):
    """
    The `Classifier` module applies a dot-product between the source and destination node embeddings to 
    derive edge-level predictions.
    
    The `forward` method takes the node embeddings for users and restaurants, as well as the edge label indices, 
    as input. It first extracts the node embeddings for the source and destination nodes of the supervision edges. 
    Then, it applies a dot-product between these embeddings to obtain the final edge-level predictions.
    """
    def forward(self, x_user, x_restaurant, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_restaurant = x_restaurant[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        embeds = (edge_feat_user * edge_feat_restaurant).sum(dim=-1)
        return embeds

class EmbeddingModel(nn.Module):
    """
    The `EmbeddingModel` class is a PyTorch module that learns node embeddings for a heterogeneous graph data. 
    It consists of the following components:
    
    - `user_lin` and `restaurant_lin`: Linear layers that project the input features of users and restaurants to 
    the hidden channel size.
    - `user_emb` and `restaurant_emb`: Embedding layers that learn node embeddings for users and restaurants.
    - `gnn`: A Graph Neural Network (GNN) module that aggregates and propagates node features through the graph.
    - `classifier`: A module that applies a dot-product between source and destination node embeddings to 
    derive edge-level predictions.
    
    The `forward` method takes a `HeteroData` object as input, which contains the graph structure and node features. 
    It first projects the input features using the linear and embedding layers, then passes them through the GNN module 
    to obtain the final node embeddings. Finally, it uses the `classifier` module to derive edge-level predictions.
    """
    def __init__(self, hidden_channels: int, data: HeteroData, dropout: float = 0.5):
        super().__init__()
        self.user_input_size = 5
        self.restaurant_input_size = 640
        self.hidden_channels = hidden_channels

        self.user_lin = nn.Linear(self.user_input_size, hidden_channels)
        self.restaurant_lin = nn.Linear(self.restaurant_input_size, hidden_channels)

        # self.user_lin = nn.Linear(5, hidden_channels)
        # self.restaurant_lin = nn.Linear(640, hidden_channels)

        self.user_emb = nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.restaurant_emb = nn.Embedding(data["restaurant"].num_nodes, hidden_channels)

        self.gnn = GNN(hidden_channels, hidden_channels, dropout=dropout)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        
        self.classifier = Classifier()

    def forward(self, data: HeteroData):
        x_dict = {
          "user": self.user_lin(data['user'].x) + self.user_emb(data["user"].node_id),
          "restaurant": self.restaurant_lin(data['restaurant'].x) + self.restaurant_emb(data["restaurant"].node_id),
        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["restaurant"],
            data["user", "rating", "restaurant"].edge_label_index,
        )
        return pred

if __name__ == "__main__":
    pass