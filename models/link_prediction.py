import torch
from torch_geometric.nn import SAGEConv, to_hetero


class SingleEdgeEncoder(torch.nn.Module):
    """
    The `SingleEdgeEncoder` class is a PyTorch module that implements a graph neural network (GNN) encoder. 
    It takes in node features `x` and edge indices `edge_index` as input, and applies three successive 
    graph convolutional layers to produce a final node embedding.
    
    The first two graph convolutional layers use the `SAGEConv` operator from the PyTorch Geometric library, 
    with a hidden channel size of `hidden_channels`. The final layer also uses `SAGEConv` 
    but with an output channel size of `out_channels`.
    
    Each graph convolutional layer is followed by a ReLU activation function.
    
    This encoder can be used as part of a larger graph neural network model, 
    where the node embeddings produced by this encoder can be used for downstream tasks such as node classification, 
    link prediction, or graph pooling.
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


class SingleEdgeDecoder(torch.nn.Module):
    '''
    The `SingleEdgeDecoder` class is a PyTorch module that implements a decoder for edge prediction in 
    a graph neural network (GNN) model. It takes in node embeddings `z_dict` and edge label indices `edge_label_index` 
    as input, and applies a series of linear layers to produce a scalar edge prediction value.
    The forward method first concatenates the node embeddings for the source and target nodes of each edge, 
    then passes this through three linear layers with ReLU activations. The final linear layer produces 
    a single scalar output for each edge, which can be used as the predicted edge label or score.
    
    This `SingleEdgeDecoder` module is typically used as part of a larger GNN model, where the node embeddings produced by 
    an encoder module (e.g. `GNNEncoder`) are passed to the `SingleEdgeDecoder` to produce edge predictions for 
    a downstream task such as link prediction or recommendation.
    '''
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['restaurant'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z).relu()
        z = self.lin3(z)
        return z.view(-1)

    
class SingleEdgeModel(torch.nn.Module):
    """
    The `SingleEdgeModel` class is a PyTorch module that combines a GNN encoder and an edge decoder to perform link prediction 
    or recommendation tasks on graph-structured data.
    
    The `__init__` method initializes the encoder and decoder modules. The encoder is a `GNNEncoder` module 
    that takes in node features `x_dict` and edge indices `edge_index_dict`, and produces node embeddings `z_dict`. 
    The decoder is an `EdgeDecoder` module that takes the node embeddings and edge label indices `edge_label_index`, 
    and produces scalar edge prediction values.
    
    The `forward` method applies the encoder to the input data to obtain the node embeddings, 
    and then passes these embeddings to the decoder to produce the final edge predictions.
    
    This `SingleEdgeModel` class can be used as part of a larger graph neural network pipeline, 
    where the encoder learns node representations and the decoder predicts edges or edge labels for downstream tasks 
    such as link prediction or recommendation.
    """
    def __init__(self, hidden_channels, data_metadata: dict):
        super().__init__()
        self.encoder = SingleEdgeEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data_metadata, aggr='sum')
        self.decoder = SingleEdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


if __name__ == "__main__":
    pass