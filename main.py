import json
import argparse
from pathlib import Path
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.retrieval import (
    retrieval_normalized_dcg, retrieval_recall, retrieval_precision, retrieval_average_precision
    )
from torchmetrics.functional.classification import multiclass_recall
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric


def gnn_model_summary(model: torch.nn.Module):
    """
    Prints summary of GNN model
    """
    model_params_list = list(model.named_parameters())
    table = PrettyTable()
    table.field_names = [f"Layer.Parameter", "Param Tensor Shape", "Param #"]
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        table.add_row([p_name, str(p_shape), str(p_count)])
    print(table)
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)


class GNNEncoder(torch.nn.Module):
    """
    The `GNNEncoder` class is a PyTorch module that implements a graph neural network (GNN) encoder. 
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


class EdgeDecoder(torch.nn.Module):
    '''
    The `EdgeDecoder` class is a PyTorch module that implements a decoder for edge prediction in 
    a graph neural network (GNN) model. It takes in node embeddings `z_dict` and edge label indices `edge_label_index` 
    as input, and applies a series of linear layers to produce a scalar edge prediction value.
    The forward method first concatenates the node embeddings for the source and target nodes of each edge, 
    then passes this through three linear layers with ReLU activations. The final linear layer produces 
    a single scalar output for each edge, which can be used as the predicted edge label or score.
    
    This `EdgeDecoder` module is typically used as part of a larger GNN model, where the node embeddings produced by 
    an encoder module (e.g. `GNNEncoder`) are passed to the `EdgeDecoder` to produce edge predictions for 
    a downstream task such as link prediction or recommendation.
    '''
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['restaurant'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z).relu()
        z = self.lin3(z)
        return z.view(-1)

    
class Model(torch.nn.Module):
    """
    The `Model` class is a PyTorch module that combines a GNN encoder and an edge decoder to perform link prediction 
    or recommendation tasks on graph-structured data.
    
    The `__init__` method initializes the encoder and decoder modules. The encoder is a `GNNEncoder` module 
    that takes in node features `x_dict` and edge indices `edge_index_dict`, and produces node embeddings `z_dict`. 
    The decoder is an `EdgeDecoder` module that takes the node embeddings and edge label indices `edge_label_index`, 
    and produces scalar edge prediction values.
    
    The `forward` method applies the encoder to the input data to obtain the node embeddings, 
    and then passes these embeddings to the decoder to produce the final edge predictions.
    
    This `Model` class can be used as part of a larger graph neural network pipeline, 
    where the encoder learns node representations and the decoder predicts edges or edge labels for downstream tasks 
    such as link prediction or recommendation.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum') # TODO: data.metadata() method is outside of this class
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
    

def weighted_rmse_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor=None):
    """
    Calulates weighted RMSE loss.
    """
    weight = 1. if weight is None else weight[target.int()]
    return (weight * (pred - target).pow(2)).mean().sqrt()


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_data: torch_geometric.data.Data):
    """
    Train the model
    """
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict, 
                 train_data['user', 'restaurant'].edge_label_index)
    target = train_data['user', 'restaurant'].edge_label
    rmse = F.mse_loss(pred, target).sqrt()
    wrmse = weighted_rmse_loss(pred, target, weight)
    wrmse.backward()
    optimizer.step()
    return model, float(wrmse), float(rmse)


@torch.no_grad()
def test(model : torch.nn.Module, data_test: torch_geometric.data.Data):
    model.eval()
    pred = model(data_test.x_dict, data_test.edge_index_dict, 
                 data_test['user', 'restaurant'].edge_label_index)
    pred = pred.clamp(min=1, max=5)
    target = data_test['user', 'restaurant'].edge_label
    rmse = F.mse_loss(pred, target).sqrt()
    wrmse = weighted_rmse_loss(pred, target, weight)
    return float(wrmse), float(rmse),


def plot_rank_hist(pred : torch.Tensor, target: torch.Tensor):
    '''
    Plot the histogram of the true and predicted rankings
    '''
    fig = plt.figure()
    plt.hist(target.cpu(), bins=30, facecolor="green", alpha=0.5)
    plt.hist(pred.cpu(), bins=30, facecolor="blue", alpha=0.5)
    plt.xlabel("ranking")
    plt.ylabel("count")
    plt.legend(("True", "Predicted"))
    fig.suptitle("Rank prediction on test data")
    return fig

@torch.no_grad()
def run_metrics(model: torch.nn.Module, data_test: torch_geometric.data.Data, k: int = 10):
    model.eval()
    pred = model(data_test.x_dict, data_test.edge_index_dict,
                 data_test['user', 'restaurant'].edge_label_index)
    # TODO: check if prediction should be clamped to [1, 5]
    pred = pred.clamp(min=1, max=5)
    target = data_test['user', 'restaurant'].edge_label
 
    ndgc = retrieval_normalized_dcg(pred, target, top_k=k)
    recall = multiclass_recall(pred, target, num_classes=6)
    recall_at_k = retrieval_recall(pred, target > target.mean(), top_k=k)
    prec = retrieval_precision(pred, target > target.mean(), top_k=k)
    ap = retrieval_average_precision(pred, target > target.mean(), top_k=k)
    return float(ndgc), float(recall), float(recall_at_k), float(prec), float(ap), plot_rank_hist(pred, target)

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='data/base/yelp-hetero.pt')
    parser.add_argument('--meta_data_path', type=str, default='data/base/yelp-hetero-meta.json')
    args = parser.parse_args()

    # Set up TensorrBoard
    # layout = {
    #     "Metrics": {
    #         "wRMSE": ["Multiline", ["train", "validation"]],
    #         "rMSE": ["Multiline", ["train", "validation"]],
    #     },
    # }
    hparams = {
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'hidden_channels': args.hidden_channels
    }
    writer = SummaryWriter()
    writer.add_hparams(hparams, {'hparam/ loss': 0})  # Initial loss is set to 0

    # writer.add_custom_scalars(layout)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    # Load the data
    print("Loading data...")
    data = torch.load(Path(args.data_path))
    # load meta data
    with open(Path(args.meta_data_path), 'r') as f:
        meta_data = json.load(f)
    num_users = meta_data['num_users']
    num_restaurants = meta_data['num_restaurants']
    user_mapping = meta_data['user_mapping']
    restaurant_mapping = meta_data['restaurant_mapping']

    # Transform the data
    print("Transforming data...")
    # TODO: needs more work, testing , different variables etc.
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=0,
        add_negative_train_samples=False,
        edge_types=("user", "rating", "restaurant"),
        rev_edge_types=("restaurant", "rev_rating", "user"), 
    )
    train_data, val_data, test_data = transform(data)

    # Create the model
    print("Creating model...")
    model = Model(args.hidden_channels).to(device)
    
    # Due to lazy initialization, we need to run one model step so the number of parameters can be inferred:
    val_data.to(device)
    with torch.no_grad():
        model.encoder(val_data.x_dict, val_data.edge_index_dict)

    # TODO: try different optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Print model summary
    gnn_model_summary(model)

    # Caluculate the weight for the weighted RMSE loss function
    weight = torch.bincount(data['user', 'rating', 'restaurant'].edge_label.int())
    weight = weight.max() / weight
    weight = weight.to(device)

    # Train the model
    print("Training model...")
    losses = []
    n_epochs = args.epochs
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    for epoch in range(1, n_epochs + 1):
        model, wrsme, rmse = train(model, optimizer, train_data)
        val_wrmse, val_rmse = test(model, val_data)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train wRMSE: {wrsme:.4f}, Train rMSE: {rmse:.4f} '
                  f'Val wRMSE: {val_wrmse:.4f}, Val rMSE: {val_rmse:.4f}')
        writer.add_scalars('wRMSE', {'train': wrsme, 'validation': val_wrmse}, epoch)
        writer.add_scalars('rMSE', {'train': rmse, 'validation': val_rmse}, epoch)
        # losses.append([wrsme, rmse, val_wrmse, val_rmse])

    # Run metrics
    print("Running metrics...")
    k = 20   
    test_data.to(device)
    ndgc, recall, recall_at_k, prec, ap , hist = run_metrics(model, test_data, k)
    # TODO: find better way to log these metrics
    writer.add_text("NDGC@{k}", f"{ndgc:.4f}", global_step=0)
    writer.add_text("Recall", f"{recall:.4f}", global_step=0)
    writer.add_text("Recall@{k}", f"{recall_at_k:.4f}", global_step=0)
    writer.add_text("Precision@{k}", f"{prec:.4f}", global_step=0)
    writer.add_text("AP@{k}", f"{ap:.4f}", global_step=0)
    writer.add_figure("Rank histogram", hist, global_step=0)

    writer.flush()
    writer.close()
    print('Done!')