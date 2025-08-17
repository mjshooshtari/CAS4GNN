import sys
import types
import torch


class _Data:
    def __init__(self, n):
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.x = None
        self.y = None

    def to(self, device):
        return self

def _from_networkx(G):
    return _Data(len(G))

class _GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.lin(x)

tg_utils = types.ModuleType("torch_geometric.utils")
tg_utils.from_networkx = _from_networkx
tg_nn = types.ModuleType("torch_geometric.nn")
tg_nn.GCNConv = _GCNConv
tg_mod = types.ModuleType("torch_geometric")

sys.modules.setdefault("torch_geometric", tg_mod)
sys.modules.setdefault("torch_geometric.utils", tg_utils)
sys.modules.setdefault("torch_geometric.nn", tg_nn)
