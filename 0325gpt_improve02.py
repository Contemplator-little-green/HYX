import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ZINC
from torch_geometric.utils import softmax, to_networkx
from torch_scatter import scatter
import torch.optim as optim
from sklearn.metrics import r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import networkx as nx
import sys

# ------------------- 模块1：离散Ricci曲率计算（静态） -------------------
class DiscreteRicciCurvature:
    def __init__(self, alpha=0.5, lambda_reg=0.1, sinkhorn_iters=20):
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.sinkhorn_iters = sinkhorn_iters

    def compute_node_distribution(self, G, node):
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            return {node: 1.0}
        dist = {node: self.alpha}
        for nb in neighbors:
            dist[nb] = (1 - self.alpha) / len(neighbors)
        return dist

    def wasserstein_distance(self, dist_u, dist_v, G):
        nodes_u, nodes_v = list(dist_u.keys()), list(dist_v.keys())
        m, n = len(nodes_u), len(nodes_v)
        C = np.zeros((m, n))
        for i, ni in enumerate(nodes_u):
            for j, nj in enumerate(nodes_v):
                try:
                    C[i,j] = nx.shortest_path_length(G, ni, nj)
                except nx.NetworkXNoPath:
                    C[i,j] = 1e6
        p = np.array([dist_u[ni] for ni in nodes_u])
        q = np.array([dist_v[nj] for nj in nodes_v])
        K = np.exp(-self.lambda_reg * C)
        K = np.maximum(K, 1e-12)
        u = np.ones(m)/m
        v = np.ones(n)/n
        for _ in range(self.sinkhorn_iters):
            u = p / (K @ v + 1e-12)
            v = q / (K.T @ u + 1e-12)
        P = u[:,None] * K * v[None,:]
        return np.sum(P*C)

    def compute_edge_curvatures(self, G):
        edge_curvatures = {}
        for u,v in G.edges():
            dist_u = self.compute_node_distribution(G,u)
            dist_v = self.compute_node_distribution(G,v)
            wass_dist = self.wasserstein_distance(dist_u, dist_v, G)
            edge_curvatures[(u,v)] = 1 - wass_dist / 1.0
        return edge_curvatures

# ------------------- 模块2：曲率编码器 -------------------
class CurvatureEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    def forward(self, curvature_values):
        return self.mlp(curvature_values)

# ------------------- 模块3：边特征编码器 -------------------
class EdgeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Linear(in_channels, out_channels)
    def forward(self, edge_attr):
        return self.encoder(edge_attr)

# ------------------- 模块4：Curvphormer Layer -------------------
class CurvphormerLayer(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.curvature_bias_proj = nn.Linear(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, curvature_embeddings):
        x_norm = self.norm1(x)
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        q = self.q_proj(x_norm).reshape(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).reshape(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).reshape(num_nodes, self.num_heads, self.head_dim)

        src, tgt = edge_index[0], edge_index[1]
        q_src, k_tgt = q[src], k[tgt]
        attn_scores = (q_src * k_tgt).sum(dim=-1) / (self.head_dim ** 0.5)
        curvature_bias = self.curvature_bias_proj(curvature_embeddings)
        attn_scores = attn_scores + curvature_bias

        attn_probs = torch.zeros_like(attn_scores)
        for node in torch.unique(tgt):
            mask = tgt==node
            attn_probs[mask] = F.softmax(attn_scores[mask], dim=0)

        v_tgt = v[tgt]
        messages = attn_probs.unsqueeze(-1) * v_tgt
        messages = messages.reshape(num_edges, -1)

        out = torch.zeros_like(x)
        out.index_add_(0, src, messages)
        out = self.out_proj(out)
        x = x + self.dropout(out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x

# ------------------- 模块5：padding邻居 -------------------
def build_padded_neighbors(edge_index, num_nodes, device):
    neighbors = [[] for _ in range(num_nodes)]
    src = edge_index[0].cpu().numpy()
    tgt = edge_index[1].cpu().numpy()
    for u,v in zip(src,tgt):
        neighbors[u].append(v)
        neighbors[v].append(u)
    for i in range(num_nodes):
        neighbors[i].append(i)
    max_deg = max(len(nb) for nb in neighbors)
    nb_mat = torch.full((num_nodes, max_deg), -1, dtype=torch.long)
    mask = torch.zeros((num_nodes, max_deg), dtype=torch.bool)
    for i, nb in enumerate(neighbors):
        nb = torch.tensor(nb,dtype=torch.long)
        nb_mat[i,:len(nb)] = nb
        mask[i,:len(nb)] = 1
    return nb_mat.to(device), mask.to(device)

# ------------------- 模块6：Curvphormer模型 -------------------
class Curvphormer(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, out_channels=1, num_layers=6, num_heads=4,
                 dropout=0.1, use_dynamic_curvature=False, edge_in_channels=None, beta=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_dynamic_curvature = use_dynamic_curvature
        self.beta = beta
        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.curvature_encoder = CurvatureEncoder(1, hidden_dim)
        if edge_in_channels is not None:
            self.edge_encoder = EdgeEncoder(edge_in_channels, hidden_dim)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.edge_encoder = None
        self.layers = nn.ModuleList([CurvphormerLayer(hidden_dim, num_heads, dropout)
                                     for _ in range(num_layers)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, out_channels)
        )

    def compute_static_curvatures(self, data):
        G = to_networkx(data, to_undirected=True)
        calc = DiscreteRicciCurvature()
        edge_curv = calc.compute_edge_curvatures(G)
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        curv_values = torch.zeros(num_edges,1,device=edge_index.device)
        for i in range(num_edges):
            u,v = edge_index[0][i].item(), edge_index[1][i].item()
            curv_values[i] = edge_curv.get((u,v), edge_curv.get((v,u),0.0))
        return curv_values

    def compute_dynamic_curvatures(self, h, edge_index, layer_idx):
        src, tgt = edge_index[0], edge_index[1]
        h_src, h_tgt = h[src], h[tgt]
        sim = (h_src * h_tgt).sum(dim=-1)
        current_beta = self.beta * (layer_idx+1)/self.num_layers
        alpha = softmax(sim*current_beta, tgt)
        dist = torch.norm(h_src - h_tgt, dim=-1).clamp(min=1e-6)
        agg_dist = scatter(alpha * dist, tgt, dim=0, reduce='sum')
        wass_approx = agg_dist[tgt]
        curvature = 1 - wass_approx / dist
        return self.curvature_encoder(curvature.unsqueeze(-1))

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        batch = getattr(data,'batch',torch.zeros(x.size(0),dtype=torch.long,device=x.device))
        h = self.node_encoder(x)

        edge_attr = getattr(data,'edge_attr',None)
        if edge_attr is not None and edge_attr.dim()==1:
            edge_attr = edge_attr.unsqueeze(-1)
        num_edges = edge_index.size(1)
        if not self.use_dynamic_curvature:
            curv_values = self.compute_static_curvatures(data).to(x.device)
            curvature_embeddings = self.curvature_encoder(curv_values)
        else:
            curvature_embeddings = torch.zeros(num_edges, self.hidden_dim, device=x.device)

        if self.edge_encoder is not None and edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = None

        for i,layer in enumerate(self.layers):
            if self.use_dynamic_curvature:
                curvature_embeddings = self.compute_dynamic_curvatures(h, edge_index, i)
            if edge_emb is not None:
                curvature_embeddings = self.fusion_mlp(torch.cat([curvature_embeddings, edge_emb],dim=-1))
            h = layer(h, edge_index, curvature_embeddings)

        num_graphs = batch.max().item()+1
        graph_out = torch.zeros(num_graphs, self.hidden_dim, device=h.device)
        graph_out.index_add_(0,batch,h)
        counts = torch.bincount(batch).float().unsqueeze(1)
        graph_out = graph_out / counts
        return self.output_layer(graph_out)

# ------------------- 训练/评估 -------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_targets = [],[]
    pbar = tqdm(loader, desc="Training", file=sys.stderr,
                mininterval=3.0, maxinterval=15.0, dynamic_ncols=True)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = criterion(out,data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*data.num_graphs
        all_preds.append(out.detach().cpu())
        all_targets.append(data.y.cpu())
        pbar.set_postfix({'loss':f'{loss.item():.4f}',
                          'avg_loss':f'{total_loss/sum([t.size(0) for t in all_targets]):.4f}'})
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = total_loss/len(loader.dataset)
    mae = torch.mean(torch.abs(all_preds-all_targets)).item()
    rmse = torch.sqrt(torch.mean((all_preds-all_targets)**2)).item()
    r2 = r2_score(all_targets.numpy(),all_preds.numpy())
    return mse, mae, rmse, r2

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [],[]
    pbar = tqdm(loader, desc="Validating", leave=False, mininterval=1.0)
    for data in pbar:
        data = data.to(device)
        out = model(data).squeeze()
        loss = criterion(out,data.y)
        total_loss += loss.item()*data.num_graphs
        all_preds.append(out.cpu())
        all_targets.append(data.y.cpu())
        pbar.set_postfix({'loss':f'{loss.item():.4f}'})
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = total_loss/len(loader.dataset)
    mae = torch.mean(torch.abs(all_preds-all_targets)).item()
    rmse = torch.sqrt(torch.mean((all_preds-all_targets)**2)).item()
    r2 = r2_score(all_targets.numpy(),all_preds.numpy())
    return mse, mae, rmse, r2

# ------------------- 主程序 -------------------
if __name__ == "__main__":
    print("this_is_a_outcome_provided_0325gpt_improve02")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集路径（ZINC）
    root_path = r"./ZINC"

    # 加载 ZINC 数据集
    train_dataset = ZINC(root=root_path, subset=True, split='train')
    val_dataset = ZINC(root=root_path, subset=True, split='val')
    test_dataset = ZINC(root=root_path, subset=True, split='test')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 获取输入节点特征维度
    sample_data = train_dataset[0]
    in_channels = sample_data.x.size(1)

    # 获取边特征维度（安全处理1D情况）
    if hasattr(sample_data, 'edge_attr') and sample_data.edge_attr is not None:
        if sample_data.edge_attr.dim() == 2:
            edge_in_channels = sample_data.edge_attr.size(1)
        elif sample_data.edge_attr.dim() == 1:
            edge_in_channels = 1
            print("Warning: edge_attr is 1D, treating as scalar feature.")
        else:
            edge_in_channels = None
            print("Warning: edge_attr has unexpected shape, ignoring.")
    else:
        edge_in_channels = None

    # ================== 配置选项 ==================
    use_dynamic = True        # True: 动态曲率, False: 静态曲率
    use_edge = True           # 是否使用边特征
    beta = 1.0                # 温度参数 β
    # =============================================

    # 初始化模型（方法一：去掉 lambda_reg / sinkhorn_iters / use_sinkhorn）
    model = Curvphormer(
        in_channels=in_channels,
        hidden_dim=128,
        out_channels=1,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_dynamic_curvature=use_dynamic,
        edge_in_channels=edge_in_channels if use_edge else None,
        beta=beta
    ).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # 训练循环
    best_val_mse = float('inf')
    print("Starting training...")
    for epoch in tqdm(range(1, 51), desc="Epochs"):
        train_mse, train_mae, train_rmse, train_r2 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae, val_rmse, val_r2 = eval_epoch(model, val_loader, criterion, device)

        print(f'Epoch {epoch:03d} | '
              f'Train MSE: {train_mse:.4f} MAE: {train_mae:.4f} RMSE: {train_rmse:.4f} R2: {train_r2:.4f} | '
              f'Val MSE: {val_mse:.4f} MAE: {val_mae:.4f} RMSE: {val_rmse:.4f} R2: {val_r2:.4f}')

        # 保存最佳模型
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), 'best_model_jyj.pth')
            print(f'  -> Best model saved (val MSE: {val_mse:.4f})')

    # 加载最佳模型并在测试集上评估
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_model_jyj.pth'))
    test_mse, test_mae, test_rmse, test_r2 = eval_epoch(model, test_loader, criterion, device)
    print(f'Test MSE: {test_mse:.4f} MAE: {test_mae:.4f} RMSE: {test_rmse:.4f} R2: {test_r2:.4f}')

    # 打印当前配置
    print("\n--- Configuration ---")
    print(f"use_dynamic_curvature: {use_dynamic}")
    print(f"use_edge_features: {use_edge}")
    print(f"beta: {beta}")