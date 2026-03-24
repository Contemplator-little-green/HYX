import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC
import torch.optim as optim
from sklearn.metrics import r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ------------------- 模块1：离散Ricci曲率计算（静态，修正版） -------------------
class DiscreteRicciCurvature:
    """
    计算图的Ollivier-Ricci曲率（静态版本，使用正确的Sinkhorn算法）
    对应论文 2.4.2 节公式 (2.2)
    """
    def __init__(self, alpha=0.5, lambda_reg=0.1, sinkhorn_iters=20):
        self.alpha = alpha               # 控制邻居分布的平滑参数
        self.lambda_reg = lambda_reg     # Sinkhorn正则化系数
        self.sinkhorn_iters = sinkhorn_iters  # Sinkhorn迭代次数

    def compute_node_distribution(self, G, node):
        """计算节点的概率分布μ_i（基于拓扑，均匀分布）"""
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            return {node: 1.0}
        # 静态曲率使用均匀分布
        dist = {node: self.alpha}
        for nb in neighbors:
            dist[nb] = (1 - self.alpha) / len(neighbors)
        return dist

    def wasserstein_distance(self, dist_u, dist_v, G):
        """
        使用Sinkhorn算法计算两个分布之间的1-Wasserstein距离
        对应论文 2.5.3 节算法2.2（Sinkhorn迭代）
        """
        nodes_u = list(dist_u.keys())
        nodes_v = list(dist_v.keys())
        m, n = len(nodes_u), len(nodes_v)

        # 构建代价矩阵 C (m x n)，元素为节点间最短路径长度
        C = np.zeros((m, n))
        for i, ni in enumerate(nodes_u):
            for j, nj in enumerate(nodes_v):
                try:
                    C[i, j] = nx.shortest_path_length(G, ni, nj)
                except nx.NetworkXNoPath:
                    C[i, j] = 1e6  # 不可达设为大数

        # 转换为概率向量
        p = np.array([dist_u[ni] for ni in nodes_u])  # [m]
        q = np.array([dist_v[nj] for nj in nodes_v])  # [n]

        # Sinkhorn算法
        K = np.exp(-self.lambda_reg * C)          # 熵正则化核矩阵
        K = np.maximum(K, 1e-12)                  # 避免除零

        u = np.ones(m) / m
        v = np.ones(n) / n

        for _ in range(self.sinkhorn_iters):
            u = p / (K @ v + 1e-12)
            v = q / (K.T @ u + 1e-12)

        # 传输计划 P = diag(u) @ K @ diag(v)
        P = u[:, None] * K * v[None, :]            # 等价于 np.diag(u) @ K @ np.diag(v)
        wass_dist = np.sum(P * C)

        return wass_dist

    def compute_edge_curvatures(self, G):
        """计算所有边的曲率"""
        edge_curvatures = {}
        for u, v in G.edges():
            dist_u = self.compute_node_distribution(G, u)
            dist_v = self.compute_node_distribution(G, v)
            wass_dist = self.wasserstein_distance(dist_u, dist_v, G)
            edge_dist = 1.0  # 假设相邻节点距离为1
            curvature = 1 - wass_dist / edge_dist
            edge_curvatures[(u, v)] = curvature
        return edge_curvatures


# ------------------- 模块2：曲率编码器 -------------------
class CurvatureEncoder(nn.Module):
    """将曲率值编码为可学习的嵌入向量（对应论文 4.3.5 节）"""
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, curvature_values):
        return self.mlp(curvature_values)


# ------------------- 模块3：边特征编码器（针对分子图） -------------------
class EdgeEncoder(nn.Module):
    """将化学键特征编码为嵌入向量（对应论文 4.2.2 节）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Linear(in_channels, out_channels)

    def forward(self, edge_attr):
        return self.encoder(edge_attr)


# ------------------- 模块4：曲率增强的Transformer层 -------------------
class CurvphormerLayer(nn.Module):
    """
    图Transformer层，使用曲率嵌入（+可选边特征）作为注意力偏置
    对应论文 4.5 节公式 (4.6)-(4.7)
    """
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.curvature_bias_proj = nn.Linear(hidden_dim, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, curvature_embeddings):
        """
        Args:
            x: [num_nodes, hidden_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            curvature_embeddings: [num_edges, hidden_dim] 融合了曲率和边特征的嵌入
        """
        # 残差 + LayerNorm
        x_norm = self.norm1(x)
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # 多头投影
        q = self.q_proj(x_norm).reshape(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).reshape(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).reshape(num_nodes, self.num_heads, self.head_dim)

        src = edge_index[0]
        tgt = edge_index[1]

        # 提取源节点和目标节点的q/k
        q_src = q[src]  # [num_edges, num_heads, head_dim]
        k_tgt = k[tgt]  # [num_edges, num_heads, head_dim]

        # 计算点积注意力分数
        attn_scores = (q_src * k_tgt).sum(dim=-1)  # [num_edges, num_heads]
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # 添加曲率偏置
        curvature_bias = self.curvature_bias_proj(curvature_embeddings)  # [num_edges, num_heads]
        attn_scores = attn_scores + curvature_bias

        # 按目标节点归一化（softmax）
        attn_probs = torch.zeros_like(attn_scores)
        for node in torch.unique(tgt):
            mask = tgt == node
            attn_probs[mask] = F.softmax(attn_scores[mask], dim=0)

        # 聚合消息
        v_tgt = v[tgt]  # [num_edges, num_heads, head_dim]
        attn_probs_expanded = attn_probs.unsqueeze(-1)  # [num_edges, num_heads, 1]
        messages = attn_probs_expanded * v_tgt  # [num_edges, num_heads, head_dim]
        messages = messages.reshape(num_edges, -1)  # [num_edges, hidden_dim]

        # 将消息累加到源节点
        out = torch.zeros_like(x)
        out.index_add_(0, src, messages)

        out = self.out_proj(out)
        x = x + self.dropout(out)

        # FFN
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + self.dropout(ffn_out)
        return x


# ------------------- 模块5：动态曲率计算工具 -------------------
def build_neighbor_list(edge_index, num_nodes, include_self=True):
    """
    从edge_index构建每个节点的邻居列表（包括自身可选）
    返回: neighbors [num_nodes] of LongTensor
    """
    device = edge_index.device
    neighbors = [set() for _ in range(num_nodes)]
    # 添加边
    src = edge_index[0].cpu().numpy()
    tgt = edge_index[1].cpu().numpy()
    for u, v in zip(src, tgt):
        neighbors[u].add(v)
        neighbors[v].add(u)  # 无向图
    if include_self:
        for i in range(num_nodes):
            neighbors[i].add(i)
    # 转换为列表排序后返回tensor
    neighbor_tensors = []
    for i in range(num_nodes):
        nb = sorted(list(neighbors[i]))
        neighbor_tensors.append(torch.tensor(nb, dtype=torch.long, device=device))
    return neighbor_tensors


def sinkhorn(mu, nu, C, lambda_reg=0.1, num_iters=20):
    """
    Sinkhorn算法求解熵正则化最优传输
    对应论文 4.3.3 节算法4.2
    mu: [m] 源分布
    nu: [n] 目标分布
    C: [m, n] 代价矩阵
    lambda_reg: 正则化系数
    num_iters: 迭代次数
    返回: 传输计划P [m, n], 近似Wasserstein距离
    """
    m, n = C.shape
    K = torch.exp(-lambda_reg * C)  # [m, n]
    K = torch.clamp(K, min=1e-12)   # 避免除零
    # 初始化
    u = torch.ones(m, device=C.device) / m
    v = torch.ones(n, device=C.device) / n
    for _ in range(num_iters):
        u = mu / (K @ v + 1e-12)
        v = nu / (K.T @ u + 1e-12)
    # 计算传输计划
    P = torch.diag(u) @ K @ torch.diag(v)
    wass_dist = torch.sum(P * C)
    return P, wass_dist


# ------------------- 模块6：完整的Curvphormer模型（支持动态曲率与边特征） -------------------
class Curvphormer(nn.Module):
    """
    DCurvphormer-Mol 模型实现
    对应论文第四章
    """
    def __init__(self,
                 in_channels,
                 hidden_dim=64,
                 out_channels=1,
                 num_layers=6,
                 num_heads=4,
                 dropout=0.1,
                 use_dynamic_curvature=False,
                 edge_in_channels=None,
                 beta=1.0,
                 lambda_reg=0.1,
                 sinkhorn_iters=20,
                 use_sinkhorn=True):   # 消融：是否使用Sinkhorn
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_dynamic_curvature = use_dynamic_curvature
        self.edge_in_channels = edge_in_channels
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.sinkhorn_iters = sinkhorn_iters
        self.use_sinkhorn = use_sinkhorn

        # 节点特征编码
        self.node_encoder = nn.Linear(in_channels, hidden_dim)

        # 曲率编码器
        self.curvature_encoder = CurvatureEncoder(1, hidden_dim)

        # 边特征编码器
        if edge_in_channels is not None:
            self.edge_encoder = EdgeEncoder(edge_in_channels, hidden_dim)
        else:
            self.edge_encoder = None

        # Transformer层
        self.layers = nn.ModuleList([
            CurvphormerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_channels)
        )

    def compute_static_curvatures(self, data):
        """使用修正后的DiscreteRicciCurvature计算静态曲率"""
        G = to_networkx(data, to_undirected=True)
        curvature_calc = DiscreteRicciCurvature(alpha=0.5, lambda_reg=self.lambda_reg, sinkhorn_iters=self.sinkhorn_iters)
        edge_curvatures_dict = curvature_calc.compute_edge_curvatures(G)

        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        curvature_values = torch.zeros(num_edges, 1, device=edge_index.device)
        for i in range(num_edges):
            u = edge_index[0][i].item()
            v = edge_index[1][i].item()
            if (u, v) in edge_curvatures_dict:
                curvature_values[i] = edge_curvatures_dict[(u, v)]
            else:
                curvature_values[i] = edge_curvatures_dict.get((v, u), 0.0)
        return curvature_values

    # ===== 修复后的动态曲率计算 =====
    def compute_dynamic_curvatures(self, h, edge_index, chunk_size=512):
        device = h.device
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)

        # ===== 1. 构建 padded neighbors =====
        nb_mat, mask = build_padded_neighbors(edge_index, num_nodes, device)
        max_deg = nb_mat.size(1)

        curvature_values = torch.zeros(num_edges, 1, device=device)

        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)

            src = edge_index[0, start:end]  # [B]
            tgt = edge_index[1, start:end]

            B = src.size(0)

            # ===== 2. gather neighbors =====
            src_nb = nb_mat[src]  # [B, max_deg]
            tgt_nb = nb_mat[tgt]

            src_mask = mask[src]  # [B, max_deg]
            tgt_mask = mask[tgt]

            # 避免 -1 index
            src_nb_clamped = src_nb.clamp(min=0)
            tgt_nb_clamped = tgt_nb.clamp(min=0)

            h_src_nb = h[src_nb_clamped]  # [B, max_deg, d]
            h_tgt_nb = h[tgt_nb_clamped]

            h_src_center = h[src]  # [B, d]
            h_tgt_center = h[tgt]

            # ===== 3. 相似度分布 =====
            sim_src = (h_src_nb * h_src_center.unsqueeze(1)).sum(-1)  # [B, max_deg]
            sim_tgt = (h_tgt_nb * h_tgt_center.unsqueeze(1)).sum(-1)

            sim_src[~src_mask] = -1e9
            sim_tgt[~tgt_mask] = -1e9

            mu = F.softmax(self.beta * sim_src, dim=1)  # [B, m]
            nu = F.softmax(self.beta * sim_tgt, dim=1)

            # ===== 4. cost matrix =====
            C = torch.cdist(h_src_nb, h_tgt_nb, p=2)  # [B, m, n]

            # mask padding
            mask_src = src_mask.unsqueeze(-1)  # [B, m, 1]
            mask_tgt = tgt_mask.unsqueeze(1)  # [B, 1, n]
            valid_mask = mask_src & mask_tgt
            C = C.masked_fill(~valid_mask, 1e6)

            # ===== 5. batched sinkhorn =====
            wass = batched_sinkhorn(
                mu, nu, C,
                lambda_reg=self.lambda_reg,
                num_iters=self.sinkhorn_iters
            )  # [B]

            # ===== 6. 边长度 =====
            d_ij = torch.norm(h_src_center - h_tgt_center, dim=1).clamp(min=1e-6)

            curvature = 1 - wass / d_ij
            curvature_values[start:end, 0] = curvature

        return self.curvature_encoder(curvature_values)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 编码节点特征
        h = self.node_encoder(x.float())

        # 获取边特征（如果有）
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        # 如果边特征是一维，扩展为 [num_edges, 1] 以匹配线性层
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # [num_edges] -> [num_edges, 1]

        # 初始化曲率嵌入
        num_edges = edge_index.size(1)
        if not self.use_dynamic_curvature:
            # 静态曲率：预先计算一次
            curv_values = self.compute_static_curvatures(data).to(x.device)
            curvature_embeddings = self.curvature_encoder(curv_values)
        else:
            # 动态曲率：第一层先用零初始化
            curvature_embeddings = torch.zeros(num_edges, self.hidden_dim, device=x.device)

        # 如果存在边特征，将其编码（静态，不随层变化）
        if self.edge_encoder is not None and edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr.float())  # [num_edges, hidden_dim]
        else:
            edge_emb = None

        # 逐层传播（每层开始前更新曲率）
        for i, layer in enumerate(self.layers):
            # 如果是动态曲率，在每层开始前基于当前h更新曲率嵌入
            if self.use_dynamic_curvature:
                curvature_embeddings = self.compute_dynamic_curvatures(h, edge_index)

            # 融合边特征（静态）
            if edge_emb is not None:
                curvature_embeddings = curvature_embeddings + edge_emb

            h = layer(h, edge_index, curvature_embeddings)

        # 图读出：按batch取平均
        num_graphs = batch.max().item() + 1
        graph_out = torch.zeros(num_graphs, self.hidden_dim, device=h.device)
        graph_out.index_add_(0, batch, h)
        counts = torch.bincount(batch).float().unsqueeze(1)
        graph_out = graph_out / counts

        # 输出预测
        out = self.output_layer(graph_out)
        return out

# ------------------- 训练/评估函数（支持多指标） -------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # 强制标准输出无缓冲
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    
    # 创建tqdm进度条
    pbar = tqdm(
        loader, 
        desc="Training", 
        file=sys.stderr,
        mininterval=3.0,
        maxinterval=15.0,
        dynamic_ncols=True
    )

    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        all_preds.append(out.detach().cpu())
        all_targets.append(data.y.cpu())
        
        # 实时更新进度条显示当前loss
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / sum([t.size(0) for t in all_targets]):.4f}'
        })
    
    # 计算指标
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = total_loss / len(loader.dataset)
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
    r2 = r2_score(all_targets.numpy(), all_preds.numpy())
    
    return mse, mae, rmse, r2

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # 验证集也可以用tqdm
    pbar = tqdm(loader, desc="Validating", leave=False)
    
    for data in pbar:
        data = data.to(device)
        out = model(data).squeeze()
        loss = criterion(out, data.y)
        
        total_loss += loss.item() * data.num_graphs
        all_preds.append(out.cpu())
        all_targets.append(data.y.cpu())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = total_loss / len(loader.dataset)
    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
    r2 = r2_score(all_targets.numpy(), all_preds.numpy())
    
    return mse, mae, rmse, r2


# ------------------- gpt 新增padding等 -------------------

def build_padded_neighbors(edge_index, num_nodes, device):
    neighbors = [[] for _ in range(num_nodes)]
    src = edge_index[0].cpu().numpy()
    tgt = edge_index[1].cpu().numpy()

    for u, v in zip(src, tgt):
        neighbors[u].append(v)
        neighbors[v].append(u)

    for i in range(num_nodes):
        neighbors[i].append(i)  # include self

    max_deg = max(len(nb) for nb in neighbors)

    nb_mat = torch.full((num_nodes, max_deg), -1, dtype=torch.long)
    mask = torch.zeros((num_nodes, max_deg), dtype=torch.bool)

    for i, nb in enumerate(neighbors):
        nb = torch.tensor(nb, dtype=torch.long)
        nb_mat[i, :len(nb)] = nb
        mask[i, :len(nb)] = 1

    return nb_mat.to(device), mask.to(device)


def batched_sinkhorn(mu, nu, C, lambda_reg=0.1, num_iters=10):
    """
    mu: [B, m]
    nu: [B, n]
    C:  [B, m, n]
    """
    K = torch.exp(-lambda_reg * C).clamp(min=1e-12)

    B, m, n = C.shape
    u = torch.ones(B, m, device=C.device) / m
    v = torch.ones(B, n, device=C.device) / n

    for _ in range(num_iters):
        u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-12)
        v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-12)

    P = u.unsqueeze(-1) * K * v.unsqueeze(1)
    wass = torch.sum(P * C, dim=(1, 2))  # [B]

    return wass



# ------------------- 主程序 -------------------
if __name__ == "__main__":
    print("this_is_a_outcome_provided_from_jyj_1st")
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 使用绝对路径并确保是原始字符串
    root_path = r"./ZINC"

    train_dataset = ZINC(root=root_path, subset=True, split='train')
    val_dataset = ZINC(root=root_path, subset=True, split='val')
    test_dataset = ZINC(root=root_path, subset=True, split='test')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 获取输入维度
    sample_data = train_dataset[0]
    in_channels = sample_data.x.size(1)  # ZINC中为28

    # 安全地获取边特征维度
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
    use_edge = True             # 是否使用化学键特征
    beta = 1.0                  # 温度参数 (论文公式4.1)
    use_sinkhorn = True         # 是否使用Sinkhorn
    lambda_reg = 0.05            # Sinkhorn正则化系数
    sinkhorn_iters = 10         # Sinkhorn迭代次数
    # =============================================

    # 初始化模型
    model = Curvphormer(
        in_channels=in_channels,
        hidden_dim=128,
        out_channels=1,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_dynamic_curvature=use_dynamic,
        edge_in_channels=edge_in_channels if use_edge else None,
        beta=beta,
        lambda_reg=lambda_reg,
        sinkhorn_iters=sinkhorn_iters,
        use_sinkhorn=use_sinkhorn
    ).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # 训练循环
    best_val_mse = float('inf')
    print("Starting training...")
    # 添加epoch级别的进度条
    for epoch in tqdm(range(1, 51), desc="Epochs"):
        train_mse, train_mae, train_rmse, train_r2 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mse, val_mae, val_rmse, val_r2 = eval_epoch(model, val_loader, criterion, device)

        print(f'Epoch {epoch:03d} | '
              f'Train MSE: {train_mse:.4f} MAE: {train_mae:.4f} RMSE: {train_rmse:.4f} R2: {train_r2:.4f} | '
              f'Val MSE: {val_mse:.4f} MAE: {val_mae:.4f} RMSE: {val_rmse:.4f} R2: {val_r2:.4f}')

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
    print(f"use_sinkhorn: {use_sinkhorn}")
    print(f"lambda_reg: {lambda_reg}")
    print(f"sinkhorn_iters: {sinkhorn_iters}")