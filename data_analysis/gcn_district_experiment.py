import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# =====================================================
# 0. 路径设置
# =====================================================

BASE_DIR = r"D:\大学\code\version2\data_analysis\data_project"

DATA_FILE = os.path.join(BASE_DIR, "community_model_data.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "gcn_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_TABLE_FILE = os.path.join(OUTPUT_DIR, "gcn_model_comparison.csv")
PREDICTION_FILE = os.path.join(OUTPUT_DIR, "gcn_predictions.csv")
GRAPH_PLOT_FILE = os.path.join(OUTPUT_DIR, "district_graphs.png")
CONFUSION_MATRIX_FILE = os.path.join(OUTPUT_DIR, "gcn_confusion_matrices.png")
BARPLOT_FILE = os.path.join(OUTPUT_DIR, "gcn_model_performance.png")


# =====================================================
# 1. 随机种子
# =====================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# =====================================================
# 2. 读取数据
# =====================================================

def read_csv_auto(path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件编码：{path}")


df = read_csv_auto(DATA_FILE)

print("=" * 80)
print("读取区县建模数据：")
print(df)
print("=" * 80)
print("数据维度：", df.shape)
print("字段：", df.columns.tolist())
print("=" * 80)


# =====================================================
# 3. 字段检查
# =====================================================

required_cols = [
    "区县",
    "经度",
    "纬度",
    "健康特征",
    "家庭支持",
    "医疗资源",
    "养老资源",
    "need_score",
    "need_level"
]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"缺少必要字段：{missing_cols}")

numeric_cols = [
    "经度",
    "纬度",
    "健康特征",
    "家庭支持",
    "医疗资源",
    "养老资源",
    "need_score",
    "need_level"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_cols).copy()
df["need_level"] = df["need_level"].astype(int)

df = df.reset_index(drop=True)

print("清洗后数据维度：", df.shape)
print("缺失值统计：")
print(df.isna().sum())
print("need_level 分布：")
print(df["need_level"].value_counts().sort_index())
print("=" * 80)


# =====================================================
# 4. 特征与标签设置
# =====================================================

"""
注意：
不能把 need_score 放进输入特征。
因为 need_level 是由 need_score 分箱得到的。
"""

feature_cols = [
    "经度",
    "纬度",
    "健康特征",
    "家庭支持",
    "医疗资源",
    "养老资源"
]

graph_feature_cols = [
    "健康特征",
    "家庭支持",
    "医疗资源",
    "养老资源"
]

target_col = "need_level"

district_names = df["区县"].values
X_raw = df[feature_cols].values.astype(float)
X_graph_raw = df[graph_feature_cols].values.astype(float)
y = df[target_col].values.astype(int)

num_nodes = len(df)
num_features = X_raw.shape[1]
num_classes = int(y.max() + 1)

print("节点数量：", num_nodes)
print("输入特征数量：", num_features)
print("类别数量：", num_classes)
print("=" * 80)


# =====================================================
# 5. 距离与图构建函数
# =====================================================

def haversine_distance_matrix(lon, lat):
    """
    根据经纬度计算球面距离，单位：公里。
    """
    lon = np.radians(lon)
    lat = np.radians(lat)

    n = len(lon)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dlon = lon[j] - lon[i]
            dlat = lat[j] - lat[i]

            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat[i]) * np.cos(lat[j]) * np.sin(dlon / 2) ** 2
            )

            c = 2 * np.arcsin(np.sqrt(a))
            dist[i, j] = 6371 * c

    return dist


def euclidean_distance_matrix(X):
    """
    计算欧氏距离矩阵。
    """
    n = X.shape[0]
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(X[i] - X[j])

    return dist


def build_knn_weighted_graph_from_distance(dist_matrix, k=3):
    """
    根据距离矩阵构建 KNN 加权图。
    距离越近，边权越大。
    """
    n = dist_matrix.shape[0]
    A = np.zeros((n, n))

    nonzero_dist = dist_matrix[dist_matrix > 0]

    if len(nonzero_dist) == 0:
        sigma = 1.0
    else:
        sigma = np.median(nonzero_dist)

    if sigma == 0:
        sigma = 1.0

    for i in range(n):
        idx = np.argsort(dist_matrix[i])

        neighbors = [j for j in idx if j != i][:k]

        for j in neighbors:
            weight = np.exp(-dist_matrix[i, j] / sigma)
            A[i, j] = weight

    # 无向化
    A = np.maximum(A, A.T)

    return A


def normalize_adjacency(A):
    """
    GCN 标准归一化：
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    """
    A = A.copy()
    n = A.shape[0]

    A = A + np.eye(n)

    degree = np.sum(A, axis=1)

    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0

    D_inv_sqrt = np.diag(degree_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm


def build_geo_graph(df, k=3):
    lon = df["经度"].values.astype(float)
    lat = df["纬度"].values.astype(float)

    dist_geo = haversine_distance_matrix(lon, lat)

    A_geo = build_knn_weighted_graph_from_distance(
        dist_geo,
        k=k
    )

    return A_geo, dist_geo


def build_feature_graph(X_graph_scaled, k=3):
    dist_feat = euclidean_distance_matrix(X_graph_scaled)

    A_feat = build_knn_weighted_graph_from_distance(
        dist_feat,
        k=k
    )

    return A_feat, dist_feat


def build_hybrid_graph(A_geo, A_feat, alpha=0.5):
    """
    混合图：
    alpha 越大，越偏向地理图；
    alpha 越小，越偏向特征图。
    """
    A_hybrid = alpha * A_geo + (1 - alpha) * A_feat

    # 保证无向
    A_hybrid = np.maximum(A_hybrid, A_hybrid.T)

    return A_hybrid


# =====================================================
# 6. GCN 模型定义
# =====================================================

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A_norm):
        AX = torch.matmul(A_norm, X)
        out = self.linear(AX)
        return out


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_classes, dropout=0.2):
        super(GCN, self).__init__()

        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, out_classes)
        self.dropout = dropout

    def forward(self, X, A_norm):
        h = self.gcn1(X, A_norm)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.gcn2(h, A_norm)
        return out


class MLP(nn.Module):
    """
    不使用图结构的神经网络对照模型。
    """
    def __init__(self, in_features, hidden_features, out_classes, dropout=0.2):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_classes)
        self.dropout = dropout

    def forward(self, X):
        h = self.fc1(X)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.fc2(h)
        return out


# =====================================================
# 7. 训练函数
# =====================================================

def get_class_weights(y_train, num_classes):
    """
    计算类别权重，缓解小样本类别不均衡。
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(float)

    weights = np.zeros(num_classes)

    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = len(y_train) / (num_classes * counts[i])
        else:
            weights[i] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


def train_gcn_one_fold(
    X_all,
    y_all,
    A_norm,
    train_idx,
    test_idx,
    hidden_dim=8,
    lr=0.01,
    weight_decay=5e-4,
    epochs=400,
    seed=42
):
    set_seed(seed)

    X_tensor = torch.tensor(X_all, dtype=torch.float32)
    y_tensor = torch.tensor(y_all, dtype=torch.long)
    A_tensor = torch.tensor(A_norm, dtype=torch.float32)

    train_idx_tensor = torch.tensor(train_idx, dtype=torch.long)
    test_idx_tensor = torch.tensor(test_idx, dtype=torch.long)

    model = GCN(
        in_features=X_all.shape[1],
        hidden_features=hidden_dim,
        out_classes=num_classes,
        dropout=0.2
    )

    class_weights = get_class_weights(y_all[train_idx], num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_loss = np.inf
    best_state = None
    patience = 80
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        logits = model(X_tensor, A_tensor)

        loss = criterion(
            logits[train_idx_tensor],
            y_tensor[train_idx_tensor]
        )

        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        logits = model(X_tensor, A_tensor)
        pred = torch.argmax(logits[test_idx_tensor], dim=1).cpu().numpy()

    return pred[0]


def train_mlp_one_fold(
    X_train,
    y_train,
    X_test,
    hidden_dim=8,
    lr=0.01,
    weight_decay=5e-4,
    epochs=400,
    seed=42
):
    set_seed(seed)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model = MLP(
        in_features=X_train.shape[1],
        hidden_features=hidden_dim,
        out_classes=num_classes,
        dropout=0.2
    )

    class_weights = get_class_weights(y_train, num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_loss = np.inf
    best_state = None
    patience = 80
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        logits = model(X_train_tensor)

        loss = criterion(logits, y_train_tensor)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        logits = model(X_test_tensor)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    return pred[0]


# =====================================================
# 8. 评价函数
# =====================================================

def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "Macro_Recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "Macro_F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "MAE_Level": mean_absolute_error(y_true, y_pred)
    }


# =====================================================
# 9. 绘制图结构
# =====================================================

def plot_single_graph(ax, A, title):
    lon = df["经度"].values.astype(float)
    lat = df["纬度"].values.astype(float)

    ax.scatter(lon, lat, s=120)

    n = len(lon)

    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                ax.plot(
                    [lon[i], lon[j]],
                    [lat[i], lat[j]],
                    linewidth=0.8,
                    alpha=0.5
                )

    for i, name in enumerate(district_names):
        ax.text(
            lon[i],
            lat[i],
            name,
            fontsize=8
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


# =====================================================
# 10. LOOCV 实验主循环
# =====================================================

loo = LeaveOneOut()

model_names = [
    "MLP-NoGraph",
    "GCN-Geo",
    "GCN-Feature",
    "GCN-Hybrid"
]

predictions = {
    name: []
    for name in model_names
}

true_labels = []
test_districts = []

# 固定地理图。地理图只依赖经纬度，不涉及标签。
A_geo_fixed, dist_geo = build_geo_graph(df, k=3)

print("地理图邻接矩阵：")
print(np.round(A_geo_fixed, 3))
print("=" * 80)


for fold, (train_idx, test_idx) in enumerate(loo.split(X_raw), start=1):
    print(f"正在进行 LOOCV 第 {fold}/{num_nodes} 折，测试区县：{district_names[test_idx][0]}")

    # -------------------------------------------------
    # 10.1 特征标准化
    # 注意：scaler 只在训练节点上 fit，避免数据泄露
    # -------------------------------------------------

    scaler_x = StandardScaler()
    scaler_x.fit(X_raw[train_idx])

    X_scaled = scaler_x.transform(X_raw)

    scaler_graph = StandardScaler()
    scaler_graph.fit(X_graph_raw[train_idx])

    X_graph_scaled = scaler_graph.transform(X_graph_raw)

    # -------------------------------------------------
    # 10.2 构建特征图和混合图
    # 特征图只使用 X_graph_scaled，不使用标签
    # -------------------------------------------------

    A_feat, dist_feat = build_feature_graph(
        X_graph_scaled,
        k=3
    )

    A_hybrid = build_hybrid_graph(
        A_geo_fixed,
        A_feat,
        alpha=0.5
    )

    A_geo_norm = normalize_adjacency(A_geo_fixed)
    A_feat_norm = normalize_adjacency(A_feat)
    A_hybrid_norm = normalize_adjacency(A_hybrid)

    # -------------------------------------------------
    # 10.3 MLP 对照
    # -------------------------------------------------

    mlp_pred = train_mlp_one_fold(
        X_train=X_scaled[train_idx],
        y_train=y[train_idx],
        X_test=X_scaled[test_idx],
        hidden_dim=8,
        lr=0.01,
        weight_decay=5e-4,
        epochs=400,
        seed=42
    )

    predictions["MLP-NoGraph"].append(mlp_pred)

    # -------------------------------------------------
    # 10.4 GCN-Geo
    # -------------------------------------------------

    pred_geo = train_gcn_one_fold(
        X_all=X_scaled,
        y_all=y,
        A_norm=A_geo_norm,
        train_idx=train_idx,
        test_idx=test_idx,
        hidden_dim=8,
        lr=0.01,
        weight_decay=5e-4,
        epochs=400,
        seed=42
    )

    predictions["GCN-Geo"].append(pred_geo)

    # -------------------------------------------------
    # 10.5 GCN-Feature
    # -------------------------------------------------

    pred_feat = train_gcn_one_fold(
        X_all=X_scaled,
        y_all=y,
        A_norm=A_feat_norm,
        train_idx=train_idx,
        test_idx=test_idx,
        hidden_dim=8,
        lr=0.01,
        weight_decay=5e-4,
        epochs=400,
        seed=42
    )

    predictions["GCN-Feature"].append(pred_feat)

    # -------------------------------------------------
    # 10.6 GCN-Hybrid
    # -------------------------------------------------

    pred_hybrid = train_gcn_one_fold(
        X_all=X_scaled,
        y_all=y,
        A_norm=A_hybrid_norm,
        train_idx=train_idx,
        test_idx=test_idx,
        hidden_dim=8,
        lr=0.01,
        weight_decay=5e-4,
        epochs=400,
        seed=42
    )

    predictions["GCN-Hybrid"].append(pred_hybrid)

    true_labels.append(y[test_idx][0])
    test_districts.append(district_names[test_idx][0])


print("=" * 80)
print("LOOCV 训练完成")
print("=" * 80)


# =====================================================
# 11. 汇总结果
# =====================================================

result_rows = []
prediction_rows = []

for model_name in model_names:
    y_pred = np.array(predictions[model_name])
    y_true = np.array(true_labels)

    metrics = evaluate_model(y_true, y_pred)
    metrics["Model"] = model_name
    result_rows.append(metrics)

    for district, true_y, pred_y in zip(test_districts, y_true, y_pred):
        prediction_rows.append({
            "Model": model_name,
            "区县": district,
            "真实_need_level": int(true_y),
            "预测_need_level": int(pred_y),
            "预测是否正确": int(true_y == pred_y),
            "等级误差": int(abs(true_y - pred_y))
        })

    print(model_name)
    print(metrics)
    print(classification_report(
        y_true,
        y_pred,
        zero_division=0
    ))
    print("-" * 80)


result_df = pd.DataFrame(result_rows)

result_df = result_df[
    [
        "Model",
        "Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "MAE_Level"
    ]
]

result_df = result_df.sort_values(
    by=["Macro_F1", "Accuracy"],
    ascending=False
).reset_index(drop=True)

prediction_df = pd.DataFrame(prediction_rows)

result_df.to_csv(
    RESULT_TABLE_FILE,
    index=False,
    encoding="utf-8-sig"
)

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("GCN 模型对比结果：")
print(result_df)
print("模型对比结果已保存：", RESULT_TABLE_FILE)
print("逐区县预测结果已保存：", PREDICTION_FILE)
print("=" * 80)


# =====================================================
# 12. 绘制最终图结构
# =====================================================

# 用全样本标准化后的特征构建最终展示图
scaler_all_graph = StandardScaler()
X_graph_all_scaled = scaler_all_graph.fit_transform(X_graph_raw)

A_feat_all, _ = build_feature_graph(
    X_graph_all_scaled,
    k=3
)

A_hybrid_all = build_hybrid_graph(
    A_geo_fixed,
    A_feat_all,
    alpha=0.5
)

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(18, 5)
)

plot_single_graph(axes[0], A_geo_fixed, "Geo Graph")
plot_single_graph(axes[1], A_feat_all, "Feature Similarity Graph")
plot_single_graph(axes[2], A_hybrid_all, "Hybrid Graph")

plt.tight_layout()
plt.savefig(GRAPH_PLOT_FILE, dpi=300)
plt.close()

print("区县图结构图已保存：", GRAPH_PLOT_FILE)


# =====================================================
# 13. 绘制模型表现柱状图
# =====================================================

plt.figure(figsize=(10, 6))

plot_df = result_df.sort_values("Macro_F1", ascending=True)

plt.barh(plot_df["Model"], plot_df["Macro_F1"])
plt.xlabel("Macro-F1")
plt.ylabel("Model")
plt.title("GCN Graph Structure Comparison Based on LOOCV")
plt.tight_layout()

plt.savefig(BARPLOT_FILE, dpi=300)
plt.close()

print("GCN 模型表现图已保存：", BARPLOT_FILE)


# =====================================================
# 14. 绘制混淆矩阵
# =====================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(12, 10)
)

axes = axes.flatten()

labels = sorted(np.unique(y))

for ax, model_name in zip(axes, model_names):
    pred_df = prediction_df[prediction_df["Model"] == model_name]

    cm = confusion_matrix(
        pred_df["真实_need_level"],
        pred_df["预测_need_level"],
        labels=labels
    )

    ax.imshow(cm)

    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

fig.suptitle("Confusion Matrices of GCN Models", fontsize=16)
plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_FILE, dpi=300)
plt.close()

print("GCN 混淆矩阵图已保存：", CONFUSION_MATRIX_FILE)


# =====================================================
# 15. 输出最佳模型
# =====================================================

best_model = result_df.iloc[0]["Model"]

print("=" * 80)
print("最佳 GCN 实验模型：", best_model)
print("=" * 80)

best_pred = prediction_df[prediction_df["Model"] == best_model]
print(best_pred)

print("=" * 80)
print("所有输出文件：")
print("1. GCN 模型对比表：", RESULT_TABLE_FILE)
print("2. GCN 逐区县预测：", PREDICTION_FILE)
print("3. 区县图结构图：", GRAPH_PLOT_FILE)
print("4. GCN 模型表现图：", BARPLOT_FILE)
print("5. GCN 混淆矩阵图：", CONFUSION_MATRIX_FILE)
print("=" * 80)