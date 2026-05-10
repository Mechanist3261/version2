import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# =====================================================
# 0. 路径设置
# =====================================================

BASE_DIR = r"D:\大学\code\version2\data_analysis\data_project"

DATA_FILE = os.path.join(BASE_DIR, "community_model_data.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "ablation_map_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABLATION_RESULT_FILE = os.path.join(OUTPUT_DIR, "ablation_feature_group_results.csv")
PRIORITY_TABLE_FILE = os.path.join(OUTPUT_DIR, "district_priority_table.csv")

ABLATION_PLOT_FILE = os.path.join(OUTPUT_DIR, "ablation_macro_f1.png")
MAP_NEED_SCORE_FILE = os.path.join(OUTPUT_DIR, "map_need_score.png")
MAP_NEED_LEVEL_FILE = os.path.join(OUTPUT_DIR, "map_need_level.png")
MAP_MISMATCH_FILE = os.path.join(OUTPUT_DIR, "map_resource_mismatch.png")
QUADRANT_FILE = os.path.join(OUTPUT_DIR, "resource_demand_quadrant.png")
MISMATCH_BAR_FILE = os.path.join(OUTPUT_DIR, "resource_mismatch_bar.png")


# =====================================================
# 1. 中文字体设置
# =====================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================
# 2. 工具函数
# =====================================================

def read_csv_auto(path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件编码：{path}")


def minmax_series(s):
    s = pd.Series(s).astype(float)
    if s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


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
# 3. 读取数据
# =====================================================

df = read_csv_auto(DATA_FILE)

print("=" * 80)
print("读取数据：")
print(df)
print("=" * 80)
print("数据维度：", df.shape)
print("字段：", df.columns.tolist())
print("=" * 80)


# =====================================================
# 4. 字段检查与清洗
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
print("=" * 80)
print("need_level 分布：")
print(df["need_level"].value_counts().sort_index())
print("=" * 80)


# =====================================================
# 5. 构造资源错配指标
# =====================================================

df["医疗资源_norm"] = minmax_series(df["医疗资源"])
df["养老资源_norm"] = minmax_series(df["养老资源"])

df["资源供给指数"] = df[["医疗资源_norm", "养老资源_norm"]].mean(axis=1)

df["资源错配指数"] = df["need_score"] - df["资源供给指数"]

df["资源错配指数_norm"] = minmax_series(df["资源错配指数"])

# 优先级划分
need_high_threshold = df["need_score"].quantile(0.70)
mismatch_high_threshold = df["资源错配指数"].quantile(0.70)

def assign_priority(row):
    if row["need_score"] >= need_high_threshold and row["资源错配指数"] >= mismatch_high_threshold:
        return "Ⅰ类：优先补齐资源"
    elif row["need_score"] >= need_high_threshold:
        return "Ⅱ类：高需求关注"
    elif row["资源错配指数"] >= mismatch_high_threshold:
        return "Ⅲ类：潜在短板区"
    else:
        return "Ⅳ类：常规优化"

df["配置优先级"] = df.apply(assign_priority, axis=1)

priority_cols = [
    "区县",
    "经度",
    "纬度",
    "健康特征",
    "家庭支持",
    "医疗资源",
    "养老资源",
    "资源供给指数",
    "need_score",
    "need_level",
    "资源错配指数",
    "配置优先级"
]

priority_df = df[priority_cols].sort_values(
    by=["资源错配指数", "need_score"],
    ascending=False
).reset_index(drop=True)

priority_df.to_csv(
    PRIORITY_TABLE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("区县资源配置优先级表：")
print(priority_df)
print("优先级表已保存：", PRIORITY_TABLE_FILE)
print("=" * 80)


# =====================================================
# 6. 消融实验设置
# =====================================================

target_col = "need_level"

feature_groups = {
    "完整特征": [
        "经度",
        "纬度",
        "健康特征",
        "家庭支持",
        "医疗资源",
        "养老资源"
    ],
    "去掉地理特征": [
        "健康特征",
        "家庭支持",
        "医疗资源",
        "养老资源"
    ],
    "去掉健康特征": [
        "经度",
        "纬度",
        "家庭支持",
        "医疗资源",
        "养老资源"
    ],
    "去掉家庭支持": [
        "经度",
        "纬度",
        "健康特征",
        "医疗资源",
        "养老资源"
    ],
    "去掉资源特征": [
        "经度",
        "纬度",
        "健康特征",
        "家庭支持"
    ],
    "仅群体需求特征": [
        "健康特征",
        "家庭支持"
    ],
    "仅资源供给特征": [
        "医疗资源",
        "养老资源"
    ],
    "仅地理特征": [
        "经度",
        "纬度"
    ]
}

models = {
    "DecisionTree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(
            max_depth=3,
            random_state=42,
            class_weight="balanced"
        ))
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            random_state=42,
            class_weight="balanced"
        ))
    ])
}

loo = LeaveOneOut()

ablation_rows = []

for group_name, features in feature_groups.items():
    X = df[features].copy()
    y = df[target_col].copy()

    print(f"正在进行消融实验：{group_name}")
    print("使用变量：", features)

    for model_name, model in models.items():
        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=loo
        )

        metrics = evaluate_model(y, y_pred)

        row = {
            "Feature_Group": group_name,
            "Model": model_name,
            "Features": "、".join(features),
            "Num_Features": len(features)
        }

        row.update(metrics)

        ablation_rows.append(row)

        print(model_name, metrics)
        print("-" * 80)


ablation_df = pd.DataFrame(ablation_rows)

ablation_df = ablation_df[
    [
        "Feature_Group",
        "Model",
        "Num_Features",
        "Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "MAE_Level",
        "Features"
    ]
]

ablation_df = ablation_df.sort_values(
    by=["Model", "Macro_F1", "Accuracy"],
    ascending=[True, False, False]
).reset_index(drop=True)

ablation_df.to_csv(
    ABLATION_RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 80)
print("消融实验结果：")
print(ablation_df)
print("消融实验结果已保存：", ABLATION_RESULT_FILE)
print("=" * 80)


# =====================================================
# 7. 绘制消融实验 Macro-F1 图
# =====================================================

plt.figure(figsize=(12, 7))

plot_df = ablation_df.copy()
plot_df["Label"] = plot_df["Model"] + " - " + plot_df["Feature_Group"]
plot_df = plot_df.sort_values("Macro_F1", ascending=True)

plt.barh(plot_df["Label"], plot_df["Macro_F1"])
plt.xlabel("Macro-F1")
plt.ylabel("模型 - 特征组")
plt.title("消融实验：不同特征组的 Macro-F1 对比")
plt.tight_layout()

plt.savefig(ABLATION_PLOT_FILE, dpi=300)
plt.close()

print("消融实验图已保存：", ABLATION_PLOT_FILE)


# =====================================================
# 8. 地图可视化函数
# =====================================================

def plot_district_scatter_map(
    data,
    value_col,
    title,
    output_file,
    size_col=None,
    annotate=True
):
    plt.figure(figsize=(10, 8))

    if size_col is None:
        sizes = np.ones(len(data)) * 180
    else:
        sizes = 120 + minmax_series(data[size_col]) * 600

    scatter = plt.scatter(
        data["经度"],
        data["纬度"],
        c=data[value_col],
        s=sizes,
        alpha=0.85
    )

    if annotate:
        for _, row in data.iterrows():
            plt.text(
                row["经度"] + 0.01,
                row["纬度"] + 0.01,
                row["区县"],
                fontsize=9
            )

    cbar = plt.colorbar(scatter)
    cbar.set_label(value_col)

    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"{title} 已保存：{output_file}")


# =====================================================
# 9. 绘制 need_score 地图
# =====================================================

plot_district_scatter_map(
    data=df,
    value_col="need_score",
    title="杭州市区县公共服务需求强度分布图",
    output_file=MAP_NEED_SCORE_FILE,
    size_col="need_score"
)


# =====================================================
# 10. 绘制 need_level 地图
# =====================================================

plot_district_scatter_map(
    data=df,
    value_col="need_level",
    title="杭州市区县公共服务需求等级分布图",
    output_file=MAP_NEED_LEVEL_FILE,
    size_col="need_level"
)


# =====================================================
# 11. 绘制资源错配地图
# =====================================================

plot_district_scatter_map(
    data=df,
    value_col="资源错配指数",
    title="杭州市区县公共服务需求—资源错配分布图",
    output_file=MAP_MISMATCH_FILE,
    size_col="资源错配指数_norm"
)


# =====================================================
# 12. 绘制需求—资源供给四象限图
# =====================================================

plt.figure(figsize=(10, 8))

x = df["资源供给指数"]
y = df["need_score"]

plt.scatter(
    x,
    y,
    s=160,
    alpha=0.85
)

x_mid = x.median()
y_mid = y.median()

plt.axvline(x_mid, linestyle="--", linewidth=1)
plt.axhline(y_mid, linestyle="--", linewidth=1)

for _, row in df.iterrows():
    plt.text(
        row["资源供给指数"] + 0.01,
        row["need_score"] + 0.01,
        row["区县"],
        fontsize=9
    )

plt.xlabel("资源供给指数")
plt.ylabel("公共服务需求强度 need_score")
plt.title("区县公共服务需求—资源供给四象限图")

plt.text(
    x_mid + 0.02,
    y_mid + 0.02,
    "高需求-高资源",
    fontsize=10
)

plt.text(
    x.min(),
    y_mid + 0.02,
    "高需求-低资源\n优先补齐",
    fontsize=10
)

plt.text(
    x_mid + 0.02,
    y.min(),
    "低需求-高资源",
    fontsize=10
)

plt.text(
    x.min(),
    y.min(),
    "低需求-低资源",
    fontsize=10
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(QUADRANT_FILE, dpi=300)
plt.close()

print("需求—资源供给四象限图已保存：", QUADRANT_FILE)


# =====================================================
# 13. 绘制资源错配指数排序柱状图
# =====================================================

bar_df = df.sort_values("资源错配指数", ascending=True)

plt.figure(figsize=(10, 7))

plt.barh(
    bar_df["区县"],
    bar_df["资源错配指数"]
)

plt.xlabel("资源错配指数")
plt.ylabel("区县")
plt.title("杭州市各区县公共服务资源错配指数排序")
plt.axvline(0, linewidth=1)
plt.tight_layout()

plt.savefig(MISMATCH_BAR_FILE, dpi=300)
plt.close()

print("资源错配指数排序图已保存：", MISMATCH_BAR_FILE)


# =====================================================
# 14. 最终输出说明
# =====================================================

print("=" * 80)
print("全部任务完成！输出文件如下：")
print("1. 消融实验结果：", ABLATION_RESULT_FILE)
print("2. 区县配置优先级表：", PRIORITY_TABLE_FILE)
print("3. 消融实验图：", ABLATION_PLOT_FILE)
print("4. need_score 地图：", MAP_NEED_SCORE_FILE)
print("5. need_level 地图：", MAP_NEED_LEVEL_FILE)
print("6. 资源错配地图：", MAP_MISMATCH_FILE)
print("7. 需求—资源供给四象限图：", QUADRANT_FILE)
print("8. 资源错配指数排序图：", MISMATCH_BAR_FILE)
print("=" * 80)