import os
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

OUTPUT_DIR = os.path.join(BASE_DIR, "baseline_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_TABLE_FILE = os.path.join(OUTPUT_DIR, "baseline_model_comparison.csv")
PREDICTION_FILE = os.path.join(OUTPUT_DIR, "baseline_predictions.csv")
CONFUSION_MATRIX_FILE = os.path.join(OUTPUT_DIR, "baseline_confusion_matrices.png")
BARPLOT_FILE = os.path.join(OUTPUT_DIR, "baseline_model_performance.png")


# =====================================================
# 1. 读取数据
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
print("原始数据：")
print(df)
print("=" * 80)
print("数据维度：", df.shape)
print("字段：", df.columns.tolist())
print("=" * 80)
print("缺失值统计：")
print(df.isna().sum())
print("=" * 80)


# =====================================================
# 2. 字段检查
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

# 确保数值字段为 numeric
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

# 删除极少数异常缺失行
df = df.dropna(subset=required_cols).copy()

df["need_level"] = df["need_level"].astype(int)

print("清洗后数据维度：", df.shape)
print("need_level 分布：")
print(df["need_level"].value_counts().sort_index())
print("=" * 80)


# =====================================================
# 3. 构造 X 和 y
# =====================================================

"""
注意：
不能把 need_score 放进 X。
因为 need_level 是由 need_score 分箱得到的。
如果把 need_score 放进 X，就是数据泄露。
"""

feature_cols = [
    "经度",
    "纬度",
    "健康特征",
    "家庭支持",
    "医疗资源",
    "养老资源"
]

target_col = "need_level"

X = df[feature_cols].copy()
y = df[target_col].copy()

district_names = df["区县"].copy()

print("输入特征 X：")
print(X.head())
print("=" * 80)
print("标签 y：")
print(y.values)
print("=" * 80)


# =====================================================
# 4. 设置 Leave-One-Out 交叉验证
# =====================================================

loo = LeaveOneOut()

print("交叉验证方法：Leave-One-Out Cross Validation")
print("样本数量：", len(df))
print("训练次数：", len(df))
print("=" * 80)


# =====================================================
# 5. 设置 baseline 模型
# =====================================================

models = {
    "Dummy-MostFrequent": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DummyClassifier(strategy="most_frequent"))
    ]),

    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=3,
            weights="distance"
        ))
    ]),

    "SVM-RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            class_weight="balanced",
            C=1.0,
            gamma="scale"
        ))
    ]),

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


# =====================================================
# 6. 模型评估函数
# =====================================================

def evaluate_model(y_true, y_pred):
    """
    返回分类指标。
    由于 need_level 是 0-4 的有序等级，
    除 Accuracy / F1 之外，也计算 MAE。
    MAE 越小，说明等级预测偏差越小。
    """
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
# 7. 运行 Leave-One-Out baseline 实验
# =====================================================

result_rows = []
prediction_dfs = {}

for model_name, model in models.items():
    print(f"正在训练并评估模型：{model_name}")

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=loo
    )

    metrics = evaluate_model(y, y_pred)
    metrics["Model"] = model_name

    result_rows.append(metrics)

    pred_df = pd.DataFrame({
        "区县": district_names.values,
        "真实_need_level": y.values,
        "预测_need_level": y_pred,
        "预测是否正确": (y.values == y_pred).astype(int),
        "等级误差": np.abs(y.values - y_pred)
    })

    prediction_dfs[model_name] = pred_df

    print(model_name, metrics)
    print(classification_report(
        y,
        y_pred,
        zero_division=0
    ))
    print("-" * 80)


# =====================================================
# 8. 输出模型对比表
# =====================================================

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

result_df.to_csv(
    RESULT_TABLE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 80)
print("baseline 模型对比结果：")
print(result_df)
print("结果表已保存：", RESULT_TABLE_FILE)
print("=" * 80)


# =====================================================
# 9. 输出各模型预测结果
# =====================================================

all_pred_rows = []

for model_name, pred_df in prediction_dfs.items():
    tmp = pred_df.copy()
    tmp.insert(0, "Model", model_name)
    all_pred_rows.append(tmp)

all_predictions = pd.concat(all_pred_rows, ignore_index=True)

all_predictions.to_csv(
    PREDICTION_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("各区县预测结果已保存：", PREDICTION_FILE)
print("=" * 80)


# =====================================================
# 10. 绘制模型表现柱状图
# =====================================================

plt.figure(figsize=(10, 6))

plot_df = result_df.sort_values("Macro_F1", ascending=True)

plt.barh(plot_df["Model"], plot_df["Macro_F1"])
plt.xlabel("Macro-F1")
plt.ylabel("Model")
plt.title("Baseline Model Comparison Based on LOOCV")
plt.tight_layout()

plt.savefig(BARPLOT_FILE, dpi=300)
plt.close()

print("模型表现柱状图已保存：", BARPLOT_FILE)


# =====================================================
# 11. 绘制混淆矩阵
# =====================================================

num_models = len(models)

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 9)
)

axes = axes.flatten()

labels = sorted(y.unique())

for ax, (model_name, pred_df) in zip(axes, prediction_dfs.items()):
    cm = confusion_matrix(
        pred_df["真实_need_level"],
        pred_df["预测_need_level"],
        labels=labels
    )

    im = ax.imshow(cm)

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

# 删除空白子图
for idx in range(len(prediction_dfs), len(axes)):
    fig.delaxes(axes[idx])

fig.suptitle("Confusion Matrices of Baseline Models", fontsize=16)
plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_FILE, dpi=300)
plt.close()

print("混淆矩阵图已保存：", CONFUSION_MATRIX_FILE)


# =====================================================
# 12. 输出最佳模型的详细预测结果
# =====================================================

best_model_name = result_df.iloc[0]["Model"]
best_pred_df = prediction_dfs[best_model_name].copy()

print("=" * 80)
print("最佳 baseline 模型：", best_model_name)
print("最佳模型逐区县预测结果：")
print(best_pred_df)
print("=" * 80)

print("所有输出文件位置：")
print("1. 模型对比表：", RESULT_TABLE_FILE)
print("2. 逐区县预测结果：", PREDICTION_FILE)
print("3. 模型表现图：", BARPLOT_FILE)
print("4. 混淆矩阵图：", CONFUSION_MATRIX_FILE)
print("=" * 80)