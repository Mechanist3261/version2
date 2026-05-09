import os
import re
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# =====================================================
# 0. 路径设置
# =====================================================

BASE_DIR = r"D:\大学\code\version2\data_analysis\data_project"

CHARLS2020_DIR = os.path.join(BASE_DIR, "charls2020")
LIFE_HISTORY_DIR = os.path.join(BASE_DIR, "life_history")

GEO_FILE = os.path.join(BASE_DIR, "community_geo.csv")
RESOURCE_FILE = os.path.join(BASE_DIR, "district_resource_features.csv")

OUTPUT_FILE = os.path.join(BASE_DIR, "community_model_data.csv")


# =====================================================
# 1. 工具函数
# =====================================================

def read_dta(path):
    print(f"正在读取：{path}")
    return pd.read_stata(path, convert_categoricals=False)


def keep_existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


def clean_special_missing(df):
    """
    CHARLS 常见特殊缺失：
    -1 不适用 / 不知道
    -2 拒绝
    -8 缺失
    997 不知道
    999 拒绝
    """
    df = df.copy()
    special_values = [-1, -2, -8, -9, 997, 998, 999, 9997, 9998, 9999]
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(special_values, np.nan)
    return df


def flatten_columns(df):
    df.columns = [
        "_".join([str(i) for i in col if str(i) != ""])
        if isinstance(col, tuple) else col
        for col in df.columns
    ]
    return df


def minmax_series(s):
    s = pd.Series(s)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def find_child_columns(df, prefix):
    """
    例如 prefix='ca014_'，自动寻找 ca014_1_, ca014_2_ ...
    """
    return [c for c in df.columns if c.startswith(prefix)]


# =====================================================
# 2. 读取 CHARLS 2020 数据
# =====================================================

demo = read_dta(os.path.join(CHARLS2020_DIR, "Demographic_Background.dta"))
health = read_dta(os.path.join(CHARLS2020_DIR, "Health_Status_and_Functioning.dta"))
family = read_dta(os.path.join(CHARLS2020_DIR, "Family_Information.dta"))

demo = clean_special_missing(demo)
health = clean_special_missing(health)
family = clean_special_missing(family)

print("Demographic:", demo.shape)
print("Health:", health.shape)
print("Family:", family.shape)


# =====================================================
# 3. 提取人口学基础变量
# =====================================================

demo_cols = keep_existing_cols(demo, [
    "ID", "householdID", "communityID",
    "xrage", "xrgender",
    "ba001", "ba008", "ba009", "ba011",
    "ba018", "ba019"
])

demo_core = demo[demo_cols].copy()

# 年龄字段优先使用 xrage
if "xrage" in demo_core.columns:
    demo_core["age"] = demo_core["xrage"]
else:
    raise ValueError("没有找到 xrage 年龄变量，请检查 Demographic_Background.dta")

# 性别字段
if "xrgender" in demo_core.columns:
    demo_core["gender"] = demo_core["xrgender"]
elif "ba001" in demo_core.columns:
    demo_core["gender"] = demo_core["ba001"]
else:
    demo_core["gender"] = np.nan

# 独居天数，可作为家庭支持反向变量
if "ba018" in demo_core.columns:
    demo_core["live_alone_days"] = demo_core["ba018"]
else:
    demo_core["live_alone_days"] = np.nan


# =====================================================
# 4. 提取健康特征
# =====================================================

health_cols = keep_existing_cols(health, [
    "ID", "householdID", "communityID",
    "da001",       # 自评健康，具体含义以 codebook 为准
    "da002", "da002_1_", "da002_2_",
    "da003",
    "db001", "db002", "db003",
    "dc001", "dc002",
    "dd001", "dd002"
])

health_core = health[health_cols].copy()

# 自评健康：通常数值越大越差，这里转成“健康风险”
if "da001" in health_core.columns:
    health_core["self_health_risk"] = health_core["da001"]
else:
    health_core["self_health_risk"] = np.nan

# 慢病数量：自动寻找 da007_1_ 到 da007_14_ 一类变量
chronic_candidates = [
    c for c in health.columns
    if re.match(r"da\d+_\d+_", c) or re.match(r"da\d+_\d+_\d+_", c)
]

# 这里不强行使用所有变量，避免误把无关字段算进去
# 更稳妥：优先寻找常见慢病变量 da007_*
chronic_cols = [c for c in health.columns if c.startswith("da007_")]

if len(chronic_cols) > 0:
    chronic_df = health[["ID"] + chronic_cols].copy()
    chronic_df = clean_special_missing(chronic_df)
    # 一般 1 表示患病，非 1 表示未患病或缺失
    chronic_df["chronic_count"] = chronic_df[chronic_cols].apply(lambda row: np.sum(row == 1), axis=1)
    health_core = health_core.merge(chronic_df[["ID", "chronic_count"]], on="ID", how="left")
else:
    health_core["chronic_count"] = np.nan

# 生活功能困难：自动找 db/dc/dd 中可能的功能限制变量
func_cols = [
    c for c in health.columns
    if c.startswith(("db", "dc", "dd")) and c not in ["ID", "householdID", "communityID"]
]

# 不想误用太多变量时，可以先取前 20 个
func_cols = func_cols[:20]

if len(func_cols) > 0:
    func_df = health[["ID"] + func_cols].copy()
    func_df = clean_special_missing(func_df)
    func_df["function_limit_mean"] = func_df[func_cols].mean(axis=1)
    health_core = health_core.merge(func_df[["ID", "function_limit_mean"]], on="ID", how="left")
else:
    health_core["function_limit_mean"] = np.nan


# 合成个体层面健康风险
health_features = [
    "self_health_risk",
    "chronic_count",
    "function_limit_mean"
]

health_core["health_feature_ind"] = health_core[health_features].mean(axis=1)


# =====================================================
# 5. 提取家庭支持特征
# =====================================================

family_core = family[keep_existing_cols(family, ["householdID", "communityID"])].copy()

# 与子女同住月份 ca014_1_, ca014_2_ ...
co_live_cols = find_child_columns(family, "ca014_")

if len(co_live_cols) > 0:
    tmp = family[["householdID"] + co_live_cols].copy()
    tmp = clean_special_missing(tmp)
    tmp["child_colive_months"] = tmp[co_live_cols].sum(axis=1)
    family_core = family_core.merge(tmp[["householdID", "child_colive_months"]], on="householdID", how="left")
else:
    family_core["child_colive_months"] = np.nan

# 见子女频率 ca015_1_, ca015_2_ ...
see_child_cols = find_child_columns(family, "ca015_")

if len(see_child_cols) > 0:
    tmp = family[["householdID"] + see_child_cols].copy()
    tmp = clean_special_missing(tmp)

    # ca015 一般数值越小代表越频繁见面
    # 所以用 11 - 原值 转成“见面支持强度”
    see_support = tmp[see_child_cols].apply(lambda x: 11 - x)
    tmp["child_contact_support"] = see_support.mean(axis=1)

    family_core = family_core.merge(tmp[["householdID", "child_contact_support"]], on="householdID", how="left")
else:
    family_core["child_contact_support"] = np.nan

# 家庭支持：同住越多、见面越频繁、独居越少，支持越强
family_core["family_support_raw"] = family_core[[
    "child_colive_months",
    "child_contact_support"
]].mean(axis=1)


# =====================================================
# 6. 合并个体层数据
# =====================================================

person = demo_core.merge(
    health_core[["ID", "health_feature_ind", "self_health_risk", "chronic_count", "function_limit_mean"]],
    on="ID",
    how="left"
)

person = person.merge(
    family_core[["householdID", "family_support_raw"]],
    on="householdID",
    how="left"
)

print("个体合并后：", person.shape)


# =====================================================
# 7. 筛选老年样本
# =====================================================

person = person[person["age"] >= 60].copy()

print("60岁及以上样本：", person.shape)


# =====================================================
# 8. 可选：用生命历程数据识别迁移经历
# =====================================================

use_life_history = True

if use_life_history:
    residence_path = os.path.join(LIFE_HISTORY_DIR, "Residence.dta")

    if os.path.exists(residence_path):
        residence = read_dta(residence_path)
        residence = clean_special_missing(residence)

        # 生命历程居住史中，t001b_* 表示是否迁出某段居住地
        move_cols = [c for c in residence.columns if re.match(r"t001b_\d+_$", c)]

        # t003c_*_s4 是 care，t003c_*_s8 是 migrate
        care_cols = [c for c in residence.columns if re.match(r"t003c_\d+_s4$", c)]
        migrate_cols = [c for c in residence.columns if re.match(r"t003c_\d+_s8$", c)]

        res_core = residence[keep_existing_cols(residence, ["ID", "householdID", "communityID"])].copy()

        if len(move_cols) > 0:
            tmp = residence[["ID"] + move_cols].copy()
            tmp["has_move_history"] = tmp[move_cols].apply(lambda row: np.any(row == 1), axis=1).astype(int)
            res_core = res_core.merge(tmp[["ID", "has_move_history"]], on="ID", how="left")
        else:
            res_core["has_move_history"] = np.nan

        if len(care_cols) > 0:
            tmp = residence[["ID"] + care_cols].copy()
            tmp["care_move_history"] = tmp[care_cols].apply(lambda row: np.any(row == 4), axis=1).astype(int)
            res_core = res_core.merge(tmp[["ID", "care_move_history"]], on="ID", how="left")
        else:
            res_core["care_move_history"] = 0

        if len(migrate_cols) > 0:
            tmp = residence[["ID"] + migrate_cols].copy()
            tmp["migrate_history"] = tmp[migrate_cols].apply(lambda row: np.any(row == 8), axis=1).astype(int)
            res_core = res_core.merge(tmp[["ID", "migrate_history"]], on="ID", how="left")
        else:
            res_core["migrate_history"] = 0

        person = person.merge(
            res_core[["ID", "has_move_history", "care_move_history", "migrate_history"]],
            on="ID",
            how="left"
        )

        person["is_old_migrant"] = (
            (person["has_move_history"].fillna(0) == 1) |
            (person["care_move_history"].fillna(0) == 1) |
            (person["migrate_history"].fillna(0) == 1)
        ).astype(int)

        # 这里建议先不严格删除，只保留一个标记
        # 如果你想严格研究“老漂族”，可以打开下面这一行：
        # person = person[person["is_old_migrant"] == 1].copy()

    else:
        print("未找到 Residence.dta，跳过生命历程迁移识别。")
        person["is_old_migrant"] = 1
else:
    person["is_old_migrant"] = 1


# =====================================================
# 9. 缺失值填补
# =====================================================

model_individual_cols = [
    "age",
    "gender",
    "live_alone_days",
    "health_feature_ind",
    "self_health_risk",
    "chronic_count",
    "function_limit_mean",
    "family_support_raw",
    "is_old_migrant"
]

for col in model_individual_cols:
    if col not in person.columns:
        person[col] = np.nan

# 缺失指示变量
for col in model_individual_cols:
    person[col + "_missing"] = person[col].isna().astype(int)

imputer = SimpleImputer(strategy="median")
person[model_individual_cols] = imputer.fit_transform(person[model_individual_cols])


# =====================================================
# 10. 聚合到社区层级
# =====================================================

agg_dict = {
    "age": ["mean", "std"],
    "gender": ["mean"],
    "live_alone_days": ["mean"],
    "health_feature_ind": ["mean", "std"],
    "self_health_risk": ["mean"],
    "chronic_count": ["mean"],
    "function_limit_mean": ["mean"],
    "family_support_raw": ["mean", "std"],
    "is_old_migrant": ["mean", "sum"],
}

community = person.groupby("communityID").agg(agg_dict)
community = flatten_columns(community)
community = community.reset_index()

# 社区样本量
community_size = person.groupby("communityID").size().reset_index(name="sample_size")
community = community.merge(community_size, on="communityID", how="left")

print("社区层数据：", community.shape)


# =====================================================
# 11. 生成健康特征、家庭支持两个主变量
# =====================================================

# 健康特征：数值越高，健康服务需求越强
health_main_cols = [
    "health_feature_ind_mean",
    "self_health_risk_mean",
    "chronic_count_mean",
    "function_limit_mean"
]

health_main_cols = keep_existing_cols(community, health_main_cols)

community["健康特征"] = community[health_main_cols].mean(axis=1)

# 家庭支持：数值越高，支持越强
family_main_cols = [
    "family_support_raw_mean"
]

family_main_cols = keep_existing_cols(community, family_main_cols)

community["家庭支持"] = community[family_main_cols].mean(axis=1)

# 对独居天数做反向补充：独居越多，家庭支持越弱
if "live_alone_days_mean" in community.columns:
    live_alone_norm = minmax_series(community["live_alone_days_mean"])
    family_norm = minmax_series(community["家庭支持"])
    community["家庭支持"] = 0.7 * family_norm + 0.3 * (1 - live_alone_norm)


# =====================================================
# 12. 合并地理数据
# =====================================================

geo = pd.read_csv(GEO_FILE, encoding="utf-8-sig")

# 兼容列名
rename_geo = {}
if "lng" in geo.columns:
    rename_geo["lng"] = "经度"
if "lon" in geo.columns:
    rename_geo["lon"] = "经度"
if "lat" in geo.columns:
    rename_geo["lat"] = "纬度"

geo = geo.rename(columns=rename_geo)

need_geo_cols = ["communityID", "经度", "纬度"]
if "区县" in geo.columns:
    need_geo_cols.append("区县")

geo = geo[keep_existing_cols(geo, need_geo_cols)].copy()

community["communityID"] = community["communityID"].astype(str)
geo["communityID"] = geo["communityID"].astype(str)

community = community.merge(geo, on="communityID", how="left")

print("合并地理数据后：", community.shape)
print("缺失经纬度社区数：", community["经度"].isna().sum())


# =====================================================
# 13. 合并资源数据
# =====================================================

resource = pd.read_csv(RESOURCE_FILE, encoding="utf-8-sig")

# 兼容常见列名
rename_resource = {}

for col in resource.columns:
    if col in ["hospital_count", "medical_count", "医疗设施数", "医院数量"]:
        rename_resource[col] = "医疗资源"
    if col in ["elderly_count", "养老设施数", "养老机构数量", "eldercare_count"]:
        rename_resource[col] = "养老资源"

resource = resource.rename(columns=rename_resource)

# 如果资源表是区县层面
if "communityID" in resource.columns:
    resource["communityID"] = resource["communityID"].astype(str)
    community = community.merge(
        resource[keep_existing_cols(resource, ["communityID", "医疗资源", "养老资源"])],
        on="communityID",
        how="left"
    )

elif "区县" in resource.columns and "区县" in community.columns:
    community = community.merge(
        resource[keep_existing_cols(resource, ["区县", "医疗资源", "养老资源"])],
        on="区县",
        how="left"
    )

else:
    raise ValueError("资源表必须包含 communityID 或 区县，并且包含医疗资源、养老资源相关列。")

# 资源缺失填 0
for col in ["医疗资源", "养老资源"]:
    if col not in community.columns:
        community[col] = 0
    community[col] = community[col].fillna(0)


# =====================================================
# 14. 构建 need_score
# =====================================================

# 逻辑：
# 健康风险越高 -> 需求越高
# 家庭支持越低 -> 需求越高
# 医疗资源越少 -> 需求越高
# 养老资源越少 -> 需求越高

community["健康特征_norm"] = minmax_series(community["健康特征"])
community["家庭支持_norm"] = minmax_series(community["家庭支持"])
community["医疗资源_norm"] = minmax_series(community["医疗资源"])
community["养老资源_norm"] = minmax_series(community["养老资源"])

community["家庭支持不足"] = 1 - community["家庭支持_norm"]
community["医疗资源不足"] = 1 - community["医疗资源_norm"]
community["养老资源不足"] = 1 - community["养老资源_norm"]

need_components = [
    "健康特征_norm",
    "家庭支持不足",
    "医疗资源不足",
    "养老资源不足"
]

# PCA 自动赋权
X_need = community[need_components].replace([np.inf, -np.inf], np.nan)
X_need = X_need.fillna(X_need.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_need)

pca = PCA(n_components=1)
community["need_score_raw"] = pca.fit_transform(X_scaled)

# 保证方向：健康风险越高，need_score 越高
corr = np.corrcoef(community["need_score_raw"], community["健康特征_norm"])[0, 1]
if corr < 0:
    community["need_score_raw"] = -community["need_score_raw"]

community["need_score"] = minmax_series(community["need_score_raw"])


# =====================================================
# 15. 构建 need_level
# =====================================================

community["need_level"] = pd.qcut(
    community["need_score"],
    q=5,
    labels=[0, 1, 2, 3, 4],
    duplicates="drop"
).astype(int)


# =====================================================
# 16. 输出最终建模表
# =====================================================

final_cols = [
    "communityID",
    "经度",
    "纬度",
    "健康特征",
    "家庭支持",
    "医疗资源",
    "养老资源",
    "need_score",
    "need_level"
]

final_data = community[final_cols].copy()

final_data = final_data.dropna(subset=["communityID"])

final_data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("=" * 60)
print("最终建模表已生成：")
print(OUTPUT_FILE)
print(final_data.head())
print(final_data.shape)
print("need_level 分布：")
print(final_data["need_level"].value_counts().sort_index())
print("=" * 60)