import os
import re
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# =====================================================
# 0. 路径设置
# =====================================================

BASE_DIR = r"D:\大学\code\version2\data_analysis\data_project"

CHARLS2020_DIR = os.path.join(BASE_DIR, "charls2020")
LIFE_HISTORY_DIR = os.path.join(BASE_DIR, "life_history")

RESOURCE_FILE = os.path.join(BASE_DIR, "district_resource_features.csv")
POI_FILE = os.path.join(BASE_DIR, "hangzhou_all_poi_enhanced.csv")

OUTPUT_FILE = os.path.join(BASE_DIR, "community_model_data.csv")
MAPPING_OUTPUT_FILE = os.path.join(BASE_DIR, "simulated_community_district_mapping.csv")


# =====================================================
# 1. 工具函数
# =====================================================

def read_dta(path):
    print(f"正在读取：{path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    return pd.read_stata(path, convert_categoricals=False)


def read_csv_auto(path):
    print(f"正在读取：{path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")

    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"无法识别文件编码：{path}")


def clean_special_missing(df):
    df = df.copy()
    special_values = [-1, -2, -8, -9, 997, 998, 999, 9997, 9998, 9999]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(special_values, np.nan)

    return df


def keep_existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


def check_required_columns(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} 缺少必要字段：{missing}")


def flatten_columns(df):
    df.columns = [
        "_".join([str(i) for i in col if str(i) != ""])
        if isinstance(col, tuple) else col
        for col in df.columns
    ]
    return df


def minmax_series(s):
    s = pd.Series(s).astype(float)

    if s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index)

    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - s.min()) / (s.max() - s.min())


def weighted_mean(values, weights):
    values = pd.Series(values).astype(float)
    weights = pd.Series(weights).astype(float)

    mask = values.notna() & weights.notna()

    if mask.sum() == 0:
        return np.nan

    if weights[mask].sum() == 0:
        return values[mask].mean()

    return np.average(values[mask], weights=weights[mask])


def safe_impute_median(df, cols):
    """
    安全中位数填补。
    解决某列全缺失时 SimpleImputer 返回列数不一致的问题。
    """
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].isna().all():
            print(f"提示：{col} 全部缺失，已填充为 0。")
            df[col] = 0

    imputer = SimpleImputer(strategy="median")
    arr = imputer.fit_transform(df[cols])

    df[cols] = pd.DataFrame(arr, columns=cols, index=df.index)

    return df


def existing_mean(df, cols, new_col):
    """
    对存在的列取均值，避免某些列不存在时报错。
    """
    real_cols = [c for c in cols if c in df.columns]

    if len(real_cols) == 0:
        print(f"警告：{new_col} 没有可用变量，已设为 0。")
        df[new_col] = 0
    else:
        df[new_col] = df[real_cols].mean(axis=1)

    return df


# =====================================================
# 2. 读取 CHARLS 2020 数据
# =====================================================

demo = read_dta(os.path.join(CHARLS2020_DIR, "Demographic_Background.dta"))
health = read_dta(os.path.join(CHARLS2020_DIR, "Health_Status_and_Functioning.dta"))
family = read_dta(os.path.join(CHARLS2020_DIR, "Family_Information.dta"))

demo = clean_special_missing(demo)
health = clean_special_missing(health)
family = clean_special_missing(family)

print("Demographic 数据维度：", demo.shape)
print("Health 数据维度：", health.shape)
print("Family 数据维度：", family.shape)


# =====================================================
# 3. 提取人口学变量
# =====================================================

demo_cols = keep_existing_cols(demo, [
    "ID",
    "householdID",
    "communityID",
    "xrage",
    "xrgender",
    "ba001",
    "ba008",
    "ba009",
    "ba011",
    "ba018",
    "ba019"
])

demo_core = demo[demo_cols].copy()

check_required_columns(
    demo_core,
    ["ID", "householdID", "communityID", "xrage"],
    "Demographic_Background.dta"
)

demo_core["ID"] = demo_core["ID"].astype(str)
demo_core["householdID"] = demo_core["householdID"].astype(str)
demo_core["communityID"] = demo_core["communityID"].astype(str)

demo_core["age"] = pd.to_numeric(demo_core["xrage"], errors="coerce")

if "xrgender" in demo_core.columns:
    demo_core["gender"] = pd.to_numeric(demo_core["xrgender"], errors="coerce")
elif "ba001" in demo_core.columns:
    demo_core["gender"] = pd.to_numeric(demo_core["ba001"], errors="coerce")
else:
    demo_core["gender"] = np.nan

if "ba018" in demo_core.columns:
    demo_core["live_alone_days"] = pd.to_numeric(demo_core["ba018"], errors="coerce")
else:
    demo_core["live_alone_days"] = np.nan


# =====================================================
# 4. 提取健康特征
# =====================================================

health_core = health[keep_existing_cols(health, [
    "ID",
    "householdID",
    "communityID",
    "da001"
])].copy()

check_required_columns(health_core, ["ID"], "Health_Status_and_Functioning.dta")

health_core["ID"] = health_core["ID"].astype(str)

# 4.1 自评健康风险
if "da001" in health_core.columns:
    health_core["self_health_risk"] = pd.to_numeric(health_core["da001"], errors="coerce")
else:
    health_core["self_health_risk"] = np.nan


# 4.2 慢病数量：你的文件里是 da002_1_ 到 da002_15_
chronic_cols = [
    c for c in health.columns
    if re.match(r"^da002_\d+_$", c)
]

print("识别到的慢病变量：", chronic_cols)

if len(chronic_cols) > 0:
    chronic_df = health[["ID"] + chronic_cols].copy()
    chronic_df["ID"] = chronic_df["ID"].astype(str)
    chronic_df = clean_special_missing(chronic_df)

    for col in chronic_cols:
        chronic_df[col] = pd.to_numeric(chronic_df[col], errors="coerce")

    chronic_df["chronic_count"] = chronic_df[chronic_cols].apply(
        lambda row: np.sum(row == 1),
        axis=1
    )

    health_core = health_core.merge(
        chronic_df[["ID", "chronic_count"]],
        on="ID",
        how="left"
    )
else:
    health_core["chronic_count"] = np.nan


# 4.3 功能受限变量：你的文件识别到 db001 等 34 个变量
function_cols = [
    c for c in health.columns
    if re.match(r"^db\d{3}$", c)
]

print("识别到的功能受限变量数量：", len(function_cols))
print("功能受限变量示例：", function_cols[:10])

if len(function_cols) > 0:
    function_df = health[["ID"] + function_cols].copy()
    function_df["ID"] = function_df["ID"].astype(str)
    function_df = clean_special_missing(function_df)

    for col in function_cols:
        function_df[col] = pd.to_numeric(function_df[col], errors="coerce")

    function_df["function_limit_mean"] = function_df[function_cols].mean(axis=1)

    health_core = health_core.merge(
        function_df[["ID", "function_limit_mean"]],
        on="ID",
        how="left"
    )
else:
    health_core["function_limit_mean"] = np.nan


health_feature_cols = [
    "self_health_risk",
    "chronic_count",
    "function_limit_mean"
]

health_core = safe_impute_median(health_core, health_feature_cols)

health_core["health_feature_ind"] = health_core[health_feature_cols].mean(axis=1)


# =====================================================
# 5. 提取家庭支持特征
# =====================================================

family_core = family[keep_existing_cols(family, ["householdID", "communityID"])].copy()

check_required_columns(family_core, ["householdID"], "Family_Information.dta")

family_core["householdID"] = family_core["householdID"].astype(str)


# 5.1 与子女同住月份 ca014_*
co_live_cols = [
    c for c in family.columns
    if re.match(r"^ca014_\d+_$", c)
]

print("识别到的与子女同住变量：", co_live_cols[:10], "数量：", len(co_live_cols))

if len(co_live_cols) > 0:
    tmp = family[["householdID"] + co_live_cols].copy()
    tmp["householdID"] = tmp["householdID"].astype(str)
    tmp = clean_special_missing(tmp)

    for col in co_live_cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    tmp["child_colive_months"] = tmp[co_live_cols].sum(axis=1, min_count=1)

    family_core = family_core.merge(
        tmp[["householdID", "child_colive_months"]],
        on="householdID",
        how="left"
    )
else:
    family_core["child_colive_months"] = np.nan


# 5.2 见子女频率 ca015_*
see_child_cols = [
    c for c in family.columns
    if re.match(r"^ca015_\d+_$", c)
]

print("识别到的见子女频率变量：", see_child_cols[:10], "数量：", len(see_child_cols))

if len(see_child_cols) > 0:
    tmp = family[["householdID"] + see_child_cols].copy()
    tmp["householdID"] = tmp["householdID"].astype(str)
    tmp = clean_special_missing(tmp)

    for col in see_child_cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    # 10 表示 Other，设为缺失
    tmp[see_child_cols] = tmp[see_child_cols].replace(10, np.nan)

    # 数值越小见面越频繁，所以反向
    contact_support = 10 - tmp[see_child_cols]
    tmp["child_contact_support"] = contact_support.mean(axis=1)

    family_core = family_core.merge(
        tmp[["householdID", "child_contact_support"]],
        on="householdID",
        how="left"
    )
else:
    family_core["child_contact_support"] = np.nan


family_support_cols = [
    "child_colive_months",
    "child_contact_support"
]

family_core = safe_impute_median(family_core, family_support_cols)

family_core["family_support_raw"] = family_core[family_support_cols].mean(axis=1)


# =====================================================
# 6. 合并个体层数据
# =====================================================

person = demo_core.merge(
    health_core[[
        "ID",
        "health_feature_ind",
        "self_health_risk",
        "chronic_count",
        "function_limit_mean"
    ]],
    on="ID",
    how="left"
)

person = person.merge(
    family_core[[
        "householdID",
        "family_support_raw",
        "child_colive_months",
        "child_contact_support"
    ]],
    on="householdID",
    how="left"
)

print("个体层合并后维度：", person.shape)


# =====================================================
# 7. 筛选 60 岁及以上样本
# =====================================================

person = person[person["age"] >= 60].copy()

print("60岁及以上样本维度：", person.shape)


# =====================================================
# 8. 生命历程数据：迁移经历识别
# =====================================================

residence_path = os.path.join(LIFE_HISTORY_DIR, "Residence.dta")

if os.path.exists(residence_path):
    residence = read_dta(residence_path)
    residence = clean_special_missing(residence)

    residence["ID"] = residence["ID"].astype(str)

    move_cols = [
        c for c in residence.columns
        if re.match(r"^t001b_\d+_$", c)
    ]

    care_cols = [
        c for c in residence.columns
        if re.match(r"^t003c_\d+_s4$", c)
    ]

    migrate_cols = [
        c for c in residence.columns
        if re.match(r"^t003c_\d+_s8$", c)
    ]

    res_core = residence[["ID"]].copy()

    if len(move_cols) > 0:
        tmp = residence[["ID"] + move_cols].copy()
        tmp["ID"] = tmp["ID"].astype(str)

        tmp["has_move_history"] = tmp[move_cols].apply(
            lambda row: np.any(row == 1),
            axis=1
        ).astype(int)

        res_core = res_core.merge(
            tmp[["ID", "has_move_history"]],
            on="ID",
            how="left"
        )
    else:
        res_core["has_move_history"] = 0

    if len(care_cols) > 0:
        tmp = residence[["ID"] + care_cols].copy()
        tmp["ID"] = tmp["ID"].astype(str)

        tmp["care_move_history"] = tmp[care_cols].apply(
            lambda row: np.any(row == 4),
            axis=1
        ).astype(int)

        res_core = res_core.merge(
            tmp[["ID", "care_move_history"]],
            on="ID",
            how="left"
        )
    else:
        res_core["care_move_history"] = 0

    if len(migrate_cols) > 0:
        tmp = residence[["ID"] + migrate_cols].copy()
        tmp["ID"] = tmp["ID"].astype(str)

        tmp["migrate_history"] = tmp[migrate_cols].apply(
            lambda row: np.any(row == 8),
            axis=1
        ).astype(int)

        res_core = res_core.merge(
            tmp[["ID", "migrate_history"]],
            on="ID",
            how="left"
        )
    else:
        res_core["migrate_history"] = 0

    person = person.merge(
        res_core[[
            "ID",
            "has_move_history",
            "care_move_history",
            "migrate_history"
        ]],
        on="ID",
        how="left"
    )

    person["is_old_migrant"] = (
        (person["has_move_history"].fillna(0) == 1) |
        (person["care_move_history"].fillna(0) == 1) |
        (person["migrate_history"].fillna(0) == 1)
    ).astype(int)

else:
    print("未找到 Residence.dta，is_old_migrant 默认设为 1。")
    person["is_old_migrant"] = 1


# =====================================================
# 9. 个体层缺失值处理
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
    "child_colive_months",
    "child_contact_support",
    "is_old_migrant"
]

for col in model_individual_cols:
    if col not in person.columns:
        person[col] = np.nan
    person[col + "_missing"] = person[col].isna().astype(int)

person = safe_impute_median(person, model_individual_cols)


# =====================================================
# 10. 聚合到 CHARLS 匿名 communityID 层面
# =====================================================

community_agg_dict = {
    "age": ["mean", "std"],
    "gender": ["mean"],
    "live_alone_days": ["mean"],
    "health_feature_ind": ["mean", "std"],
    "self_health_risk": ["mean"],
    "chronic_count": ["mean"],
    "function_limit_mean": ["mean"],
    "family_support_raw": ["mean", "std"],
    "child_colive_months": ["mean"],
    "child_contact_support": ["mean"],
    "is_old_migrant": ["mean", "sum"]
}

community = person.groupby("communityID").agg(community_agg_dict)
community = flatten_columns(community)
community = community.reset_index()

community_size = person.groupby("communityID").size().reset_index(name="sample_size")

community = community.merge(
    community_size,
    on="communityID",
    how="left"
)

community["communityID"] = community["communityID"].astype(str)

print("匿名 communityID 层数据维度：", community.shape)
print("匿名 communityID 数量：", community["communityID"].nunique())
print("community 聚合后字段：")
print(community.columns.tolist())


# =====================================================
# 11. 读取区县资源表
# =====================================================

resource = read_csv_auto(RESOURCE_FILE)

print("区县资源表字段：", resource.columns.tolist())

check_required_columns(resource, ["区县"], "district_resource_features.csv")

resource = resource.copy()
resource["区县"] = resource["区县"].astype(str)

rename_resource = {
    "hospital": "医疗资源",
    "elderly": "养老资源",
    "social": "社交资源"
}

resource = resource.rename(columns=rename_resource)

check_required_columns(
    resource,
    ["区县", "医疗资源", "养老资源"],
    "district_resource_features.csv"
)

if "社交资源" not in resource.columns:
    resource["社交资源"] = 0

for col in ["医疗资源", "养老资源", "社交资源"]:
    resource[col] = pd.to_numeric(resource[col], errors="coerce").fillna(0)

districts = resource[[
    "区县",
    "医疗资源",
    "养老资源",
    "社交资源"
]].copy()

print("区县数量：", districts["区县"].nunique())
print(districts.head())


# =====================================================
# 12. 从 POI 文件生成区县经纬度
# =====================================================

poi = read_csv_auto(POI_FILE)

print("POI 表字段：", poi.columns.tolist())

check_required_columns(
    poi,
    ["区县", "lng", "lat"],
    "hangzhou_all_poi_enhanced.csv"
)

poi = poi.copy()
poi["区县"] = poi["区县"].astype(str)
poi["lng"] = pd.to_numeric(poi["lng"], errors="coerce")
poi["lat"] = pd.to_numeric(poi["lat"], errors="coerce")

poi = poi.dropna(subset=["区县", "lng", "lat"]).copy()

if "name" in poi.columns:
    district_geo = poi.groupby("区县").agg(
        经度=("lng", "mean"),
        纬度=("lat", "mean"),
        poi_count=("name", "count")
    ).reset_index()
else:
    district_geo = poi.groupby("区县").agg(
        经度=("lng", "mean"),
        纬度=("lat", "mean"),
        poi_count=("lng", "count")
    ).reset_index()

districts = districts.merge(
    district_geo,
    on="区县",
    how="left"
)

districts["经度"] = districts["经度"].fillna(districts["经度"].mean())
districts["纬度"] = districts["纬度"].fillna(districts["纬度"].mean())
districts["poi_count"] = districts["poi_count"].fillna(0)

print("合并区县经纬度后：")
print(districts.head())
print("缺失经度数量：", districts["经度"].isna().sum())
print("缺失纬度数量：", districts["纬度"].isna().sum())


# =====================================================
# 13. 模拟区级映射：communityID -> 区县
# =====================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

districts_for_mapping = districts.copy()

districts_for_mapping["mapping_weight_raw"] = (
    districts_for_mapping["poi_count"].fillna(0)
    + districts_for_mapping["医疗资源"].fillna(0)
    + districts_for_mapping["养老资源"].fillna(0)
    + districts_for_mapping["社交资源"].fillna(0)
)

if districts_for_mapping["mapping_weight_raw"].sum() == 0:
    districts_for_mapping["mapping_weight"] = 1 / len(districts_for_mapping)
else:
    districts_for_mapping["mapping_weight"] = (
        districts_for_mapping["mapping_weight_raw"]
        / districts_for_mapping["mapping_weight_raw"].sum()
    )

unique_communities = community["communityID"].dropna().astype(str).unique()

mapped_districts = np.random.choice(
    districts_for_mapping["区县"].values,
    size=len(unique_communities),
    replace=True,
    p=districts_for_mapping["mapping_weight"].values
)

community_district_map = pd.DataFrame({
    "communityID": unique_communities,
    "区县": mapped_districts
})

community_district_map.to_csv(
    MAPPING_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("模拟映射表已保存：", MAPPING_OUTPUT_FILE)
print("模拟映射区县分布：")
print(community_district_map["区县"].value_counts())


# =====================================================
# 14. communityID 层升级为区县层
# =====================================================

community = community.merge(
    community_district_map,
    on="communityID",
    how="left"
)

# 注意：groupby 后，function_limit_mean 会变成 function_limit_mean_mean
community = existing_mean(
    community,
    [
        "health_feature_ind_mean",
        "self_health_risk_mean",
        "chronic_count_mean",
        "function_limit_mean_mean"
    ],
    "健康特征_community"
)

community = existing_mean(
    community,
    [
        "family_support_raw_mean",
        "child_colive_months_mean",
        "child_contact_support_mean"
    ],
    "家庭支持_community"
)

# 独居天数修正：独居越多，家庭支持越弱
if "live_alone_days_mean" in community.columns:
    live_alone_norm = minmax_series(community["live_alone_days_mean"])
    family_norm = minmax_series(community["家庭支持_community"])

    community["家庭支持_community"] = (
        0.7 * family_norm
        + 0.3 * (1 - live_alone_norm)
    )

district_rows = []

for district, sub in community.groupby("区县"):
    row = {
        "区县": district,
        "健康特征": weighted_mean(
            sub["健康特征_community"],
            sub["sample_size"]
        ),
        "家庭支持": weighted_mean(
            sub["家庭支持_community"],
            sub["sample_size"]
        ),
        "样本量": sub["sample_size"].sum(),
        "映射社区数": sub["communityID"].nunique()
    }

    district_rows.append(row)

district_feature = pd.DataFrame(district_rows)

print("区县特征表：")
print(district_feature.head())


# =====================================================
# 15. 合并区县资源和经纬度
# =====================================================

district_model = districts.merge(
    district_feature,
    on="区县",
    how="left"
)

for col in ["健康特征", "家庭支持"]:
    district_model[col] = pd.to_numeric(district_model[col], errors="coerce")

    if district_model[col].isna().all():
        district_model[col] = 0
    else:
        district_model[col] = district_model[col].fillna(district_model[col].median())

district_model["样本量"] = district_model["样本量"].fillna(0)
district_model["映射社区数"] = district_model["映射社区数"].fillna(0)

print("区县建模表初步结果：")
print(district_model.head())


# =====================================================
# 16. 构建 need_score
# =====================================================

district_model["健康特征_norm"] = minmax_series(district_model["健康特征"])
district_model["家庭支持_norm"] = minmax_series(district_model["家庭支持"])
district_model["医疗资源_norm"] = minmax_series(district_model["医疗资源"])
district_model["养老资源_norm"] = minmax_series(district_model["养老资源"])

district_model["家庭支持不足"] = 1 - district_model["家庭支持_norm"]
district_model["医疗资源不足"] = 1 - district_model["医疗资源_norm"]
district_model["养老资源不足"] = 1 - district_model["养老资源_norm"]

need_components = [
    "健康特征_norm",
    "家庭支持不足",
    "医疗资源不足",
    "养老资源不足"
]

X_need = district_model[need_components].copy()
X_need = X_need.replace([np.inf, -np.inf], np.nan)
X_need = X_need.fillna(X_need.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_need)

pca = PCA(n_components=1)
district_model["need_score_raw"] = pca.fit_transform(X_scaled)

corr = np.corrcoef(
    district_model["need_score_raw"],
    district_model["健康特征_norm"]
)[0, 1]

if np.isnan(corr):
    corr = 1

if corr < 0:
    district_model["need_score_raw"] = -district_model["need_score_raw"]

district_model["need_score"] = minmax_series(district_model["need_score_raw"])


# =====================================================
# 17. 构建 need_level
# =====================================================

district_model["need_rank"] = district_model["need_score"].rank(method="first")

district_model["need_level"] = pd.qcut(
    district_model["need_rank"],
    q=5,
    labels=[0, 1, 2, 3, 4]
).astype(int)


# =====================================================
# 18. 输出最终区县建模表
# =====================================================

final_cols = [
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

final_data = district_model[final_cols].copy()

final_data = final_data.sort_values(
    "need_score",
    ascending=False
).reset_index(drop=True)

final_data.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 80)
print("最终区县建模表已生成：")
print(OUTPUT_FILE)
print("=" * 80)
print(final_data)
print("=" * 80)
print("最终表维度：", final_data.shape)
print("缺失值统计：")
print(final_data.isna().sum())
print("=" * 80)
print("need_level 分布：")
print(final_data["need_level"].value_counts().sort_index())
print("=" * 80)
print("模拟映射文件：")
print(MAPPING_OUTPUT_FILE)
print("=" * 80)