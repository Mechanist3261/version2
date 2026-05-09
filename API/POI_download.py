import requests
import pandas as pd
import time

KEY = "86d2df5928332ee0447cda046b39674f"
CITY = "杭州"

# 每类多放几个关键词，样本量会明显增加
poi_tasks = {
    "hospital": [
        "医院",
        "综合医院",
        "专科医院",
        "社区卫生服务中心",
        "卫生院",
        "诊所"
    ],
    "elderly": [
        "养老院",
        "养老服务中心",
        "养老机构",
        "老年公寓",
        "护理院",
        "康养中心"
    ],
    "social": [
        "老年活动中心",
        "社区服务中心",
        "党群服务中心",
        "文化礼堂",
        "社区居委会",
        "街道办事处"
    ]
}

def fetch_poi(keyword, poi_type, max_pages=10):
    results = []

    for page in range(1, max_pages + 1):
        url = "https://restapi.amap.com/v3/place/text"

        params = {
            "key": KEY,
            "keywords": keyword,
            "city": CITY,
            "citylimit": "true",
            "offset": 25,
            "page": page,
            "extensions": "all"
        }

        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if data.get("status") != "1":
            print(f"{keyword} 第{page}页请求失败：", data)
            break

        pois = data.get("pois", [])

        if len(pois) == 0:
            break

        for poi in pois:
            location = poi.get("location", "")
            if "," in location:
                lng, lat = location.split(",")
            else:
                lng, lat = None, None

            results.append({
                "name": poi.get("name"),
                "address": poi.get("address"),
                "province": poi.get("pname"),
                "city": poi.get("cityname"),
                "区县": poi.get("adname"),
                "lng": lng,
                "lat": lat,
                "keyword": keyword,
                "type": poi_type
            })

        print(f"{poi_type} - {keyword} 第{page}页：{len(pois)}条")
        time.sleep(0.2)

    return results


all_results = []

for poi_type, keywords in poi_tasks.items():
    for keyword in keywords:
        all_results.extend(fetch_poi(keyword, poi_type, max_pages=10))

all_poi = pd.DataFrame(all_results)

# 经纬度转数值
all_poi["lng"] = pd.to_numeric(all_poi["lng"], errors="coerce")
all_poi["lat"] = pd.to_numeric(all_poi["lat"], errors="coerce")

# 删除无区县、无坐标数据
all_poi = all_poi.dropna(subset=["区县", "lng", "lat"])

# 只保留杭州常用区县
valid_districts = [
    "上城区", "拱墅区", "西湖区", "滨江区", "萧山区",
    "余杭区", "临平区", "钱塘区", "富阳区", "临安区",
    "桐庐县", "淳安县", "建德市"
]

all_poi = all_poi[all_poi["区县"].isin(valid_districts)]

# 去重：同名 + 同区县 + 同类型，只保留一条
all_poi = all_poi.drop_duplicates(
    subset=["name", "区县", "type"]
)

# 保存明细
all_poi.to_csv("hangzhou_all_poi_enhanced.csv", index=False, encoding="utf-8-sig")

print("\nPOI总量：")
print(all_poi["type"].value_counts())

print("\n各区县分布：")
print(pd.crosstab(all_poi["区县"], all_poi["type"]))

# 区县聚合
district_resource = pd.crosstab(
    all_poi["区县"],
    all_poi["type"]
).reset_index()

# 保证三列都存在
for col in ["hospital", "elderly", "social"]:
    if col not in district_resource.columns:
        district_resource[col] = 0

district_resource = district_resource[
    ["区县", "hospital", "elderly", "social"]
]

district_resource.to_csv(
    "district_resource_features.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n区县公共服务特征已保存：district_resource_features.csv")
print(district_resource)