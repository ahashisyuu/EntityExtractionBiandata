import json

import pandas as pd
import pickle as pkl


company_name1 = pd.read_csv("./company_name/company_name1.csv")

# print(company_name1.head(5))
all_company_name = set()

for c in company_name1["公司简称"].tolist():
    # print(c)
    c = "".join(c.split())
    all_company_name.add(c)

for c in company_name1["公司全称"].tolist():
    c = "".join(c.split())
    all_company_name.add(c)

company_name2 = pd.read_csv("./company_name/company_name2.csv",
                            header=None, names=["code", "公司简称", "公司全称"])

for c in company_name2["公司简称"].tolist():
    # print(c)
    c = "".join(c.split())
    all_company_name.add(c)

for c in company_name2["公司全称"].tolist():
    c = "".join(c.split())
    all_company_name.add(c)

company_name3 = pd.read_csv("./company_name/company_name3.csv",
                            header=None, names=["公司简称"])

for c in company_name2["公司简称"].tolist():
    c = "".join(c.split())
    all_company_name.add(c)

print(all_company_name)
with open("all_company_name.json", "w") as fw:
    json.dump(list(all_company_name), fw)

# print("万科Ａ" in all_company_name)
# from utils import strQ2B, strB2Q
# print("万科A" in all_company_name)
# print(strQ2B("万科Ａ"), strB2Q("万科A"))
# print("-----", strQ2B("万科") == strB2Q("万科"))
# print(strB2Q("万科A") in all_company_name)
# print(strQ2B("万科A") in all_company_name)


