import pandas as pd

train_data = pd.read_csv("./data/event_type_entity_extract_train.csv",
                         header=None,
                         names=["id", "sent", "entity", "label"])

# print("all size: ", train_data.size)
size = train_data.size
print(size)
# train_data.iloc[size-1]
for i, item in enumerate(train_data.iterrows()):
    print(item)
    print(item[0], item[1]["sent"])
    break


# new_data = train_data[train_data["entity"] != "其他"]
#
# new_data.to_csv("filter_data/train_data.csv", header=False, index=False)
#
# eval_data = pd.read_csv("./data/event_type_entity_extract_eval.csv",
#                          header=None,
#                          names=["id", "sent", "entity", "label"])
# print(eval_data.head(5))
# # print("all size: ", train_data.size)
#
#
# def trans(x):
#     entity = x.iloc[2]
#     label = "unk"
#     # print(entity)
#     if entity == "其他":
#         label = "NaN"
#     return label
#
#
# eval_data["label"] = eval_data.apply(trans, axis=1)
#
# eval_data.to_csv("filter_data/eval_data.csv", header=False, index=False)

