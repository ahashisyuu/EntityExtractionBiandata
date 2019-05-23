import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("./data/event_type_entity_extract_train.csv",
                         header=None,
                         names=["id", "sent", "entity", "label"])

print("all size: ", train_data.count())
print('--------------------------------')

new_data = train_data[train_data["entity"] != "其他"]

print("new data size: ", new_data.count())
print('--------------------------------')

train_label = new_data["label"]
x_train, x_dev, y_train, y_dev = train_test_split(new_data, train_label,
                                                  test_size=0.25,
                                                  random_state=0)


x_train.to_csv("filter_data/train_data.csv", header=False, index=False)
x_dev.to_csv("filter_data/dev_data.csv", header=False, index=False)

eval_data = pd.read_csv("./data/event_type_entity_extract_eval.csv",
                         header=None,
                         names=["id", "sent", "entity", "label"])
print(eval_data.head(5))
print("eval size: ", eval_data.count())


def trans(x):
    entity = x.iloc[2]
    label = "unk"
    # print(entity)
    if entity == "其他":
        label = "NaN"
    return label


eval_data["label"] = eval_data.apply(trans, axis=1)

eval_data.to_csv("filter_data/test_data.csv", header=False, index=False)

