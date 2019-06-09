import os

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

train_data = pd.read_csv("./data/event_type_entity_extract_train.csv",
                         header=None,
                         names=["id", "sent", "entity", "label"])

print("all size: ", train_data.count())
print('--------------------------------')

new_data = train_data[train_data["entity"] != "其他"]

sample = train_data[train_data["id"] == 106211]

sample_sent = sample["sent"].tolist()[0]


new_data = new_data.dropna(axis=0)

# drop [[+_+]]
def _drop_noise(sent: str):
    # print(sent)
    while True:
        try:
            index = sent.index("[[+_+]]")
            # print(index)
            sent = sent[:index] + sent[index + len("[[+_+]]"):]
        except ValueError:
            break
    return sent

# print(_drop_noise(sample_sent))

new_data["sent"] = new_data["sent"].apply(_drop_noise)

# print(new_data[new_data["id"] == 103844])
#
# print("new data size: ", new_data.count())
# print('--------------------------------')

train_label = new_data["label"]

kf = KFold(n_splits=5, random_state=5)

for i, (train_index, dev_index) in enumerate(kf.split(new_data, train_label)):
    x_train = new_data.iloc[train_index]
    x_dev = new_data.iloc[dev_index]

    if os.path.exists("CV_data/data{}".format(i)) is False:
        os.mkdir("CV_data/data{}".format(i))

    x_train.to_csv("CV_data/data{}/train_data.csv".format(i), header=False, index=False)
    x_dev.to_csv("CV_data/data{}/dev_data.csv".format(i), header=False, index=False)

# eval_data = pd.read_csv("./data/event_type_entity_extract_eval.csv",
#                          header=None,
#                          names=["id", "sent", "entity", "label"])
# # print(eval_data.head(5))
# # print("eval size: ", eval_data.count())
#
#
# eval_data["sent"] = eval_data["sent"].apply(_drop_noise)
#
# eval_data.to_csv("filter_data/test_data.csv", header=False, index=False)

