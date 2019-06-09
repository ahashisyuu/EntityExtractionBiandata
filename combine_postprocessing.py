import os

import pandas as pd
import pickle as pkl
from collections import Counter

def search_text(sent: str, *entity_list):
    case_sent = sent.lower()
    length = len(entity_list)
    i = 0
    index = -1
    while True:
        try:
            index = case_sent.index(entity_list[i])
            break
        except ValueError:
            # print(entity_list[i], "----------", case_sent)
            i += 1
            if i >= length:
                return "empty"
    entity = sent[index:index+len(entity_list[i])]
    return entity


# with open("all_company_name.pkl", "rb") as fr:
#     all_company_name = pkl.load(fr)


def is_normal(entity):
    # if len(entity) == 1:
    #     return False
    #
    # if entity not in all_company_name:
    #     print("===", entity)
    if len(entity) > 1:
        return True
    else:
        return False


def process(x):
    index = x[0]
    s1, s2, s3 = x[1:4]
    sent = x[4]
    entity = x[5]
    if entity == "其他":
        return "NaN"

    if is_normal(s1):
        final_text = search_text(sent, s1, s2, s3)
    elif is_normal(s2):
        final_text = search_text(sent, s2, s3)
    else:
        final_text = search_text(sent, s3)

    return final_text


def search_texts(sent, *entity_list):
    case_sent = sent.lower()
    length = len(entity_list)
    i = 0
    final_texts = []
    while i < length:
        try:
            index = case_sent.index(entity_list[i])
            final_texts.append(sent[index:index+len(entity_list[i])])
        except ValueError:
            # print(entity_list[i], "----------", case_sent)
            pass
        finally:
            i += 1
    return final_texts


def process_multiple(x):
    index = x[0]
    s1, s2, s3 = x[1:4]
    sent = x[4]
    entity = x[5]

    if entity == "其他":
        return ["NaN"]

    if is_normal(s1):
        final_texts = search_texts(sent, s1, s2, s3)
    elif is_normal(s2):
        final_texts = search_texts(sent, s2, s3)
    else:
        final_texts = search_texts(sent, s3)

    if len(final_texts) == 0:
        print(index)

    return final_texts


def select_best(x):
    if str(x[1]).lower() == "nan":
        return "NaN"

    counter = Counter(x[1:])
    best_text = counter.most_common(1)[0][0]
    return best_text


def main():
    # combine_results = pd.DataFrame()

    test_data = pd.read_csv("./filter_data/test_data.csv",
                            header=None,
                            names=["id", "sent", "entity", "label"])
    combine_results = test_data[["id"]]

    file_list = os.listdir("combine2_results")
    for i, name in enumerate(file_list):

        results_data = pd.read_csv("combine2_results/" + name,
                                   header=None,
                                   names=["id", "pred"])
        combine_results["cv{}".format(i)] = results_data["pred"]

    # print(combine_multiple_results["cv0"].iloc[:5])
    combine_results["best_text"] = combine_results.apply(select_best, axis=1)
    combine_results[["id", "best_text"]].to_csv("results/all_model20_best2.txt", header=False, index=False)



if __name__ == "__main__":
    main()

