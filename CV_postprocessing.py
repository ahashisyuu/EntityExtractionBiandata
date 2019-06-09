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
    counter = Counter(x[3:8])
    best_text = counter.most_common(1)[0][0]
    return best_text


def select_multiple_best(x):
    cv0, cv1, cv2, cv3, cv4 = x[3:8]
    final_list = cv0[:2] + cv1[:2] + cv2[:2] + cv3[:2] + cv4[:2]
    if len(final_list) == 0:
        return "empty"
    counter = Counter(final_list)
    best_text = counter.most_common(1)[0][0]
    return best_text


def main():
    # combine_results = pd.DataFrame()

    test_data = pd.read_csv("./filter_data/test_data.csv",
                            header=None,
                            names=["id", "sent", "entity", "label"])
    combine_results = test_data[["id", "sent", "entity"]]
    combine_multiple_results = test_data[["id", "sent", "entity"]]

    for i in range(5):

        results_data = pd.read_csv("results/cv_results_lstm_2th_{}.csv".format(i),
                                   header=None,
                                   names=["id", "s1", "s2", "s3"])
        results_data["sent"] = test_data["sent"]
        results_data["entity"] = test_data["entity"]
        results_data["final"] = results_data.apply(process, axis=1)
        final_results = results_data[["id", "final"]]
        final_results.to_csv("results/final_cv_results_lstm_2th_{}.txt".format(i), header=False, index=False)

        combine_results["cv{}".format(i)] = results_data["final"]
        combine_multiple_results["cv{}".format(i)] = results_data.apply(process_multiple, axis=1)

    # print(combine_multiple_results["cv0"].iloc[:5])
    combine_results["best_text"] = combine_results.apply(select_best, axis=1)
    combine_results[["id", "best_text"]].to_csv("results/cv_best_text_results_lstm_2th.txt", header=False, index=False)

    combine_multiple_results["best_text"] = combine_multiple_results.apply(select_multiple_best, axis=1)
    combine_multiple_results[["id", "best_text"]].to_csv("results/cv_multiple_best_results_lstm_2th.txt",
                                                         header=False,
                                                         index=False)



if __name__ == "__main__":
    main()

