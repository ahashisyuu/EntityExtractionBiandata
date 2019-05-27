import pandas as pd
import pickle as pkl


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


with open("all_company_name.pkl", "rb") as fr:
    all_company_name = pkl.load(fr)


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


def main():
    test_data = pd.read_csv("./filter_data/test_data.csv",
                            header=None,
                            names=["id", "sent", "entity", "label"])
    results_data = pd.read_csv("results.csv",
                               header=None,
                               names=["id", "s1", "s2", "s3"])
    results_data["sent"] = test_data["sent"]
    results_data["entity"] = test_data["entity"]
    results_data["final"] = results_data.apply(process, axis=1)
    final_results = results_data[["id", "final"]]
    final_results.to_csv("final_results.txt", header=False, index=False)


if __name__ == "__main__":
    main()

