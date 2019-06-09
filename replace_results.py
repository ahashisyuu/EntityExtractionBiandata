import pandas as pd


def main():
    train_data = pd.read_csv("./data/event_type_entity_extract_train.csv",
                             header=None,
                             names=["id", "sent", "entity", "label"])
    test_data = pd.read_csv("./filter_data/test_data.csv",
                            header=None,
                            names=["id", "sent", "entity", "label"])
    results = pd.read_csv("results/cv_best_text_results.txt",
                          header=None,
                          names=["id", "pred"])
    id_list = results["id"].tolist()

    count = 0
    not_same_count = 0
    for index in id_list:
        temp = test_data[test_data["id"] == index]
        search_sent = temp["sent"].tolist()[0]
        exist_temp = train_data[train_data["sent"] == search_sent]

        if len(exist_temp["id"].tolist()) > 0:
            inner_search = exist_temp[exist_temp["entity"] == temp["entity"].tolist()[0]]
            # print(inner_search)
            # print(results[results["id"] == index])
            if len(inner_search["id"].tolist()) <= 0:
                continue

            count += 1

            org_label = inner_search["label"].tolist()[0]
            pred_res = results[results["id"] == index]["pred"].tolist()[0]

            if org_label != pred_res:
                print("{:8}, {:20}, {:20}".format(index, org_label, pred_res))
                not_same_count += 1

            results[results["id"] == index]["pred"] = inner_search["label"].tolist()[0]

    results["entity"] = test_data["entity"]

    def process(x):
        _index = x[0]
        _pred = x[1]
        _entity = x[2]
        if _entity == "其他":
            return "NaN"
        return _pred

    results["pred"] = results.apply(process, axis=1)
    results[["id", "pred"]].to_csv("results/cv_post2_results.txt", header=False, index=False)
    print("all covering: ", count)
    print("not same sample: ", not_same_count)


if __name__ == "__main__":
    main()
