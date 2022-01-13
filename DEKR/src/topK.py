import torch
import numpy as np


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def get_dcg(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def get_ndcg(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
    idcg = get_dcg(relevance)
    dcg = get_dcg(rank_scores)
    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg


def topk_eval(model, user_list, train_record, test_record, item_set, k_list, batch_size, device):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        user_test_loader = torch.utils.data.DataLoader(test_item_list, batch_size=batch_size)
        for items in user_test_loader:
            users = [user] * len(items)
            users = torch.tensor(users)
            users_id, items_id = users.to(device), items.to(device)
            outputs_g, outputs_t = model(users_id, items_id)
            outputs = (outputs_g + outputs_t) / 2

            items = items_id.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()
            for item, score in zip(items, outputs):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            user_ndcg = get_ndcg(item_sorted[:k], list(test_record[user]))
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))
            ndcg_list[k].append(user_ndcg)

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg

