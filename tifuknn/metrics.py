import math


def get_ndcg(ideal_rank_list, pred_rank_list, k):
    """
    item_relevance_score containing relevance score for each item.
    It is a q-encoded vector of size (number of items)
    where bought items are set to 1 representing the relevance score
    [1, 0, 1...] would mean item 1 and 3 are bought in the basket
    pred_rank_list contains the index of items that are predicted
    such as [0, 1,3] would mean we predict item 1 and 2 and 4

    """
    # note that we are dealing with implicit dataset
    # the relevance is 1 for items bought, 0 otherwise
    # see wiki: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    k = min(k, len(ideal_rank_list))
    dcg = 0
    for idx, pred in enumerate(pred_rank_list[:k]):
        gain = 1 if pred in ideal_rank_list else 0
        dcg += gain / math.log2(idx + 1 + 1)
    # the ideal DCG is when all items are bought, regardless of k
    # one can also argue we should consider up to k, like in dcg
    idcg = 0
    for i in range(k):
        idcg += 1 / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg


def get_precision_recall_fscore(ground_truth, prediction, topn):
    prediction = prediction[:topn]
    num_all_target_items = len(ground_truth)
    num_all_predict_items = len(prediction)
    num_correct_predictions = 0
    for p in prediction:
        if p in ground_truth:
            num_correct_predictions += 1
    precision = num_correct_predictions / num_all_predict_items
    recall = num_correct_predictions / num_all_target_items
    if precision + recall == 0:
        # avoid zero division when both p and r are zeros
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore, num_correct_predictions


def has_hit_in_top_k(ground_truth, prediction, k):
    """
    This function checks if the prediction has hit a ground truth in its top k predictions.
    """
    for p in prediction[:k]:
        if p in ground_truth:
            return 1
    return 0
