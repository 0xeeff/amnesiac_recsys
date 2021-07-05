"""
First run user_vector_incremental to generate user vectors which are stored in a checkpoint path, needed to run this.
"""
import argparse
import os
import random
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack

from dataset import Dataset
from metrics import get_precision_recall_fscore, get_ndcg, has_hit_in_top_k


def predict(checkpoint, num_neighbours=200, alpha=0.5, topn=20, test_split=0.1):
    # load user vectors
    user_vector_file = [file for file in os.listdir(checkpoint) if "user" in file]
    user_vector_dict = {}
    customer_ids = []
    for uf in user_vector_file:
        cust_id = int(uf.split("__")[1])  # for later sorting integer
        user_vector = joblib.load(os.path.join(checkpoint, uf))["mean"]
        customer_ids.append(cust_id)
        user_vector_dict[cust_id] = user_vector
    customer_ids.sort()
    total_customer = len(customer_ids)
    seed = 1027
    random.seed(seed)  # make sure we have the same sample [1025, 1026,1027]
    print("random seed:", seed)
    # these customer only have 1 basket, hence no future prefiction label
    customer_ids_copy = customer_ids.copy()
    customer_ids_copy.remove(2939)
    test_customer_ids = random.sample(customer_ids_copy, int(test_split * len(customer_ids_copy)))  # we sample 10% from all
    test_customer_ids.sort()
    # training_customer_ids = customer_ids[0: int((1-test_split) * total_customer)]
    # test_customer_ids = customer_ids[int((1-test_split) * total_customer):]

    training_user_mat = vstack([user_vector_dict[cid] for cid in customer_ids])
    # all_user_mat = vstack([user_vector_dict[cid] for cid in customer_ids])

    test_user_mat = vstack([user_vector_dict[cid] for cid in test_customer_ids])
    print(f"Num of training users: {len(customer_ids)}")
    print(f"Num of test users: {len(test_customer_ids)}")
    print("Searching for neighbors for test users in training users...")
    # find neighbors
    # NOTE: we deliberately limit the search space to include only subset instead of all users
    # arguably, searching in all users should boost the performance
    # we do the training test split to be consistent with the original paper.
    nbrs = NearestNeighbors(n_neighbors=num_neighbours, algorithm='brute').fit(training_user_mat)
    # the indices are row number in all_user_mat, note it includes the test user itself!
    distances, indices = nbrs.kneighbors(test_user_mat)

    final_prediction_vector = dict()
    for index in indices:
        neighbours = training_user_mat[index[1:], :]
        neighbor_mean = np.mean(neighbours, axis=0)
        test_cid = index[0] + 1
        pred_m = alpha * user_vector_dict[test_cid] + (1 - alpha) * neighbor_mean
        # somehow the mean is a numpy matrix, we convert it to ndarray and flatten it
        final_prediction_vector[test_cid] = pred_m.A.flatten()
    # now we just need to sort the prediction vector to arrive at the final top n predictions
    topn_recommendations = dict()
    for cid, pred_v in final_prediction_vector.items():
        topn_items = pred_v.argsort()[::-1][:topn] + 1
        # note that we convert both customer id and items ids to str to later evaluation
        topn_recommendations[str(cid)] = topn_items.astype("str")
    return topn_recommendations


def evaluate(predictions, ground_truth, topn):
    all_precisions = []
    all_recalls = []
    all_fscores = []
    all_ndcgs = []
    all_hits = 0
    for uid, predict_item_ids in predictions.items():
        # note that future basket are stored as list of 1 basket, hence the extra index 0
        target_item_ids = ground_truth[uid]  # these contain item IDs counting from 1!

        precision, recall, fscore, _ = get_precision_recall_fscore(target_item_ids, predict_item_ids, topn)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_fscores.append(fscore)

        ndcg = get_ndcg(target_item_ids, predict_item_ids, topn)
        all_ndcgs.append(ndcg)
        all_hits += has_hit_in_top_k(target_item_ids, predict_item_ids, topn)

    recall = np.mean(all_recalls)
    precision = np.mean(all_precisions)
    fscore = np.mean(all_fscores)
    ndcg = np.mean(all_ndcgs)
    hit_ratio = all_hits / len(predictions)
    return recall, precision, fscore, ndcg, hit_ratio


def main(args):
    user_vector_checkpoint = args.checkpoint
    ground_truth_path = args.ground_truth_path
    topn = args.topn
    num_neighbours = args.num_neighbours
    alpha = args.alpha
    test_split = args.test_split  # test on this fraction of all users
    print("Started running prediction...")
    predictions = predict(user_vector_checkpoint,
                          num_neighbours=num_neighbours,
                          alpha=alpha,
                          topn=topn,
                          test_split=test_split)
    # getting the ground truth
    ground_truth_data = Dataset()
    ground_truth_data.load_from_file(ground_truth_path)
    ground_truth = {cid: array[0][1] for cid, array in ground_truth_data.customer_baskets.items()}
    # evaluate metrics
    print("Started evaluating...")
    topn= 10
    recall, precision, fscore, ndcg, hit_ratio = evaluate(predictions=predictions,
                                                          ground_truth=ground_truth,
                                                          topn=topn)
    print(f'top n: {topn}')
    # print(f'precision@{topn}:{precision}')
    print(f'recall@{topn}:{recall}')
    print(f'ndcg@{topn}: {ndcg}')
    # print(f'hit ratio: {hit_ratio}')
    topn= 20
    recall, precision, fscore, ndcg, hit_ratio = evaluate(predictions=predictions,
                                                          ground_truth=ground_truth,
                                                          topn=topn)
    print(f'top n: {topn}')
    # print(f'precision@{topn}:{precision}')
    print(f'recall@{topn}:{recall}')
    print(f'ndcg@{topn}: {ndcg}')
    # print(f'hit ratio: {hit_ratio}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ground_truth_path", default="./data/TaFang_future_NB.csv")
    parser.add_argument("--checkpoint", default="TaFang_checkpoint/dec_1k2rg_0.7_rb_0.9_gs_7", help="checkpoint path")
    parser.add_argument("--num_neighbours", default=300, type=int, help="the size of a group")
    parser.add_argument("--topn", default=20, type=int, help="the size of a group")
    parser.add_argument("--alpha", default=0.7, type=float, help="clear path")
    parser.add_argument("--test_split", default=1.0, type=float, help="for single file which does not contain all")
    args = parser.parse_args()
    print(args)
    main(args)
