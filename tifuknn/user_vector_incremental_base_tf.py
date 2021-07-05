"""
Run this to generate user vectors in incremental fashion.
"""
import argparse
import json
import math
import os
import random
import shutil

import numpy as np
import tqdm
from scipy.sparse import csr_matrix
from typing import List, Dict

from dataset import Dataset
from meanofstream_base import MeanStreamAppBase

random.seed(11)  # 10, 11,12

VOCAB = [str(i) for i in range(1, 5)]  # for faker dataset
VOCAB = [str(i) for i in range(1, 11998)]  # for tafang dataset

IS_DEC = True


def print_user(new_user_v):
    if isinstance(new_user_v, csr_matrix):
        print(f"the new user vector is {new_user_v.toarray()}")
    else:
        # it might be integer 0 if all baskets are gone
        print(f"the new user vector is {new_user_v}")


def increment_user_vector(customer_baskets: Dict[str, list],
                          customer_ids: List[str],
                          group_size: int,
                          basket_decay_rate: float,
                          group_decay_rate: float,
                          checkpoint: str) -> Dict[str, csr_matrix]:
    """
    This is according to the original paper. The provided code by the author does not comply
    with their own paper..
    """
    customer_vectors = {}
    inner_path = f"rg_{group_decay_rate}_rb_{basket_decay_rate}_gs_{group_size}"
    os.makedirs(f"{checkpoint}/{inner_path}", exist_ok=True)
    if IS_DEC:
        checkpoint = f"{checkpoint}/dec_1k2{inner_path}"
    else:
        checkpoint = f"{checkpoint}/{inner_path}"
    random.shuffle(customer_ids)
    removed_customer_count = 0
    remove_basket_customer_ids = random.sample(customer_ids, int(0.001 * len(customer_ids)))  # we sample 0.1% from all
    print("sample dec baskets length:", len(remove_basket_customer_ids))
    for customer_id in tqdm.tqdm(customer_ids, desc="Creating user vector"):

        list_of_basket_vetorized = customer_baskets[customer_id]
        if IS_DEC and removed_customer_count <= len(remove_basket_customer_ids):
            remove_count = int(0.1 * len(list_of_basket_vetorized))
            print("prepare to remove basket for customer: ", customer_id)
            if remove_count >= 1:
                removed_customer_count += 1
                remove_indice = random.sample(range(len(list_of_basket_vetorized)), remove_count)
                print(f"removing {remove_indice} baskets...")
                temp = [list_of_basket_vetorized[i] for i in range(len(list_of_basket_vetorized)) if
                        i not in remove_indice]
                list_of_basket_vetorized = temp

        # divide basket vectors into groups and average to get group vectors
        for order_id, basket_vector in list_of_basket_vetorized:
            msa_user_vector = MeanStreamAppBase(f"{checkpoint}/customer_id__{customer_id}__user_vector",
                                                decay_rate=group_decay_rate)

            # print_user(msa_user_vector.memory_state.values['mean'])
            prev_vu = msa_user_vector.memory_state.values["mean"]
            # from user vector state, we can get the group vector ids
            group_ids_in_memory = msa_user_vector.memory_state.values["ids"]
            if len(group_ids_in_memory) == 0:
                latest_group_id = 1
                msag = MeanStreamAppBase(f"{checkpoint}/customer_id__{customer_id}__group_vector__{latest_group_id}",
                                         decay_rate=basket_decay_rate)
                msag.increment(order_id, basket_vector)
                new_additional_vg = msag.memory_state.values["mean"]
                msa_user_vector.increment(latest_group_id, new_additional_vg)
                final_user_vector = msa_user_vector.memory_state.values['mean']
                # print_user(final_user_vector)
            else:
                latest_group_id = max(group_ids_in_memory)
                msag = MeanStreamAppBase(f"{checkpoint}/customer_id__{customer_id}__group_vector__{latest_group_id}",
                                         decay_rate=basket_decay_rate)
                num_of_baskets = msag.memory_state.values["count"]
                if num_of_baskets < group_size:
                    # there is still space left for the last group
                    prev_vg = msag.memory_state.values["mean"]
                    msag.increment(order_id, basket_vector)
                    updated_vg = msag.memory_state.values["mean"]
                    final_user_vector = prev_vu + (updated_vg - prev_vg) / len(group_ids_in_memory)
                    # print_user(final_user_vector)
                    # update the state
                    user_vector_mem = msa_user_vector.memory_state.values["memory"].copy()
                    mean = final_user_vector
                    user_vector_mem.pop()
                    user_vector_mem.append((latest_group_id, updated_vg))
                    msa_user_vector.update_memory_state(update_dict={"mean": mean,
                                                                     "memory": user_vector_mem
                                                                     })

                else:
                    # we will need to add a new group with this one basket
                    msag = MeanStreamAppBase(
                        f"{checkpoint}/customer_id__{customer_id}__group_vector__{latest_group_id + 1}",
                        decay_rate=basket_decay_rate)
                    msag.increment(order_id, basket_vector)
                    new_additional_vg = msag.memory_state.values["mean"]
                    msa_user_vector.increment(latest_group_id + 1, new_additional_vg)
                    final_user_vector = msa_user_vector.memory_state.values['mean']
                    # print_user(final_user_vector)

        customer_vectors[customer_id] = final_user_vector
        # save the customer vector in a json file
        # json.dump(customer_vectors, open(f"user_vector_store/tafeng_rg_{group_decay_rate}_rb_{basket_decay_rate}_gs_{group_size}.json", "w"))
    return customer_vectors


def main_runner(args):
    print("Started.")
    checkpoint = args.checkpoint
    if args.clear_state:
        if os.path.exists(checkpoint):
            shutil.rmtree(checkpoint)
    data_file = args.data_file
    group_size = args.group_size
    basket_decay_rate = args.basket_decay_rate
    group_decay_rate = args.group_decay_rate
    use_external_vocab = args.use_external_vocab
    dataset = Dataset()
    # loading real data
    dataset.load_from_file(data_file)
    # convert baskets to one hot encoded
    if use_external_vocab:
        dataset.vectorize(vocabulary=VOCAB)
    else:
        dataset.vectorize(vocabulary=dataset.item_ids)

    print(dataset)

    customer_vectors_training = increment_user_vector(
        customer_baskets=dataset.customer_baskets_vectorized,
        customer_ids=dataset.customer_ids,
        group_size=group_size,
        basket_decay_rate=basket_decay_rate,
        group_decay_rate=group_decay_rate,
        checkpoint=checkpoint)

    # for cid, user_vector in customer_vectors_training.items():
    #     print(cid)
    #     print(user_vector)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--basket_decay_rate", default=0.9, type=float, help="time decay ratio within a group")
    parser.add_argument("--group_decay_rate", default=0.7, type=float, help="time decay ratio across groups")
    parser.add_argument("--data_file", help="historical baskets", default="./data/TaFang_history_NB.csv")
    parser.add_argument("--group_size", default=7, type=int, help="the size of a group")
    parser.add_argument("--checkpoint", default="TaFang_checkpoint", help="checkpoint path")
    parser.add_argument("--clear_state", default=False, type=bool, help="clear path")
    parser.add_argument("--use_external_vocab", default=False, type=bool,
                        help="for single file which does not contain all")
    args = parser.parse_args()
    print(args)
    main_runner(args)
