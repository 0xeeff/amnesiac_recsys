import csv
import json
import os
import random
import time

import pandas as pd
import numpy as np

from utils import BasketVectorizer


class Dataset:
    def __init__(self):
        self.customer_baskets = dict()
        # this is to store the sparse matrix
        self.customer_baskets_vectorized = dict()
        self.item_ids = None
        self.customer_ids = None

    def __repr__(self):
        res = f"customers: {len(self.customer_ids)}, unique items: {len(self.item_ids)}"
        return res

    def load_from_file(self, file_path):
        # note that it is important to keep order_number as int, so that it is sorted properly, that order is preserved.
        df = pd.read_csv(file_path, dtype={"CUSTOMER_ID": str, "ORDER_NUMBER": int, "MATERIAL_NUMBER": str})
        print(f"number of records in {file_path}: {len(df)}")
        for customer_id, customer_baskets_df in df.groupby("CUSTOMER_ID"):
            self.customer_baskets[customer_id] = []
            for order_id, basket in customer_baskets_df.groupby("ORDER_NUMBER"):
                items_in_basket = basket["MATERIAL_NUMBER"].values
                self.customer_baskets[customer_id].append((order_id, items_in_basket))
        # update other attributes based on customer_baskets
        self._update_stats()

    def generate(self, number_of_baskets_per_customer=10, max_customer_id=10000, max_item_id=1000, basket_size=20):
        print("generating fake dataset")
        self.customer_ids = {cid for cid in range(1, max_customer_id + 1)}
        self.item_ids = [iid for iid in range(1, max_item_id + 1)]
        self.number_of_baskets_per_customer = {cid: number_of_baskets_per_customer for cid in self.customer_ids}
        # we manually partition customers into 3 clusters, 1/3 likes the first 1/3 of items, ...
        for cid in self.customer_ids:
            self.customer_baskets[cid] = []
            if cid <= max_customer_id // 3:
                items_they_like = range(1, max_item_id // 3)
            elif cid <= max_customer_id * 2 // 3:
                items_they_like = range(max_item_id // 3, 2 * max_item_id // 3)
            else:
                items_they_like = range(2 * max_item_id // 3, max_item_id + 1)

            for i in range(number_of_baskets_per_customer):
                basket = np.array([random.choice(items_they_like) for i in range(basket_size)])
                self.customer_baskets[cid].append(basket)

    def _update_stats(self):
        self.number_of_baskets_per_customer = dict()
        customer_ids = set()
        item_ids = set()
        for _cid, _baskets in self.customer_baskets.items():
            customer_ids.add(_cid)
            self.number_of_baskets_per_customer[_cid] = len(_baskets)
            for _order_id, _b in _baskets:
                item_ids.update(_b)
        # we first sort the item ids, then convert it to string to be used as vocabulary
        # note that this is done in two steps to avoid miss sorting, [1,2,3..11,] into [1,11,2,3]
        self.item_ids = sorted(item_ids, key=lambda x: int(x))  # sort a dict will return a list
        self.customer_ids = sorted(customer_ids, key=lambda x: int(x))  # sort a dict will return a list

    def vectorize(self, vocabulary=None):
        if vocabulary is None:
            vocabulary = self.item_ids
        bc = BasketVectorizer(vocabulary=vocabulary)
        for cid in self.customer_ids:
            self.customer_baskets_vectorized[cid] = []
            for order_id, raw_basket in self.customer_baskets[cid]:
                vec_tuple = (order_id, bc.transform([raw_basket], toarray=False)[0])
                self.customer_baskets_vectorized[cid].append(vec_tuple)

    def prune(self, min_baskets_per_customer=2, min_items_per_basket=2):
        """DONT USE THIS YET. USE the prune function in tifuknn"""
        # removing small baskets
        for _cid, _baskets in self.customer_baskets.items():
            left_baskets = []
            for _basket in _baskets:
                if len(_basket) >= min_items_per_basket:
                    left_baskets.append(_basket)
            self.customer_baskets[_cid] = left_baskets
        # remove customer with less than min_baskets_per_customer
        self.customer_baskets = {_cid: _baskets for _cid, _baskets in self.customer_baskets.items()
                                 if len(_baskets) >= min_baskets_per_customer}
        self._update_stats()

    def to_vocab(self, file_path):
        if ds.item_ids is None:
            raise ValueError("no vocab to save, item Ids missing")
        df = pd.DataFrame({"itemId": ds.item_ids})
        df.to_csv(file_path, index=False)

    def to_json_baskets(self, jsondata_path):
        for customerId, list_of_baskets in self.customer_baskets.items():
            # suppose the list_of_baskets is sorted
            for orderId, basket_array in list_of_baskets:
                d_temp = {"customerId": int(customerId), "orderId": orderId,
                          "basket": list(map(lambda x: int(x), basket_array.tolist())), "isDeletion": False}

                fp = os.path.join(jsondata_path, f"customer_{customerId}_order_{orderId}.csv")
                json.dump(d_temp, open(fp, "w"))
                print(f"Writing {fp} ...")
                time.sleep(0.1)

if __name__ == '__main__':
    ds = Dataset()
    ds.load_from_file("/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn/data/TaFang_history_NB.csv")
    ds.to_vocab("/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn_spark/jsondata/tafang_vocab.csv")
    ds.to_json_baskets("/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn_spark/jsondata/tafang")
    print(ds)
