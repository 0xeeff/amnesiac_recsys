"""
An toy demo showing how to incrementally/decremental update a decayed mean of stream
"""
import os
import sys
from collections import deque

import joblib
import numpy as np
import scipy
from scipy.sparse import csr_matrix

from mystream.utils import get_first_order_diff_decaying_sum
from state import State


class MeanStreamAppBase:
    def __init__(self, checkpoint_file_path, decay_rate=1.0):
        # we use a deque as our history memory, it has a nice feature of maxlen, older items are automatically
        # removed, once the max size if reached
        self.checkpoint = checkpoint_file_path + ".memory"
        self.memory_state = State(file_path=self.checkpoint, schema={"mean": 0.0,
                                                                     "count": 0,
                                                                     "ids": set(),
                                                                     "decay_rate": decay_rate,
                                                                     "memory": deque()})
        self.memory_state.initialize()
        if self.memory_state.is_loaded_from_checkpoint:
            existed_decay = self.memory_state.values["decay_rate"]
            if existed_decay != decay_rate:
                raise ValueError(f"decay rate from checkpoint {existed_decay} not equal provided {decay_rate}")
        self.decay_rate = decay_rate

    def start(self):
        while True:
            print("state values:", self.memory_state.values)
            try:
                input_str = input()
                mode, identifier, value_or_index = input_str.split(" ")
            except Exception as e:
                print(e)
                continue
            if mode == "add":
                # we expect an list and convert to numpy array for easy vector calculation, copy this [0,0,1]
                element = eval(value_or_index)
                order_id = eval(identifier)
                if not isinstance(element, list):
                    print("add only element list type e.g. [1], or [0,1,1]")
                    continue
                element = np.array(element)
                self.increment(order_id, element)
            if mode == "rm":
                order_id = int(value_or_index)
                self.decrement(order_id)

    def decrement(self, order_id):
        order_ids_memory = self.memory_state.values["ids"]
        if order_id not in order_ids_memory:
            print(f"{order_id} does not exist in memory")
            return

        current_mem = list(self.memory_state.values["memory"])
        for idx, x in enumerate(current_mem):
            if x[0] == order_id:
                index = idx
        # we get only the array itself without the array id
        array_after_k = [x[1] for x in current_mem][index:]
        # we convert it to an numpy array
        if isinstance(array_after_k[0], csr_matrix):
            # get_first_order_diff_decaying_sum only works with dense matrix
            array_after_k = scipy.sparse.vstack(array_after_k).toarray()
        else:
            array_after_k = np.array(array_after_k)
        decay_sum = get_first_order_diff_decaying_sum(array_after_k, self.decay_rate)
        decay_sum = csr_matrix(decay_sum)  # convert to csr_matrix to match the new_mean calculation
        prev_mean = self.memory_state.values["mean"]
        prev_count = self.memory_state.values["count"]
        new_count = prev_count - 1
        order_ids_memory.remove(order_id)
        if new_count > 0:
            new_mean = prev_mean * prev_count / (new_count * self.decay_rate) + decay_sum / (
                    new_count * self.decay_rate)
        else:
            new_mean = 0
        # update states
        new_memory = self.memory_state.values["memory"].copy()
        # remove the element from the deque
        del new_memory[index]

        new_state = {"mean": new_mean,
                     "count": new_count,
                     "ids": order_ids_memory,
                     "decay_rate": self.decay_rate,
                     "memory": new_memory}
        self.memory_state.update(new_state)

    def print_res(self):
        print("state values:", self.memory_state.values)

    def update_memory_state(self, update_dict: dict) -> None:
        order_ids_memory = self.memory_state.values["ids"].copy()
        mean = self.memory_state.values["mean"]
        count = self.memory_state.values["count"]
        decay_rate = self.memory_state.values["decay_rate"]
        memory = self.memory_state.values["memory"].copy()

        for key, val in update_dict.items():
            if key == "mean":
                mean = val
            elif key == "count":
                count = val
            elif key == "memory":
                memory = val
            elif key == "ids":
                order_ids_memory = val
        new_state = {"mean": mean,
                     "count": count,
                     "decay_rate": decay_rate,
                     "ids": order_ids_memory,
                     "memory": memory}
        self.memory_state.update(new_state)

    def increment(self, order_id, new_array):
        order_ids_memory = self.memory_state.values["ids"]
        if order_id in order_ids_memory:
            print(f"incremental order id: {order_id} is already included, thus skipped")
            return
        prev_mean = self.memory_state.values["mean"]
        prev_count = self.memory_state.values["count"]

        new_count = prev_count + 1
        new_mean = prev_mean * prev_count * self.decay_rate / new_count + new_array / new_count
        order_ids_memory.add(order_id)
        # update states
        new_memory = self.memory_state.values["memory"].copy()
        # do we need to spill over, because the buffer is full?
        new_memory.append((order_id, new_array))
        new_state = {"mean": new_mean,
                     "count": new_count,
                     "ids": order_ids_memory,
                     "decay_rate": self.decay_rate,
                     "memory": new_memory}

        self.memory_state.update(new_state)


if __name__ == '__main__':
    msa = MeanStreamAppBase("msa_decay_checkpoint/memory_test_base2", decay_rate=0.5)
    print(msa)
    msa.start()
