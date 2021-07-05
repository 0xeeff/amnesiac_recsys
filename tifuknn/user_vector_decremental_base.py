"""
Run this to forget about some baskets for a user.
Prerequisite: user vectors are present by the incremental_base script
"""
import collections

from scipy.sparse import csr_matrix

from user_vector_incremental_base import MeanStreamAppBase


# TODO: Add docstrings
# TODO: Add some unit tests

def print_user(new_user_v):
    print(new_user_v)
    if isinstance(new_user_v, csr_matrix):
        print(f"the new user vector is {new_user_v.toarray()}")
    else:
        # it might be integer 0 if all baskets are gone
        print(f"the new user vector is {new_user_v}")


def decrement_user_vector(checkpoint, customer_id, basket_id, group_decay_rate, basket_decay_rate):
    # TODO: persist group_decay_rate, memory_size to state
    msau = MeanStreamAppBase(f"{checkpoint}/customer_id__{customer_id}__user_vector",
                             decay_rate=group_decay_rate)

    # from user vector state, we can get the group vector ids
    group_ids_in_memory = msau.memory_state.values["ids"]
    print_user(msau.memory_state.values['mean'])
    prev_vu = msau.memory_state.values["mean"]
    num_groups = len(group_ids_in_memory)
    # now how to find our which group vector this basket id belongs to ?
    for group_id in group_ids_in_memory:
        msag = MeanStreamAppBase(f"{checkpoint}/customer_id__{customer_id}__group_vector__{group_id}",
                                 decay_rate=basket_decay_rate)
        basket_ids_in_memory = msag.memory_state.values["ids"]
        if basket_id in basket_ids_in_memory:
            print(f"decrementing basket id {basket_id} found in group vector {group_id}")
            prev_vg = msag.memory_state.values["mean"]  # this is a csr matrix, we cache it for usr vector update later
            # now let's remove the basket from the group vector
            msag.decrement(basket_id)
            updated_vg = msag.memory_state.values["mean"]
            # if the vg is csr matrix, meaning it is an inplace update, otherwize it has vanished
            if isinstance(updated_vg, csr_matrix):
                # mean there is a updated group vector, we use the user vector update rule (eq 46 atm)
                temp_group_idx = group_id - (max(
                    group_ids_in_memory) - num_groups)  ## TODO: this is a bug for not reindexing group after group indexing
                updated_vu = prev_vu + pow(group_decay_rate, num_groups - temp_group_idx) * (
                        updated_vg - prev_vg) / num_groups
                # we need to update the new user vector as well as the group vector in its memory

                memory_list = list(msau.memory_state.values["memory"].copy())
                for idx, (oid, ele) in enumerate(memory_list):
                    if oid == group_id:
                        memory_list.pop(idx)
                        memory_list.insert(idx, (group_id, updated_vg))
                new_memory = collections.deque(memory_list)
                msau.update_memory_state(update_dict={"mean": updated_vu, "memory": new_memory})
            else:
                # ie, the group vector contains only the basket, deleting it causing the vg to vanish
                msau.decrement(group_id)
            new_user_v = msau.memory_state.values['mean']
            print_user(new_user_v)
            break
        else:
            print(f"basket id {basket_id} not found in group vector {group_id}")
        new_user_v = msau.memory_state.values['mean']
        print_user(new_user_v)


if __name__ == '__main__':
    chckpoint = "faker4_all_tafang"
    group_decay_rate = 0.7
    basket_decay_rate = 0.9
    decrement_user_vector(checkpoint=chckpoint,
                          customer_id=1,
                          basket_id=13,
                          group_decay_rate=group_decay_rate,
                          basket_decay_rate=basket_decay_rate
                          )
