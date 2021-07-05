# {"customerId": 1, "orderId": 1, "basket": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "isDeletion": false}
# we need to convert each basket to a json file of the above format
# we also need to generate a vocabulary file csv
import os
import pandas as pd
history_file = "/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn/data/TaFang_history_NB.csv"
future_file = "/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn/data/TaFang_future_NB.csv"
vocab_file_name = "tafeng_vocab.csv"
base_path = "/Users/longxiang/devel/github/UvAAI/thesis_ai/thesis/tifuknn/data/json_tafeng"
vocab_path = os.path.join(base_path, vocab_file_name)

# 1. generate vocaburary
raw_df = pd.read_csv(history_file)
# get all itemIds and store into csv file

# 2. convert all baskets into single json files






