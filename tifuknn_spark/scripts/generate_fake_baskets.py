import random
import json
import sys
# {"customerId": 1,"orderId": 1, "basket": [1,2,3], "isDeletion": false}
from time import sleep

target_dir = "../jsondata/baskets10"

customerId = 1

# vocab = [1, 2, 3, 4]
vocab = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

isDeletion = False

upperRange = sys.argv[1]

for orderId in range(int(upperRange)):
    # generate a
    # basket = random.sample(vocab, random.randint(1, 4))
    basket = vocab  # always full rank
    mydict = {"customerId": customerId,
              "orderId": 1,
              "basket": basket,
              "isDeletion": isDeletion
              }
    file = f"{target_dir}/basket_{orderId}.json"
    print(f"Saving file {file}...")
    with open(file, "w") as fp:
        json.dump(mydict, fp)
    sleep(0.1)
