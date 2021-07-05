import random
import json
import sys
from time import sleep
# {"customerId": 1,"orderId": 1, "basket": [1,2,3], "isDeletion": false}

target_dir = "../jsondata/baskets10"
# target_dir = "./temp"

customerId = 1


isDeletion = True

upperRange = sys.argv[1]

for orderId in range(int(upperRange)):
    # basket = random.sample(vocab, random.randint(1, 4))
    mydict = {"customerId": customerId,
              "orderId": 1,
              "basket": [],
              "isDeletion": isDeletion
              }
    file = f"{target_dir}/basket_deletion_{orderId}.json"
    print(f"saving file {file}...")
    with open(file, "w") as fp:
        json.dump(mydict, fp)
    sleep(0.1)





