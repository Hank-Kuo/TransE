from collections import Counter
from typing import Dict, Tuple
entity_counter = Counter()
relation_counter = Counter()

with open('./train.txt', "r") as f:
    for line in f:
        # -1 to remove newline sign
        head, relation, tail = line[:-1].split("\t")
        entity_counter.update([head, tail])
        relation_counter.update([relation])
entity2id = {}
relation2id = {}
for idx, (mid, _) in enumerate(entity_counter.most_common()):
    entity2id[mid] = idx
for idx, (relation, _) in enumerate(relation_counter.most_common()):
    relation2id[relation] = idx

print(entity2id, relation2id)