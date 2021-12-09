from collections import Counter
from torch.utils import data
from typing import Dict, Tuple

Mapping = Dict[str, int]


def create_mappings(dataset_path: str) -> Tuple[Mapping, Mapping]:
    """Creates separate mappings to indices for entities and relations."""
    # counters to have entities/relations sorted from most frequent
    entity_counter = Counter()
    relation_counter = Counter()
    with open(dataset_path, "r") as f:
        for line in f:
            # -1 to remove newline sign
            head, relation, tail = line[:-1].split("\t")
            entity_counter.update([head, tail])
            relation_counter.update([relation])
    entity2id = {}
    relation2id = {}
    for idx, (mid, _) in enumerate(entity_counter.most_common()):
        entity2id[mid] = idx+1 
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx+1
    return entity2id, relation2id

def load_data(path, reverse=False):
    with open(path, "r", encoding="utf-8") as f:
        data = [line[:-1].split("\t") for line in f]
        if reverse:
            data += [[i[2], i[1], i[0]] for i in data]
    return data


class FB15KDataset(data.Dataset):
    """Dataset implementation for handling FB15K and FB15K-237."""

    def __init__(self, data, entity2id: Mapping, relation2id: Mapping):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.data= data

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        head, relation, tail = self.data[index]
        head_id = self._to_idx(head, self.entity2id)
        relation_id = self._to_idx(relation, self.relation2id)
        tail_id = self._to_idx(tail, self.entity2id)
        return head_id, relation_id, tail_id

    @staticmethod
    def _to_idx(key: str, mapping: Mapping) -> int:
        try:
            return mapping[key]
        except KeyError:
            return 0
