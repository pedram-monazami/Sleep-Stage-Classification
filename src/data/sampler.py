import random
from collections import defaultdict
from torch.utils.data import Sampler


class BalancedSubsetSampler(Sampler):
    def __init__(self, dataset, samples_per_class=6000, sequenced=False):
        super().__init__()
        self.dataset = dataset
        self.samples_per_class = samples_per_class

        self.class_indices = defaultdict(list)
        if sequenced:
            for idx, (_, _, _, label) in enumerate(dataset.sequences):
                self.class_indices[label].append(idx)
        else:
            for idx, label in enumerate(dataset.labels):
                self.class_indices[label].append(idx)

        # Shuffle each class list once initially
        for cls in self.class_indices:
            random.shuffle(self.class_indices[cls])

        self.class_pointers = {cls: 0 for cls in self.class_indices}

    def __iter__(self):
        indices = []

        for cls, cls_idxs in self.class_indices.items():
            start = self.class_pointers[cls]
            end = start + self.samples_per_class

            # If not enough left, reshuffle and start over
            if end > len(cls_idxs):
                random.shuffle(cls_idxs)
                start = 0
                end = self.samples_per_class

            batch = cls_idxs[start:end]
            self.class_pointers[cls] = end
            indices.extend(batch)

        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_class * len(self.class_indices)

    def state_dict(self):
        return {
            'class_indices': self.class_indices,
            'class_pointers': self.class_pointers,
        }

    def load_state_dict(self, state):
        self.class_indices = state['class_indices']
        self.class_pointers = state['class_pointers']
