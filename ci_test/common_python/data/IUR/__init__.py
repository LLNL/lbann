import random
import numpy as np
import json
from tqdm import tqdm


class IUR_Dataset:
    def __init__(
        self, data_path, num_sample_per_author=4, episode_length=16, max_token_length=32
    ):

        self.dataset_path = data_path
        self.num_sample_per_author = num_sample_per_author
        self.episode_length = episode_length
        self.max_token_length = max_token_length
        self.load_data()

    def load_data(self):
        print("Loading dataset file: {}".format(self.dataset_path))

        with open(self.dataset_path) as f:
            data = [json.loads(l.strip()) for l in tqdm(f.readlines())]

        feats = [f for f in data[0].keys() if f != "author_id"]

        self.num_authors = len(data)
        self.author_id = [d["author_id"] for d in data]
        self.data = {f: [d[f] for d in data] for f in feats}
        self.num_docs = [len(d) for d in self.data["syms"]]

    def sample_random_episode(self, index, episode_length):
        maxval = self.num_docs[index] - episode_length
        start_index = random.randint(0, maxval)

        episode = {
            k: v[index][start_index : start_index + episode_length]
            for k, v in self.data.items()
            if len(v) > 0
        }

        episode["author_id"] = self.author_id[index]

        return episode

    def __len__(self):
        return self.num_authors

    def sample_size(self):
        return 1 + self.max_token_length * self.episode_length * 2

    def __getitem__(self, index):

        sample_size = min(self.episode_length, self.num_docs[index])

        episode = self.sample_random_episode(index, sample_size)
        input_ids = episode["syms_input_ids"]
        attn_mask = episode["syms_attention_mask"]

        # pad and truncate
        input_ids = [
            x[: self.max_token_length] + [-1] * max(0, self.max_token_length - len(x))
            for x in input_ids
        ]
        attn_mask = [
            x[: self.max_token_length] + [0] * max(0, self.max_token_length - len(x))
            for x in attn_mask
        ]

        author_id = episode["author_id"]

        input_ids = np.array(input_ids).reshape(1, -1).flatten()
        attn_mask = np.array(attn_mask).reshape(1, -1).flatten()

        return np.concatenate((np.array([author_id]), input_ids, attn_mask))


train_data = IUR_Dataset(
    "/p/vast1/brain/iur_dataset/bert_tokenization/validation.jsonl"
)


def sample_dims():
    return (train_data.sample_size(),)


def num_train_samples():
    return len(train_data)


def get_train_sample(i):
    return train_data[i]
