import json
import numpy as np
import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token =  ["[SEP]"]
        self.repo_root = Path(data_path).resolve().parents[2]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def _resolve_image_path(self, img_path):
        candidate = Path(os.path.expanduser(img_path))
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if candidate.exists():
            return candidate

        repo_relative = self.repo_root / candidate
        if repo_relative.exists():
            return repo_relative

        data_relative = Path(self.data_dir) / candidate
        if data_relative.exists():
            return data_relative

        return candidate

    def __getitem__(self, index):
        sentence = (
            self.text_start_token
            + self.tokenizer(self.data[index]["text"])[
                : (self.max_seq_len - 1)
            ] + self.text_start_token
        )
        segment = torch.zeros(len(sentence))
        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            if self.data[index]["label"] == '':
                self.data[index]["label"] = "'Others'"
            else:
                pass  
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"].split(', ')]
            ] = 1
        else:
            pass

        image = None
        if self.data[index]["img"]:
            image = Image.open(self._resolve_image_path(self.data[index]["img"]))
        else:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        image = self.transforms(image)

        # The first SEP is part of Image Token.
        segment = segment[1:]
        sentence = sentence[1:]
        # The first segment (0) is of images.
        segment += 1

        return sentence, segment, image, label
