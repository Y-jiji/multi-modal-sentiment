import torch as t
import torchvision as tv
import torch.utils.data as td
import transformers as tfm
import pandas as pd
import config
from sklearn.model_selection import train_test_split
t.manual_seed(15721)

class TrainingSet(td.Dataset):
    def __init__(self, meta: pd.DataFrame, device='cuda:0') -> None:
        super().__init__()
        self.meta = meta
        self.meta["tag"] = self.meta["tag"].map(lambda x: {"positive": 0, "neutral": 1, "negative": 2}[x])
        self.image_func = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
        ])
        self.tokenizer = tfm.AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.device = device

    def __getitem__(self, index):
        uid = self.meta["guid"][index]
        img = self.image_func(tv.io.read_image(f'data/{uid}.jpg'))
        with open(f"data/{uid}.txt", encoding="gbk", errors="surrogateescape") as f:
            txt = f.read().strip()
        return img, txt, self.meta["tag"][index], uid

    def __len__(self):
        return self.meta.__len__()

    def to_dataloader(self, batch_size, shuffle=False):
        def collate_fn(input):
            img = t.stack([i[0] for i in input]).to(self.device)
            txt = self.tokenizer([i[1] for i in input], padding=True, return_tensors='pt').to(self.device)
            tag = t.tensor([i[2] for i in input]).to(self.device)
            uid = t.tensor([i[3] for i in input]).to(self.device)
            return (img/255), txt, tag, uid
        return td.DataLoader(self, batch_size, shuffle, collate_fn=collate_fn)

    @staticmethod
    def load():
        meta = pd.read_csv("meta/train.txt")
        return TrainingSet(meta)

    @staticmethod
    def load_two(frac = 0.2):
        meta = pd.read_csv("meta/train.txt")
        train, test = train_test_split(meta, test_size=frac, random_state=15721)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        return TrainingSet(train), TrainingSet(test)

class TestSet(td.Dataset):
    def __init__(self, meta: pd.DataFrame, device='cuda:0') -> None:
        super().__init__()
        self.meta = meta
        self.image_func = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
        ])
        self.tokenizer = tfm.AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.device = device

    def __getitem__(self, index):
        uid = self.meta["guid"][index]
        img = self.image_func(tv.io.read_image(f'data/{uid}.jpg'))
        with open(f"data/{uid}.txt", encoding="gbk", errors="surrogateescape") as f:
            txt = f.read().strip()
        return img, txt, uid

    def __len__(self):
        return self.meta.__len__()

    def to_dataloader(self, batch_size, shuffle=False):
        def collate_fn(input):
            img = t.stack([i[0] for i in input]).to(self.device)
            txt = self.tokenizer([i[1] for i in input], padding=True, return_tensors='pt').to(self.device)
            uid = t.tensor([i[2] for i in input]).to(self.device)
            return (img/255), txt, uid
        return td.DataLoader(self, batch_size, shuffle, collate_fn=collate_fn)

    @staticmethod
    def load():
        meta = pd.read_csv("meta/test.txt")
        return TestSet(meta)
