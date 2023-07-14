import torch as t
import torchvision as tv
import transformers as tfm
import warnings
import config
warnings.simplefilter("ignore")
t.manual_seed(15721)

class ResidualFC(t.nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.fc = t.nn.Sequential(
            t.nn.Linear(dim, dim),
            t.nn.LeakyReLU(),
            t.nn.Linear(dim, dim),
            t.nn.LeakyReLU(),
            t.nn.Linear(dim, dim),
            t.nn.LeakyReLU(),
            t.nn.Linear(dim, dim),
            t.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.fc(x) + x

class PostAGG(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        TXT_DIM = 768
        IMG_DIM = 2048
        self.txt_model = tfm.AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model = t.nn.Sequential(
            *list(tv.models.resnet50(pretrained=True).children())[:-1])
        self.txt_layer = t.nn.TransformerEncoder(t.nn.TransformerEncoderLayer(
            TXT_DIM, 4, batch_first=True, norm_first=True), 6)
        self.fc = t.nn.Sequential(
            ResidualFC(TXT_DIM+IMG_DIM),
            ResidualFC(TXT_DIM+IMG_DIM),
            t.nn.Linear(TXT_DIM+IMG_DIM, 3),
            t.nn.Softmax(dim=-1),
        )

    def forward(self, img, txt):
        with t.no_grad():
            mask = ~txt['attention_mask'].to(t.bool)
            img = self.img_model(img).flatten(-3, -1)
            txt = self.txt_model(**txt).last_hidden_state
        txt = self.txt_layer(txt, src_key_padding_mask=mask)
        txt = (txt * (~mask).unsqueeze(dim=-1)).sum(dim=-2) / ((~mask).sum(dim=-1).unsqueeze(-1) + 1e-10)
        txt = txt + t.rand_like(txt)
        feature = t.concat([txt, img], dim=-1)
        return self.fc(feature)

class OnlyText(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        TXT_DIM = 768
        IMG_DIM = 2048
        self.txt_model = tfm.AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model = t.nn.Sequential(
            *list(tv.models.resnet50(pretrained=True).children())[:-1])
        self.txt_layer = t.nn.TransformerEncoder(t.nn.TransformerEncoderLayer(
            TXT_DIM, 4, batch_first=True, norm_first=True), 6)
        self.fc = t.nn.Sequential(
            ResidualFC(TXT_DIM+IMG_DIM),
            ResidualFC(TXT_DIM+IMG_DIM),
            t.nn.Linear(TXT_DIM+IMG_DIM, 3),
            t.nn.Softmax(dim=-1),
        )

    def forward(self, img, txt):
        with t.no_grad():
            mask = ~txt['attention_mask'].to(t.bool)
            img = self.img_model(t.ones_like(img)).flatten(-3, -1)
            txt = self.txt_model(**txt).last_hidden_state
        txt = self.txt_layer(txt, src_key_padding_mask=mask).mean(dim=-2)
        feature = t.concat([txt, img], dim=-1)
        return self.fc(feature)

class OnlyText(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        TXT_DIM = 768
        IMG_DIM = 2048
        self.txt_model = tfm.AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model = t.nn.Sequential(
            *list(tv.models.resnet50(pretrained=True).children())[:-1])
        self.txt_layer = t.nn.TransformerEncoder(t.nn.TransformerEncoderLayer(
            TXT_DIM, 4, batch_first=True, norm_first=True), 6)
        self.fc = t.nn.Sequential(
            ResidualFC(TXT_DIM+IMG_DIM),
            ResidualFC(TXT_DIM+IMG_DIM),
            t.nn.Linear(TXT_DIM+IMG_DIM, 3),
            t.nn.Softmax(dim=-1),
        )

    def forward(self, img, txt):
        with t.no_grad():
            mask = ~txt['attention_mask'].to(t.bool)
            img = self.img_model(t.ones_like(img)).flatten(-3, -1)
            txt['input_ids'] *= 0
            txt = self.txt_model(**txt).last_hidden_state
        txt = self.txt_layer(txt, src_key_padding_mask=mask)
        txt = txt.mean(dim=-2)
        feature = t.concat([txt, img], dim=-1)
        return self.fc(feature)