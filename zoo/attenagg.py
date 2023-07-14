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

class ProjDecoderLayer(t.nn.Module):
    def __init__(self, idim, mdim) -> None:
        super().__init__()
        self.projection = t.nn.Sequential(
            ResidualFC(idim+mdim),
            ResidualFC(idim+mdim),
            t.nn.Linear(idim+mdim, mdim),
        )
        self.txt_layer = t.nn.TransformerDecoderLayer(mdim, 4, batch_first=True, norm_first=True)

    def forward(self, feature, image, memory, attention_mask):
        projected = self.projection(t.concat([feature, image.unsqueeze(-2)], dim=-1))
        return self.txt_layer(projected, memory, memory_key_padding_mask=attention_mask) * 0.5 + feature * 0.5

class AttenAGG(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        TXT_DIM = 768
        IMG_DIM = 2048
        self.txt_model = tfm.AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model = t.nn.Sequential(
            *list(tv.models.resnet50(pretrained=True).children())[:-1])
        self.dec_layer = t.nn.ModuleList([ProjDecoderLayer(IMG_DIM, TXT_DIM) for _ in range(2)])
        self.enc_layer = t.nn.TransformerEncoder(t.nn.TransformerEncoderLayer(
            TXT_DIM, 4, batch_first=True, norm_first=True), 2)
        self.fc = t.nn.Sequential(
            t.nn.Linear(TXT_DIM, 3),
            t.nn.Softmax(dim=-1),
        )

    def forward(self, img, txt):
        with t.no_grad():
            mask = ~txt['attention_mask'].to(t.bool)
            img = self.img_model(img).flatten(-3, -1)
            txt = self.txt_model(**txt).last_hidden_state
        txt = self.enc_layer(txt, src_key_padding_mask=mask)
        feature = (txt * (~mask).unsqueeze(dim=-1)).sum(dim=-2) / ((~mask).sum(dim=-1).unsqueeze(-1) + 1e-10)
        if not self.training:
            feature = (feature + t.randn_like(feature) * config.RAND_EPS) / (1.0 + config.RAND_EPS)
        feature = feature.unsqueeze(-2)
        for layer in self.dec_layer:
            feature = layer(feature, img, txt, mask)
        return self.fc(feature.squeeze(-2))

class OnlyText(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        TXT_DIM = 768
        IMG_DIM = 2048
        self.txt_model = tfm.AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model = t.nn.Sequential(
            *list(tv.models.resnet50(pretrained=True).children())[:-1])
        self.dec_layer = t.nn.ModuleList([ProjDecoderLayer(IMG_DIM, TXT_DIM) for _ in range(2)])
        self.enc_layer = t.nn.TransformerEncoder(t.nn.TransformerEncoderLayer(
            TXT_DIM, 4, batch_first=True, norm_first=True), 2)
        self.fc = t.nn.Sequential(
            t.nn.Linear(TXT_DIM, 3),
            t.nn.Softmax(dim=-1),
        )

    def forward(self, img, txt):
        with t.no_grad():
            mask = ~txt['attention_mask'].to(t.bool)
            img = self.img_model(img).flatten(-3, -1) * 0.0
            txt = self.txt_model(**txt).last_hidden_state
        txt = self.enc_layer(txt, src_key_padding_mask=mask)
        feature = (txt * (~mask).unsqueeze(dim=-1)).sum(dim=-2) / ((~mask).sum(dim=-1).unsqueeze(-1) + 1e-10)
        if not self.training:
            feature = (feature + t.randn_like(feature) * config.RAND_EPS) / (1.0 + config.RAND_EPS)
        feature = feature.unsqueeze(-2)
        for layer in self.dec_layer:
            feature = layer(feature, img, txt, mask)
        return self.fc(feature.squeeze(-2))

class OnlyImage(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        TXT_DIM = 768
        IMG_DIM = 2048
        self.txt_model = tfm.AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model = t.nn.Sequential(
            *list(tv.models.resnet50(pretrained=True).children())[:-1])
        self.dec_layer = t.nn.ModuleList([ProjDecoderLayer(IMG_DIM, TXT_DIM) for _ in range(2)])
        self.enc_layer = t.nn.TransformerEncoder(t.nn.TransformerEncoderLayer(
            TXT_DIM, 4, batch_first=True, norm_first=True), 2)
        self.fc = t.nn.Sequential(
            t.nn.Linear(TXT_DIM, 3),
            t.nn.Softmax(dim=-1),
        )

    def forward(self, img, txt):
        with t.no_grad():
            mask = ~txt['attention_mask'].to(t.bool)
            img = self.img_model(img).flatten(-3, -1)
            txt = self.txt_model(**txt).last_hidden_state * 0.0
        txt = self.enc_layer(txt, src_key_padding_mask=mask)
        feature = (txt * (~mask).unsqueeze(dim=-1)).sum(dim=-2) / ((~mask).sum(dim=-1).unsqueeze(-1) + 1e-10)
        if not self.training:
            feature = (feature + t.randn_like(feature) * config.RAND_EPS) / (1.0 + config.RAND_EPS)
        feature = feature.unsqueeze(-2)
        for layer in self.dec_layer:
            feature = layer(feature, img, txt, mask)
        return self.fc(feature.squeeze(-2))
