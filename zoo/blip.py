
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch as t

class BlipClf(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

    def forward(self, text, image):
        inputs = self.processor(image, 
            f"\nWhat is in the picture? ", return_tensors="pt").to('cuda')
        y = self.processor.decode(self.base.generate(**inputs)[0], skip_special_tokens=True).strip()
        if y in ["yes", "positive"]: return 0
        elif y in ["no", "negative"]: return 2
        else: return 1

if __name__ == '__main__':
    import pandas as pd
    model = BlipClf().to('cuda')
    train_set = pd.read_csv("meta/train.txt")
    sum = 0
    for i, row in train_set.iterrows():
        guid = row["guid"]
        tag = row["tag"]
        with open(f"data/{guid}.txt", encoding="gbk", errors="surrogateescape") as f:
            text = f.read()
        with open(f"data/{guid}.jpg", "rb") as f:
            image = Image.open(f.raw).convert('RGB')
        sum += (["positive", "neutral", "negative"][model(text, image)] == tag)
        print(sum / (i + 1))