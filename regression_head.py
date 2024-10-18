import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

model_name = "DeepPavlov/rubert-base-cased-sentence"

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_sentence_embedding(sentence: str) -> torch.Tensor:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding

class RegressionHead(nn.Module):

    def __init__(self):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(model.config.hidden_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(32, 16)
        self.dropout4 = nn.Dropout(0.3)
        
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        
        x = self.fc5(x)
        return x
