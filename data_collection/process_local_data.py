import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch,padding=True, truncation=True)

def _get_localdata():
    csv_file = "./local_data/collected_content.csv"
    df = pd.read_csv(csv_file)
    return df

# Define dataset
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

if __name__ == "__main__":

    csv_file = "./local_data/collected_content.csv"
    df = pd.read_csv(csv_file)
    
    schools = df.iloc[:,0].tolist()
    y = df.iloc[:,1].tolist()
    X1,X2 = df.iloc[:,2].tolist(),df.iloc[:,-1].tolist()

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    batch_size = 32
    dataset = TextDataset(X1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # fetch [CLS] embeddings
    all_cls_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
    
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
    
    
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # 提取 [CLS] 嵌入
    
            all_cls_embeddings.append(cls_embeddings.cpu())
    
    # gather all [CLS] embeddings
    all_cls_embeddings = torch.cat(all_cls_embeddings, dim=0)
    print("All CLS Embeddings shape:", all_cls_embeddings.shape)
    
    # store embeddings
    hidden_cache_path = "optimized_class_hidden_cache.pt"
    torch.save(all_cls_embeddings, hidden_cache_path)
    print(f"CLS embeddings saved to {hidden_cache_path}")
  
