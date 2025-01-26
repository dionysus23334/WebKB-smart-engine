import re
import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader


# 导入自定义包
from information_extraction.text_cleaning import clean_text, clean_html, tokenize_data

# 作者：Guo Yuxi
def load_bert_dataset_model_optimizer():

    csv_file = "./data/collected_content.csv"
    df = pd.read_csv(csv_file,encoding='gbk')
    y = df.iloc[:,1]
    df = df.dropna(subset=['Content', 'Class'])

    # 检查模型文件是否存在
    model_path = "model_weights.pth"
    optimizer_path = "optimizer_state.pth"

    df['Content'] = df['Content'].apply(clean_text)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    print(f"Using device: {device}")

    df['Content'] = df['Content'].apply(clean_html)
    X = df['Content']
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 创建标签编码器
    label_encoder = LabelEncoder()

    # 将分类标签转换为数值
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(label_encoder.classes_)  # 查看标签对应的数值编码
    # 定义分类类别数量
    num_classes = len(label_encoder.classes_)

    # 分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_inputs, train_labels = tokenize_data(X_train, y_train_encoded, tokenizer)
    test_inputs, test_labels = tokenize_data(X_test, y_test_encoded, tokenizer)
    train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


    # 判断是否已有模型文件
    if os.path.exists(model_path):
        # 如果模型文件存在，加载模型和优化器
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        print("Model loaded successfully!")

        # 加载优化器（如果存在优化器文件）
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
            print("Optimizer loaded successfully!")
    else:
        # 如果模型文件不存在，创建新模型并开始训练
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
        model.to(device)
        print("New model created for training.")

        # 定义优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    return train_dataloader, test_dataloader, model, optimizer

# 作者：Yang Fenglin
# Guo Yuxi进行整理
def load_svm_dataset_model():

    csv_file = "collected_content.csv"
    df = pd.read_csv(csv_file)
    
    schools = df.iloc[:,0].tolist()
    y = df.iloc[:,1].tolist()
    X1,X2 = df.iloc[:,2].tolist(),df.iloc[:,-1].tolist()
    
    model_ckpt = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    def tokenize(batch):
        return tokenizer(batch,padding=True, truncation=True)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AutoModel.from_pretrained(model_ckpt).to(device)
    
    
    # Define dataset
    class TextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts
    
        def __len__(self):
            return len(self.texts)
    
        def __getitem__(self, idx):
            return self.texts[idx]
    
    
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


    csv_file = "collected_content.csv"
    df = pd.read_csv(csv_file)
    
    y = df.iloc[:,1]
    cls_embeddings = torch.load("optimized_class_hidden_cache.pt",weights_only=True)
    
    X = cls_embeddings.numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=0)
    Scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = Scaler.transform(X_train)
    X_test_scaled = Scaler.transform(X_test)

    svm = SVC(gamma=0.1, C=10).fit(X_train_scaled,y_train)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, svm



