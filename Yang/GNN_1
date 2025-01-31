import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import re
from collections import defaultdict
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

csv_file = "collected_content.csv"
df = pd.read_csv(csv_file)

link = df.iloc[:,-1].tolist()

def extract_domain(url):
    url_clean = url.replace('^','/')
    match = re.findall(r"http[s]?:\/\/([^\/]+)", url_clean)
    return match[0] if match else None

def extract_path(url):
    url_clean = url.replace('^','/')
    parts = url_clean.split('/')
    return parts[2:] if len(parts) > 2 else []

domain_map = defaultdict(list)
path_map = defaultdict(list)

for i, url in enumerate(link):
    domain = extract_domain(url)
    path = extract_path(url)
    if domain:
        domain_map[domain].append(i)

    if path:
        path_map['/'.join(path)].append(i)

edges = []

for nodes in domain_map.values():
    for src in nodes:
        for dst in nodes:
            if src != dst:
                edges.append((src,dst))

for nodes in path_map.values():
    for src in nodes:
        for dst in nodes:
            if src!= dst:
                edges.append((src, dst))


edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)


cls_embeddings = torch.load("optimized_class_hidden_cache.pt", weights_only=True)
X = cls_embeddings.numpy()
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df.iloc[:,1])

smote = SMOTE(sampling_strategy='auto', random_state=42)  # auto: 让所有类别的样本数匹配多数类

# 进行过采样
X_balanced, y_balanced = smote.fit_resample(X_scaled, y_encoded)

X_train, X_test, y_train,y_test, train_idx, test_idx = train_test_split(X_balanced,
                                                   y_balanced,
                                                   np.arange(len(y_balanced)),
                                                   stratify= y_balanced,
                                                   random_state=0,
                                                   train_size= 0.7)

X_tensor = torch.tensor(X_balanced, dtype=torch.float32)
y_tensor = torch.tensor(y_balanced, dtype=torch.long)

train_mask = torch.zeros(len(y_balanced), dtype=torch.bool)
test_mask = torch.zeros(len(y_balanced), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[train_idx] = True

graph_data = Data(x = X_tensor,
                  edge_index=edge_index,
                  y=y_tensor,
                  train_mask=train_mask,
                  test_mask=test_mask)

class ResidualGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidualGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.conv3 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False)
        self.shortcut = nn.Linear(input_dim, hidden_dim)  # **只保留一个残差连接**
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index).relu()
        x1 = self.dropout(x1)
        x2 = self.conv2(x1, edge_index).relu()
        x2 = self.dropout(x2)
        x_out = self.conv3(x2, edge_index) + self.shortcut(x)  # **仅此处残差连接**

        return F.log_softmax(x_out, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualGNN(input_dim=X.shape[1], hidden_dim=64, output_dim=len(np.unique(y_encoded))).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
loss_epoch, accuracy_epoch = [], []

graph_data = graph_data.to(device)
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(graph_data)
    loss = criterion(output[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()

    pred = output.argmax(dim=1)
    acc = (pred[graph_data.train_mask] == graph_data.y[graph_data.train_mask]).sum().item() / graph_data.train_mask.sum().item()

    loss_epoch.append(loss.item())
    accuracy_epoch.append(acc)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(loss_epoch, label="Loss")
axes[0].set_title("Training Loss")
axes[1].plot(accuracy_epoch, label="Accuracy", color="orange")
axes[1].set_title("Training Accuracy")
plt.tight_layout()
plt.show()

model.eval()
with torch.no_grad():
    output = model(graph_data)
    pred = output.argmax(dim=1)

test_acc = (pred[graph_data.test_mask] == graph_data.y[graph_data.test_mask]).sum().item() / graph_data.test_mask.sum().item()
print(f"Test Accuracy: {test_acc:.4f}")

y_true = label_encoder.inverse_transform(graph_data.y[graph_data.test_mask].cpu().numpy())
y_pred = label_encoder.inverse_transform(pred[graph_data.test_mask].cpu().numpy())
f1 = f1_score(y_true, y_pred, average="macro")
print(f"F1-score: {f1:.4f}")

labels = np.unique(y_true)
cm = confusion_matrix(y_true, y_pred, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Normalized Confusion Matrix for GNN")
plt.show()
