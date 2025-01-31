import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re
from collections import defaultdict, Counter

# Read and store graph structure
csv_file = "collected_content.csv"
df = pd.read_csv(csv_file)

def extract_domain(url):
    url_clean = url.replace('^', '/')
    match = re.findall(r"http[s]?:\/\/([^\/]+)", url_clean)
    return match[0] if match else None

def extract_path(url):
    url_clean = url.replace('^', '/')
    parts = url_clean.split('/')
    return parts[2:] if len(parts) > 2 else []

domain_map = defaultdict(list)
path_map = defaultdict(list)

for i, url in enumerate(df.iloc[:, -1]):
    domain = extract_domain(url)
    path = extract_path(url)
    if domain:
        domain_map[domain].append(i)
    if path:
        path_map['/'.join(path)].append(i)

edges = [(src, dst) for nodes in domain_map.values() for src in nodes for dst in nodes if src != dst]
edges += [(src, dst) for nodes in path_map.values() for src in nodes for dst in nodes if src != dst]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# （TF-IDF）
vectorizer = TfidfVectorizer(max_features=500)
text_features = vectorizer.fit_transform(df.iloc[:, -2]).toarray()

cls_embeddings = torch.load("optimized_class_hidden_cache.pt", weights_only=True)
gnn_features = MinMaxScaler().fit_transform(cls_embeddings.numpy())

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df.iloc[:, 1])

# Oversampling SMOTE
class_counts = Counter(y_encoded)
max_class_samples = max(class_counts.values())

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_combined = np.hstack((text_features, gnn_features))  # 合并不同特征
X_resampled, y_resampled = smote.fit_resample(X_combined, y_encoded)


X_text_resampled = X_resampled[:, :text_features.shape[1]]
X_gnn_resampled = X_resampled[:, text_features.shape[1]:]

X_text_train, X_text_test, X_gnn_train, X_gnn_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X_text_resampled, X_gnn_resampled, y_resampled, np.arange(len(y_resampled)),
    test_size=0.2, stratify=y_resampled, random_state=42
)

train_mask = torch.zeros(len(y_resampled), dtype=torch.bool)
test_mask = torch.zeros(len(y_resampled), dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

graph_data = Data(
    x=torch.tensor(X_gnn_resampled, dtype=torch.float32),
    edge_index=edge_index,
    y=torch.tensor(y_resampled, dtype=torch.long),
    train_mask=train_mask,
    test_mask=test_mask
)

# Define MLP & GNN
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, output_size)

        self.shortcut = nn.Linear(input_size, output_size)


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out) + self.shortcut(x)
        return out

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2, concat=True)
        self.conv2 = GATConv(hidden_dim * 2, output_dim, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class HybridModel(nn.Module):
    def __init__(self, text_dim, gnn_dim, hidden_dim, output_dim):
        super(HybridModel, self).__init__()
        self.text_mlp = MLP(text_dim, output_dim)
        self.gnn = GNN(gnn_dim, hidden_dim, output_dim)
        self.fc_final = nn.Linear(output_dim * 2, output_dim)

    def forward(self, text_features, graph_data, mask):
        text_out = self.text_mlp(text_features)  # (batch_size, num_classes)
        gnn_out = self.gnn(graph_data)  # (num_nodes, num_classes)

        gnn_out = gnn_out[mask]
        assert text_out.shape == gnn_out.shape, f"Shape mismatch: text_out={text_out.shape}, gnn_out={gnn_out.shape}"

        combined = torch.cat([text_out, gnn_out], dim=1)
        return F.log_softmax(self.fc_final(combined), dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridModel(500, X_gnn_train.shape[1], 128, len(np.unique(y_resampled))).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    output = model(torch.tensor(X_text_train, dtype=torch.float32).to(device), graph_data, graph_data.train_mask)

    loss = criterion(output, torch.tensor(y_train, dtype=torch.long).to(device))
    loss.backward()
    optimizer.step()

    pred = output.argmax(dim=1)
    acc = (pred == torch.tensor(y_train, dtype=torch.long).to(device)).float().mean().item()

    train_losses.append(loss.item())
    train_accuracies.append(acc)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")

torch.save(model, "hybrid_model.pth")


plt.figure(figsize=(12, 5))

# Loss curve
fig, axes = plt.subplots(1, 2)
axes[0].plot(range(1, num_epochs + 1), train_losses, label="Loss", color='red')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss Curve")


axes[1].plot(range(1, num_epochs + 1), train_accuracies, label="Accuracy", color='blue')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Training Accuracy Curve")

plt.legend()
plt.savefig(f'Hybrid_1,acc={train_accuracies[-1]:.4f}.png')


model.eval()
with torch.no_grad():
    output = model(torch.tensor(X_text_test, dtype=torch.float32).to(device), graph_data, graph_data.test_mask)
    pred = output.argmax(dim=1)

test_acc = (pred == torch.tensor(y_test, dtype=torch.long).to(device)).float().mean().item()
print(f"Test Accuracy: {test_acc:.4f}")

#F1-score
f1 = f1_score(y_test, pred.cpu().numpy(), average="macro")
print(f"F1-score: {f1:.4f}")

#Confusion Matrix
cm = confusion_matrix(y_test, pred.cpu().numpy(),normalize='true')
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues",
          values_format=".2f",
          ax = ax
          )
plt.text(0.1, -0.2,
         f"Test Accuracy: {test_acc:.4f}\nF1-score: {f1:.4f}",
         fontsize=12, ha="left", va="top", transform=ax.transAxes,
         bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))
plt.title("Confusion Matrix")
plt.savefig('Hybrid_cm.png', dpi=300, format="png", bbox_inches="tight")
