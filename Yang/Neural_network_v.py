import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

csv_file = "collected_content.csv"
df = pd.read_csv(csv_file)

y = df.iloc[:,1]
cls_embeddings = torch.load("optimized_class_hidden_cache.pt",weights_only=True)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X = cls_embeddings.numpy()

X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,stratify=y,random_state=0, train_size=0.8)
Scaler = MinMaxScaler().fit(X_train)
X_train_scaled = Scaler.transform(X_train)
X_test_scaled= Scaler.transform(X_test)

X_tensor= torch.tensor(X_train_scaled)
y_tensor = torch.tensor(y_train)

dataset = TensorDataset(X_tensor,y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def plot_confusion_matrix(y_pred, y_true, labels, model_name):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap = 'Blues',
              values_format='.2f',
              ax = ax,
              colorbar = False)
    plt.title("Normalized Confusion Matrix")
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=300, format="png", bbox_inches="tight")

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout()

    def forward(self,x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = X_train_scaled.shape[1]
num_classes = len(np.unique(y))
model = Model(input_size, num_classes)
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
model.to(device)

loss_epoch = []
accuracy_epoch = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加损失
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    # 计算每个 epoch 的损失和准确率
    _ = loss_epoch.append(epoch_loss / len(dataloader))  # 平均损失
    _ = accuracy_epoch.append(100 * correct / total)  # 准确率（百分比）

    print(f"Epoch[{epoch + 1}/{num_epochs}], Loss: {loss_epoch[-1]:.3f}, Accuracy: {accuracy_epoch[-1]:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].plot(loss_epoch, label="Loss")
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(accuracy_epoch, label="Accuracy", color="orange")
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()

# 测试模型
model.eval()
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred = torch.max(outputs, 1)

# 还原预测的字符串标签
y_test = label_encoder.inverse_transform(y_test)
y_pred = label_encoder.inverse_transform(y_pred.cpu().numpy())
print("Predicted String Labels:", y_pred)

labels = np.unique(y_test)
plot_confusion_matrix(y_pred,y_test,labels,'NN')
