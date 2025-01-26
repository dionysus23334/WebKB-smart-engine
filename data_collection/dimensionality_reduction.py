import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


csv_file = "./local_data/collected_content.csv"
df = pd.read_csv(csv_file)

df.iloc[:,1].value_counts(ascending=True).plot(kind='barh',color='#3498db')
plt.title('Frequency of Classes')
plt.savefig("counts.png", dpi=300, format="png", bbox_inches="tight")

y = df.iloc[:,1]
cls_embeddings = torch.load("optimized_class_hidden_cache.pt",weights_only=True)

X = cls_embeddings.numpy()
X_scaled = MinMaxScaler().fit_transform(X)

mapper = UMAP(n_components=2, metric='cosine')
embedding = mapper.fit_transform(X_scaled)  # 确保生成 embedding

# 创建 DataFrame 并添加降维结果和标签
df_cls = pd.DataFrame(embedding, columns=['X', 'Y'])
df_cls['Label'] = y

print(df_cls.head())


fig = plt.figure(figsize=(9, 6))

axes = []
for i in range(4):
    ax = fig.add_subplot(2, 4, i + 1)  # 第 1 行的子图
    axes.append(ax)

for i in range(3):
    ax = fig.add_subplot(2, 4, 5 + i)  # 第 2 行的子图
    axes.append(ax)

labels = np.unique(y.tolist())

for i, label in enumerate(labels):
    df_cls_sub = df_cls.query(f"Label == '{label}'")
    axes[i].hexbin(df_cls_sub['X'],
                   df_cls_sub['Y'],
                   cmap = 'Blues',
                   gridsize = 20)
    axes[i].set_title(label)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.savefig("hexbin_plot.png", dpi=300, format="png", bbox_inches="tight")
plt.show()
