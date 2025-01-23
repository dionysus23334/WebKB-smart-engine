from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


csv_file = "collected_content.csv"
df = pd.read_csv(csv_file)

y = df.iloc[:,1]
cls_embeddings = torch.load("optimized_class_hidden_cache.pt",weights_only=True)

X = cls_embeddings.numpy()

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,random_state=0)
Scaler = MinMaxScaler().fit(X_train)
X_train_scaled = Scaler.transform(X_train)
X_test_scaled= Scaler.transform(X_test)


svm = SVC(gamma=0.1, C=10).fit(X_train_scaled,y_train)

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

y_preds = svm.predict(X_test_scaled)
labels = np.unique(y_test)

plot_confusion_matrix(y_preds,y_test, labels, svm)
print(f'scores for SVM:{svm.score(X_test_scaled,y_test)}')

NB = GaussianNB().fit(X_train_scaled,y_train)
y_pred_NB = NB.predict(X_test_scaled)

plot_confusion_matrix(y_pred_NB,y_test,labels, NB)
print(f'accuracy for naive bayes:{NB.score(X_test_scaled,y_test)}')