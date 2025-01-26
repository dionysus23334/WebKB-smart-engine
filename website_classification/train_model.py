import tqdm
from classifier import load_bert_dataset_model_optimizer, load_svm_dataset_model
from sklearn.metrics import accuracy_score, classification_report
import torch
from tqdm import tqdm


# 作者 Author：Guo Yuxi
def train_bert(epoch):

    train_dataloader, test_dataloader, model, optimizer = load_bert_dataset_model_optimizer()

    for epoch in range(epoch):  # cpu训练1个epoch大约4个小时
        for batch in tqdm(train_dataloader):
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed. Loss: {loss.item()}")

    # 保存模型的 state_dict
    torch.save(model.state_dict(), "model_weights.pth")
    # 保存优化器的状态字典
    torch.save(optimizer.state_dict(), "optimizer_state.pth")
    # 保存整个模型对象（不推荐，但可以使用）
    torch.save(model, "complete_model.pth")

# 作者 Author: Yang Fenglin
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

# 作者 Author: Yang Fenglin
def train_svm():
    X_train_scaled, X_test_scaled, y_train, y_test, svm = load_svm_dataset_model()
    svm = SVC(gamma=0.1, C=10).fit(X_train_scaled,y_train)
    y_preds = svm.predict(X_test_scaled)
    labels = np.unique(y_test)
    print(f'scores for SVM:{svm.score(X_test_scaled,y_test)}')
    plot_confusion_matrix(y_preds,y_test, labels, svm)
def train_NB():
    X_train_scaled, X_test_scaled, y_train, y_test, svm = load_svm_dataset_model()
    NB = GaussianNB().fit(X_train_scaled,y_train)
    y_pred_NB = NB.predict(X_test_scaled)
    labels = np.unique(y_test)
    plot_confusion_matrix(y_pred_NB,y_test,labels, NB)
    print(f'accuracy for naive bayes:{NB.score(X_test_scaled,y_test)}')
