import tqdm
from classifier import load_bert_dataset_model_optimizer
from sklearn.metrics import accuracy_score, classification_report
import torch
from tqdm import tqdm


def eval_bert():


    train_dataloader, test_dataloader, model, optimizer = load_bert_dataset_model_optimizer()


    # 将模型切换到评估模式
    model.eval()

    # 初始化存储预测结果的列表
    predictions = []
    true_labels = []

    # 使用测试集的 DataLoader
    with torch.no_grad():  # 禁用梯度计算
        for batch in tqdm(test_dataloader):  # 使用测试集的 DataLoader
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            

            # 获取预测类别
            preds = torch.argmax(logits, dim=-1)

            # 存储结果
            predictions.extend(preds.cpu().numpy())  # 将结果从 GPU 移到 CPU
            true_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # 计算分类报告（包括精确度、召回率、F1 得分）
    report = classification_report(true_labels, predictions, output_dict=True)
    print("Classification Report:\n", report)

    # 将准确率写入文件
    with open("model_accuracy.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(str(report))

