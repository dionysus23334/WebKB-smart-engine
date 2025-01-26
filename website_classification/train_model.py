import tqdm
from classifier import load_bert_dataset_model_optimizer
from sklearn.metrics import accuracy_score, classification_report
import torch
from tqdm import tqdm


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
    
