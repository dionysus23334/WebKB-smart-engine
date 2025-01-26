# 作者 Author：Guo Yuxi
import spacy

def entity_recognition(text):
    """
    使用 spaCy 进行实体识别，从文本中提取实体（如人名、组织名、地点等）。
    :param text: 输入的文本
    :return: 包含实体及其类别的字典列表
    """

    # 加载预训练的英语模型
    nlp = spacy.load("en_core_web_sm")


    # 使用spaCy处理文本
    doc = nlp(text)
    
    # 提取所有实体
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    
    return entities


if __name__ == "__main__":

    # 示例文本
    text = '''Moodle Outreach is a service provided by NC State University. 
          Dr. John Doe is a professor at NC State and will attend the event in New York.'''

    # 调用实体识别函数
    entities = entity_recognition(text)

    # 打印识别出的实体
    for entity in entities:
        print(f"Entity: {entity['text']}, Label: {entity['label']}, Start: {entity['start']}, End: {entity['end']}")



