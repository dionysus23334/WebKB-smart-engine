# 作者 Author：Yang Fenglin
import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import csv


def clean_content_thoroughly(text):
    """
    clean original text
    """
    if not isinstance(text, str): 
        return ""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text() 

    text = re.sub(r"&nbsp;|&lt;|&gt;|&amp;|&quot;|&#39;", " ", text)

    text = re.sub(r"[\t\n\r]", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[\"',]", " ", text)  

    return text.strip()


def load_clean_filter_webkb_data(dataset_path, output_file):
    """
    Read and drop_Nan webKB, write into CSV, with the structure of: University, Class, Content, Link
    :param dataset_path: path to Web->KB
    :param output_file: output path
    """
    data = []  

    # 遍历数据集的目录结构
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)

        if os.path.isdir(class_path):  # 确保是文件夹
            for university in os.listdir(class_path):
                university_path = os.path.join(class_path, university)

                if os.path.isdir(university_path):  # 确保是文件夹
                    for file_name in os.listdir(university_path):
                        file_path = os.path.join(university_path, file_name)

                        # 读取并清理文件内容
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                            raw_content = file.read()
                            clean_text = clean_content_thoroughly(raw_content)

                        # 添加到数据列表
                        data.append({
                            "University": university,
                            "Class": class_name,
                            "Content": clean_text,
                            "Filename": file_name
                        })

    # 将数据保存到原始 CSV 文件
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print('Done')

if __name__ == "__main__":

    dataset_path = "webkb"
    output_file = "collected_content.csv"
    allowed_universities = ['cornell','misc','texas','washington','wisconsin']

    # 运行函数
    load_clean_filter_webkb_data(dataset_path, output_file)
