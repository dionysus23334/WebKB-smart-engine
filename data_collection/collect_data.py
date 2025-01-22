import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import csv


def clean_content_thoroughly(text):
    """
    更严格地清理内容，移除所有换行符和分隔符。
    """
    if not isinstance(text, str):  # 确保输入是字符串
        return ""

    # 使用 BeautifulSoup 去除 HTML 标签
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()  # 提取纯文本

    # 替换 HTML 特殊字符
    text = re.sub(r"&nbsp;|&lt;|&gt;|&amp;|&quot;|&#39;", " ", text)
    # 替换换行符和制表符为空格
    text = re.sub(r"[\t\n\r]", " ", text)
    # 合并多余空格
    text = re.sub(r"\s+", " ", text)
    # 移除潜在的 CSV 分隔符（如逗号和双引号）
    text = re.sub(r"[\"',]", " ", text)  # 将逗号和引号替换为空格
    # 去除首尾多余空格
    return text.strip()


def load_and_clean_webkb_data(dataset_path, output_file):
    """
    Read webKB and save as CSV, whose structure is University -- class -- content -- link
    :param dataset_path: WebKB path
    :param output_file: output path
    """
    data = []  # 用于保存每条记录

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

    # 将数据保存到 CSV 文件
    df = pd.DataFrame(data)
    df.dropna()
    df.to_csv(output_file, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print(f"数据已保存到 {output_file}")


# 示例：WebKB 数据集路径和输出文件路径
dataset_path = "webkb"
output_file = "Collected_data.csv"

# 运行函数
load_and_clean_webkb_data(dataset_path, output_file)