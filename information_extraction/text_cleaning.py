import re
import html
import torch
from bs4 import BeautifulSoup

def clean_text(text):
    """
    清洗文本内容，移除无关字符和格式化文本
    :param text: 原始文本内容
    :return: 清洗后的文本
    """
    # 移除 Unicode 编码字符（例如 \u200e）
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    
    # 解码 HTML 实体字符
    text = html.unescape(text)
    
    # 去掉英文字母以外的字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 去除多余的空格和换行
    text = re.sub(r'\s+', ' ', text).strip()
    
    # # 可以根据需要移除特定的无关文本
    # unwanted_text = ["Skip to main content", "Powered by Moodle", "Data retention summary"]
    # for phrase in unwanted_text:
    #     text = text.replace(phrase, "")
    
    return text

def clean_links(links):
    """
    清洗链接，去除重复链接和不相关的链接
    :param links: 原始链接列表
    :return: 清洗后的链接列表
    """
    # 去除重复的链接
    links = list(set(links))
    
    # 可以根据需要去除一些特定的链接
    unwanted_links = [
        "https://wolfware.ncsu.edu",  # 示例：移除特定的无关链接
        "https://moodle.com"           # 例如，移除 Moodle 官方链接
    ]
    
    links = [link for link in links if link not in unwanted_links]
    
    return links

# 清洗示例数据

def get_cleaned_data(data):
    # 执行清洗
    cleaned_text = clean_text(data['text_content'])
    cleaned_links = clean_links(data['links'])

    # 输出清洗后的数据
    cleaned_data = {
        'title': data['title'],
        'text_content': cleaned_text,
        'links': cleaned_links
    }

    # 打印结果
    return cleaned_data


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # 移除多余空格
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5 ]', '', text)  # 保留中英文和数字
    return text


def clean_html(content):
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text()

# 分词和编码文本
def tokenize_data(texts, labels, tokenizer, max_length=512):
    inputs = tokenizer(texts.tolist(), max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    return inputs, torch.tensor(labels, dtype=torch.long)


if __name__ == "__main__":

    data = {
    'title': 'Log in to the site | Moodle Outreach',
    'text_content': 'Log in to the site | Moodle Outreach Skip to main content Side panel WolfWare Home More English (United States) \u200e(en_us)\u200e Deutsch \u200e(de_old)\u200e English \u200e(en)\u200e English (United States) \u200e(en_us)\u200e Español - Internacional \u200e(es_old)\u200e Français \u200e(fr_old)\u200e Português - Brasil \u200e(pt_br_old)\u200e اردو \u200e(ur_old)\u200e हिंदी \u200e(hi_old)\u200e 简体中文 \u200e(zh_cn_old)\u200e Log in Home Log in to the site Moodle Outreach NC State Unity Login If you have a Unity ID Login or an @ncsu.edu email address, use the red Unity ID login button below. Unity ID Login NC State Students/Faculty/Staff Forgot your Unity ID or Password? Brickyard Login If you have a Brickyard Login and do not have an @ncsu.edu email address, use the Brickyard Login button below. This includes many users registered through REPORTER. Brickyard Login NC State Guests/Affiliates/Parents Forgot your Brickyard Account ID or Password? Outreach User Login If the other options do not apply to you, use the Outreach User Login button below. Outreach User Login Moodle Outreach Users Username Password Log in Forgot your Moodle Outreach username or password? Cookies must be enabled New Outreach Users If you are not affiliated with NC State or do not have a Unity ID please create a new Brickyard account by clicking the "Create New Account" button below. Create New Account NOTE: If you are an NC State instructor, staff member, or student, do not create a new account. Instead, you should log in with your Unity ID and password issued by the University. If you need help with your Unity ID, please contact help@ncsu.edu (919-515-HELP). Contact site support You are not logged in. Data retention summary Get the mobile app Powered by Moodle',
    'links': ['https://wolfware.ncsu.edu', 'https://moodle-outreach.wolfware.ncsu.edu/', 'https://moodle-outreach.wolfware.ncsu.edu/login/index.php?lang=de_old', 'https://moodle-outreach.wolfware.ncsu.edu/login/index.php?lang=en', 'https://moodle-outreach.wolfware.ncsu.edu/login/index.php?lang=es_old', 'https://moodle-outreach.wolfware.ncsu.edu/login/index.php?lang=fr_old', 'https://moodle-outreach.wolfware.ncsu.edu/login/index.php?lang=pt_br_old', 'https://moodle-outreach.wolfware.ncsu.edu/login/index.php?lang=ur_old', 'https://moodle-outreach.wolfware.ncsu.edu/login/index.php?lang=hi_old', 'https://moodle-outreach.wolfware.ncsu.edu/login/index.php?lang=zh_cn_old', 'https://moodle-outreach.wolfware.ncsu.edu/login/index.php', 'https://moodle-outreach.wolfware.ncsu.edu/', 'https://moodle-outreach.wolfware.ncsu.edu/auth/shibboleth/index.php', 'https://oit.ncsu.edu/my-it/unity-credentials/', 'https://moodle-outreach.wolfware.ncsu.edu/auth/shibbolethea/index.php', 'https://go.ncsu.edu/parent_password', 'https://moodle-outreach.wolfware.ncsu.edu/login/forgot_password.php', 'https://passport.oit.ncsu.edu/link/signup', 'https://delta.ncsu.edu/get-help/', 'https://moodle-outreach.wolfware.ncsu.edu/admin/tool/dataprivacy/summary.php', 'https://go.ncsu.edu/moodle-mobile:moodle-outreach', 'https://moodle.com']
    }

    cleaned_data = get_cleaned_data(data)
    
    print(cleaned_data['text_content'])


