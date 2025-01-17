import os
import requests
from bs4 import BeautifulSoup

class SimpleSpider:
    def __init__(self, is_local=False, source=None):
        """
        初始化爬虫
        :param is_local: 如果为True, 表示爬取本地文件
        :param source: 本地文件路径或网站URL
        """
        self.is_local = is_local
        self.source = source

    def set_is_local(self, is_local):
        self.is_local=is_local

    def set_source(self, source):
        self.source=source

    def fetch_data(self):
        """
        根据输入的来源抓取数据
        :return: 页面内容（HTML）
        """
        if self.is_local:
            # 如果是本地文件
            if not os.path.exists(self.source):
                raise FileNotFoundError(f"文件 {self.source} 未找到")
            with open(self.source, 'r', encoding='utf-8') as file:
                html_content = file.read()
        else:
            # 如果是网站链接
            try:
                response = requests.get(self.source)
                response.raise_for_status()  # 如果响应状态码不是200，会抛出异常
                html_content = response.text
            except requests.exceptions.RequestException as e:
                print(f"请求失败: {e}")
                html_content = None

        return html_content

    def parse_html(self, html_content):
        """
        解析 HTML 页面并提取信息
        :param html_content: HTML 页面内容
        :return: 解析后的数据
        """
        if html_content is None:
            print("无法解析空内容")
            return None

        # 使用 BeautifulSoup 解析 HTML 内容
        soup = BeautifulSoup(html_content, 'html.parser')

        # 例如：提取页面标题
        title = soup.title.string if soup.title else "无标题"
        # 你可以根据需要提取更多的内容，如文本、链接、图片等

        # 提取所有文本内容
        text_content = soup.get_text(separator=' ', strip=True)

        # 提取所有跳转链接
        links = []
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            if link.startswith('http'):  # 仅抓取完整的URL
                links.append(link)
        
        return {"title": title, "text_content": text_content, "links": links}

    def run(self):
        """
        执行爬虫任务
        :return: 解析后的数据
        """
        print(f"正在爬取: {'本地文件' if self.is_local else '网站'} - {self.source}")

        # 获取页面内容
        html_content = self.fetch_data()

        if html_content:
            # 解析内容
            parsed_data = self.parse_html(html_content)
            return parsed_data
        else:
            print("没有获取到有效的内容")
            return None

# 使用示例
if __name__ == '__main__':
    # 从本地 HTML 文件抓取
    local_spider = SimpleSpider(is_local=True, source='D:/research/research5/data/WebKB/webkb-data.gtar/webkb/course/cornell/http_^^cs.cornell.edu^Info^Courses^Current^CS415^CS414.html')
    local_data = local_spider.run()
    print(local_data)

    # 从网站抓取
    website_spider = SimpleSpider(is_local=False, source='https://moodle-outreach.wolfware.ncsu.edu/course/view.php?id=1675https://moodle-outreach.wolfware.ncsu.edu/course/view.php?id=1675')
    website_data = website_spider.run()
    print(website_data)

