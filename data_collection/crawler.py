
from spiders.simple_spider import SimpleSpider
from spiders.agent_spider import AgentSpider

def get_spider_config(spider):
    spider_dict = {'SimpleSpider':SimpleSpider,'AgentSpider':AgentSpider}
    return spider_dict[spider]


class Crawler:

    def __init__(self, name):
        
        self.spider = get_spider_config(name)()

    def crawl(self,is_local,source):

        self.spider.set_is_local(is_local)
        self.spider.set_source(source)

        return self.spider.run()

if __name__ == "__main__":

    crawler = Crawler(name='SimpleSpider')

    parsed_data = crawler.crawl(is_local=True, source='D:/research/research5/data/WebKB/webkb-data.gtar/webkb/course/cornell/http_^^cs.cornell.edu^Info^Courses^Current^CS415^CS414.html')
 
    print(parsed_data)

