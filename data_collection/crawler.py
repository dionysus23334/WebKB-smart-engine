
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
