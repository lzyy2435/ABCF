from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
import csv
import re
import jieba.analyse as analyse


def log(func):
    def wrapper(*args, **kw):
        r = func(*args, **kw)
        print(",".join(r))
        return r

    return wrapper


# @log
def getMsg(driver, url):
    if not re.match(r"https?://", url):
        url = "http://" + url
    try:
        driver.get(url)
        msg = driver.page_source
    except TimeoutException:
        print("加载页面太慢，停止加载，继续下一步操作")
        driver.execute_script("window.stop()")
        msg = driver.page_source
    except WebDriverException:
        print("访问网页失败")
        msg = "访问网页失败"
    msg = re.findall(r'[\u4e00-\u9fa5]+', msg)
    return msg


def simpleWash(str_list):
    d = {"去重后句子集": []}
    dd = {}
    for s in str_list:
        if s in dd:
            dd[s] += 1
        else:
            dd[s] = 1
            d["去重后句子集"].append(s)
    nn = max(len(dd), 1)
    n = sum(dd.values()) / nn
    d["高频句子"] = list(filter(lambda x: dd[x] > n, dd.keys()))
    d["高频词语"] = list(analyse.textrank("，".join(str_list), topK=30))
    return d


def readTrain1(driver, n):
    i = 0
    label = []
    data = []
    for _ in csv.reader(open("data/train1.csv")):
        if i > n: break
        data.append(simpleWash(getMsg(driver, _[0])))
        label.append(_[1])
        i += 1

    return data, label


# 不加载图片
chrome_options = webdriver.ChromeOptions()
prefs = {'profile.managed_default_content_settings.images': 2}
chrome_options.add_experimental_option('prefs', prefs)

# executable_path是driver的地址
# driver下载地址：http://chromedriver.storage.googleapis.com/index.html
# 选择合适版本
# 如果之前配置好了就直接: driver = webdriver.Chrome()
driver = webdriver.Chrome(executable_path="chromedriver.exe", options=chrome_options)

# 渲染时间
t = 20
driver.set_page_load_timeout(t)
driver.set_script_timeout(t)

print(readTrain1(driver, 10))
