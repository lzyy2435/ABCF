import requests
from bs4 import BeautifulSoup
import re
import csv


def getDomain(url):
    domain = re.search(r"(https?://)?(www\.)?(.+?)/.*", url + '/').group(3)
    return domain


def getWhiteList(WhiteListFile):
    d = {}
    for _ in csv.reader(open(WhiteListFile)):
        temp = d
        for i in _[1].split('.')[::-1]:
            if i not in temp:
                temp[i] = {}
            temp = temp[i]
        temp['END'] = None
    return d


def isInWhiteList(url, whiteList):
    domain = getDomain(url).split('.')[::-1]
    temp = whiteList
    for i in domain:
        if i not in temp:
            return "END" in temp
        temp = temp[i]
    return "END" in temp


def getBlackList(BlackListFile, weight=1):
    l = []
    for _ in csv.reader(open(BlackListFile, encoding='utf-8')):
        l.append((_[0], int(_[1]) * weight))
    return l


def STRisInBlackLists(s, BlackListFiles):
    l = []
    # skey = dict(analyse.textrank(s, topK=50, withWeight=True))
    for ll in BlackListFiles:
        temp = 0
        for key, value in ll:
            temp += str(s).count(key) * value
        l.append(temp)
    return l


def URLisInBlackLists(domain, BlackListFiles):
    url = "http://" + domain
    try:
        response = requests.get(url, timeout=(3, 7))
        response.raise_for_status()
        response.encoding = response.apparent_encoding
    except:
        print(f"\033[0;31mNETWORK ERROR: {domain} 爬取失败\033[0m")
        return
    html = BeautifulSoup(response.text, 'lxml')
    return STRisInBlackLists(html, BlackListFiles)


def whiteThinking(url):
    whiteList = getWhiteList("domain/top-1m1.csv")
    if isInWhiteList(url, whiteList):
        print(f"白名单系统认为\033[0;34m{url}\033[0m为\033[0;32m正常网站\033[0m")
    else:
        print(f"白名单系统认为\033[0;34m{url}\033[0m可能为\033[0;31m非正常网站\033[0m")


def blackThinking(url):
    # 黑名单
    b2 = getBlackList("black/black2.txt", 3)
    b6 = getBlackList("black/black6.txt", 2)
    b10 = getBlackList("black/black10.txt", 2)
    b11 = getBlackList("black/black11.txt", 2)
    b = (b2, b6, b10, b11)
    out = (2, 6, 10, 11)

    prob = URLisInBlackLists(url, b)

    if prob is None:
        print(f"发生错误，黑名单系统无法判断")
    elif max(prob) <= 30:
        # print(f"黑名单系统认为\033[0;34m{url}\033[0m为\033[0;32m正常网站\033[0m")
        return 0
    else:
        # print(f"黑名单系统认为\033[0;34m{url}\033[0m为\033[0;31m{out[prob.index(max(prob))]}\033[0m")
        return out[prob.index(max(prob))]


if __name__ == '__main__':
    url = input("请输入域名:")
    whiteThinking(url)
    blackThinking(url)
