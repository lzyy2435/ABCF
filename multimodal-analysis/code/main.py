import logging
from src.homepage2vec.model import WebsiteClassifier, Webpage
import csv
from Stage2_XGBoost import stage2_xgb
from Stage1 import stage1
from 黑白名单 import blackThinking
import operator

"""
demo
"""

classes = {0: '正常', 1: '购物消费', 2: '婚恋交友', 3: '假冒身份', 4: '钓鱼网站', 5: '冒充公检法', 6: '平台诈骗',
           7: '招聘兼职',
           8: '杀猪盘', 9: '博彩赌博', 10: '信贷理财', 11: '刷单诈骗', 12: '中奖诈骗'}
logging.getLogger().setLevel(logging.DEBUG)

# 这是一个演示脚本
model = WebsiteClassifier(use_my_model=True)

sus_pool = list(csv.reader(open('./suspect_pool_new.txt', encoding='UTF-8')))
paths = [path.strip('\n') for path in list(open('./path.txt', encoding='UTF-8'))]
urls_idx = [int(path.split('.')[0].split('-')[-1]) for path in paths]
urls = [sus_pool[idx] for idx in urls_idx]


htmls = []
for p in paths:
    with open(p, encoding='utf-8') as f:
        html = f.read()
    htmls.append(html)

print(paths)
print(urls_idx)
print(urls)
print(len(htmls))
all_ans = ''
for i in range(len(htmls)):
    website = Webpage(urls[i][0])
    website.html = htmls[i]
    website.is_valid = True
    scores, embeddings = model.predict(website)

    print(urls[i][1])
    ans_str = str(i) + ' ' + urls[i][0] + ' '
    score_out = []
    for j in list(scores.values()):
        score_out.append(str(int((100*j + 1)**0.5 * 10)))
    ans_str += ' '.join(score_out)
    ans_str += '\n'
    all_ans += ans_str

with open('ans.txt', 'w', encoding='UTF-8') as f:
    f.write(all_ans)


# if website.is_valid:
#     print('爬虫成功')
#     scores, embeddings = model.predict(website)
#     label = max(zip(scores.values(), scores.keys()))[1]
#     if label == '正常':
#         print('使用黑名单再次检查')
#         label = classes[blackThinking(url)]
#
# else:
#     print('爬虫失败，采用xgboost')
#     label = classes[stage2_xgb(url)]
#
#     print("这个网址的类别是", label)
