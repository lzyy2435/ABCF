import gensim
from gensim import corpora
# import pyLDAvis.gensim


import time

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
from tqdm import tqdm
from bs4 import BeautifulSoup
import jieba
import jieba.posseg as jp

import re
import csv

import openai

# 我的api key，额度应该还够，7月1号过期
openai.api_key = "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"

from Data_prepare import read_wiz_html

warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel


class Chat:
    def __init__(self, conversation_list=[]) -> None:
        # 初始化对话列表，可以加入一个key为system的字典，有助于形成更加个性化的回答
        self.conversation_list = []
        # openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.conversation_list)
        self.prompt_list = [{'role': 'system', 'content': '用户将输入一系列主题词以及其显著性得分例如：贷款*0'
                                                          '.01 + 0.012*银行，你要用精炼的语言总结这些词所属主题，字数限制在5到10个，不要有标点'
                                                          '尽量与诈骗、色情、敏感言论等有害网站相关'}]

    # 打印对话
    @staticmethod
    def show_conversation(msg_list):
        describe_list = []
        for msg in msg_list:
            if msg['role'] == 'user':
                print(f"\U0001f47b: {msg['content']}\n")
            elif msg['role'] == 'assistant':
                describe_list.append(msg['content'])
                print(f"\U0001f47D: {msg['content']}\n")
        return describe_list

    # 提示chatgpt
    def ask(self, prompt, show=False):
        self.conversation_list.append({"role": "user", "content": prompt})

        self.prompt_list.append({"role": "user", "content": prompt})
        # print(self.prompt_list)
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.prompt_list)
        self.prompt_list.pop()

        answer = response.choices[0].message['content']
        # 下面这一步是把chatGPT的回答也添加到对话列表中，这样下一次问问题的时候就能形成上下文了
        self.conversation_list.append({"role": "assistant", "content": answer})

        if show is True:
            return self.show_conversation(self.conversation_list)


def split_in_sentences(soup):
    """ From the raw html content of a website, extract the text visible to the user and splits it in sentences """

    sep = soup.get_text('[SEP]').split('[SEP]')  # separate text elements with special separators [SEP]
    strip = [s.strip() for s in sep if s != '\n']
    clean = [s for s in strip if len(s) != 0]
    # 去除非中英
    clean_zh = [clean_zh_text(i) for i in clean if len(i) != 0]

    return clean_zh


# make Chinese text clean
def clean_zh_text(text):
    # keep English, digital and Chinese
    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    return comp.sub('', text)


flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
stopwords = ('不', '没', '就', '的', '是', '才', '但', '第', '集', '期', '准', '网站', '已', '开')  # 停词

if __name__ == '__main__':
    # train2_data = list(csv.reader(open('../../dataset/train2.csv', encoding='UTF-8')))
    # train3_data = list(csv.reader(open('../../dataset/train3.csv', encoding='gbk')))
    # all_data_path = list(read_wiz_html().items())
    # 可能用更多

    sus_pool = list(csv.reader(open('./suspect_pool_new.txt', encoding='UTF-8')))
    show_data_path = [path.strip('\n') for path in list(open('./path.txt', encoding='UTF-8'))]
    urls_idx = [int(path.split('.')[0].split('-')[-1]) for path in show_data_path]
    urls = [sus_pool[idx] for idx in urls_idx]

    # print(show_data_path)
    # print(urls)

    soups = []
    for item in show_data_path:
        path = item
        with open(path, encoding='utf-8') as f:
            html = f.read()
            soup = BeautifulSoup(html, 'lxml')
        soups.append(soup)

    corpus = []
    for s in soups:
        clean_sentences = split_in_sentences(s)
        corpus.append("".join(clean_sentences))

    # 分词
    corpus_cut = []
    for c in tqdm(corpus):
        cut = [w.word for w in jp.cut(c) if w.flag in flags and w.word not in stopwords]  # 可以做一些筛选
        corpus_cut.append(cut)
    corpus_cut = [x for x in corpus_cut if x != []]

    # make balance
    avg_len = 0
    for i in corpus_cut:
        avg_len += len(i)
    avg_len = int(avg_len / len(corpus_cut))
    for i, _ in enumerate(corpus_cut):
        if len(corpus_cut[i]) > avg_len:
            corpus_cut[i] = corpus_cut[i][:avg_len]
        elif len(corpus_cut[i]) < 0.5 * avg_len:
            corpus_cut[i] = corpus_cut[i] * int(avg_len / len(corpus_cut[i]))

    # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
    dictionary = corpora.Dictionary(corpus_cut)

    # 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus_cut]  # 文档(行)×词(列)

    # choose best num of topics
    lda = LdaModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=8)

    # for i in range(1, 15):
    #     lda_m = LdaModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=i)
    #     ldacm = CoherenceModel(model=lda_m, texts=corpus_cut, dictionary=dictionary, coherence='c_v')
    #     print(i, ldacm.get_coherence())

    topic_list = lda.print_topics()
    for o in topic_list:
        print(o)

    url_topic_dict = {}
    for e, values in enumerate(lda.inference(doc_term_matrix)[0]):
        print(urls[e])
        max_val = 0
        candidate = 0
        for ee, value in enumerate(values):
            print('\t主题%d推断值%.2f' % (ee, value))
            if value > max_val:
                max_val = value
                candidate = ee
        url_topic_dict[urls[e][0]] = topic_list[candidate]

    print(url_topic_dict)
    # prompt

    # chat_helper = Chat()
    # des_list = None
    # for i in range(len(urls)):
    #     show = False if i != len(urls) - 1 else True
    #     time.sleep(20)  # 太频繁会超出限制
    #     print('wait for 20 sec')
    #     des_list = chat_helper.ask(list(url_topic_dict.items())[i][-1][-1].replace('\"', ''), show)
    #
    # print(des_list)

    # pyLDAvis.enable_notebook()
    # data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    # pyLDAvis.save_html(data, 'E:/data/3topic.html')



