import csv
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from tqdm import tqdm
from src.homepage2vec.textual_extractor import *
import collections
from Stage2_XGBoost import stage2_xgb
from Stage1 import stage1
from 黑白名单 import blackThinking

import whois

def get_whois_info(domain):
    try:
        info = whois.whois(domain)  # Info返回了所有的whois查询信息，可根据需要选择想要提取的查询方法
        whoisInfo = {}
        whoisInfo.update({'registrar': info['registrar']})
        whoisInfo.update({'emails': info['emails']})
        whoisInfo.update({'creationDate': str(info['creation_date'][0])})
        whoisInfo.update({'updatedDate': str(info['updated_date'][-1])})
        whoisInfo.update({'expirationDate': str(info['expiration_date'][0])})
        return whoisInfo
    except:
        return None

LIMIT = 50000

root_path = '../../爬虫/url_data'
root_path_test = '../../爬虫/test_url_data'
root_path_ip = '../../爬虫/ip_data/ip_data'
sus_root = '../../dataset/suspect_pool.txt'
rep_root = '../../dataset/train1补充.xlsx'

test_path = '../../dataset/url_suspect_test.txt'
ip_test_path = '../../dataset/ip_pool_test.txt'

feature_path = '../feature_vectors'

feature_path_new_xlm = '../feature_vectors_new'
feature_path_test = '../feature_vec_test'

exp_html_path = '../../爬虫/url_data/9/sus-9855.html'

label_dict = {'正常网址': 0, '购物消费': 1, '婚恋交友': 2, '假冒身份': 3, '钓鱼网站': 4, '冒充公检法': 5,
              '平台诈骗': 6, '招聘兼职': 7, '杀猪盘': 8, '博彩赌博': 9, '信贷理财': 10, '刷单诈骗': 11, '中奖诈骗': 12}
train3_label_dict = {'1': 11, '2': 10}


# tokenizer = SentenceTransformer('paraphrase-xlm-r-multilingual-v1').tokenizer
# max_seq_len = SentenceTransformer('paraphrase-xlm-r-multilingual-v1').max_seq_length

def read_train1(fdir):
    urls_label = []
    df = pd.read_excel(io=fdir)
    for i, row in df.iterrows():
        urls_label.append((row['域名'], row['诈骗类型']))
    return urls_label


def prepare_train2(mode='split'):
    train2_data = list(csv.reader(open('../../dataset/train2.csv', encoding='UTF-8')))
    result = []
    for e in tqdm(train2_data):
        if e[2] == '类型':
            continue
        trunc_list = []
        len_text = len(e[1])
        if len_text <= 256:
            trunc_list.append(e[1])
        elif 256 < len_text <= 512:
            split_num = len_text // 2
            for sample_i in range(0, 2):
                trunc_list.append(e[1][sample_i * split_num: sample_i * split_num + 128])

        elif 512 < len_text <= 1024:
            split_num = len_text // 4
            for sample_i in range(0, 4):
                trunc_list.append(e[1][sample_i * split_num: sample_i * split_num + 128])

        else:
            split_num = len_text // 8
            for sample_i in range(0, 8):
                trunc_list.append(e[1][sample_i * split_num: sample_i * split_num + 128])

        # tok_s = [trunc(s, tokenizer, max_seq_len) for s in trunc_list]
        # for sentence in trunc_list:
        #     result.append([e[0], sentence, label_dict[e[2]]])

        # if mode == 'split':
        #     tok_s = [trunc(s, tokenizer, max_seq_len) for s in trunc_list]
        #     for sentence in tok_s:
        #         result.append([e[0], sentence, label_dict[e[2]]])
        if mode == 'combine':
            result.append([e[0], trunc_list, label_dict[e[2]]])

    return result


def prepare_train3(mode='split'):
    data3 = list(csv.reader(open('../../dataset/train3.csv', encoding='gbk')))
    result = []
    for e in tqdm(data3):
        trunc_list = []
        len_text = len(e[1])
        if len_text <= 256:
            trunc_list.append(e[1])
        elif 256 < len_text <= 512:
            split_num = len_text // 2
            for sample_i in range(0, 2):
                trunc_list.append(e[1][sample_i * split_num: sample_i * split_num + 128])
        elif 512 < len_text <= 1024:
            split_num = len_text // 4
            for sample_i in range(0, 4):
                trunc_list.append(e[1][sample_i * split_num: sample_i * split_num + 128])
        else:
            split_num = len_text // 8
            for sample_i in range(0, 8):
                trunc_list.append(e[1][sample_i * split_num: sample_i * split_num + 128])

        # if mode == 'split':
        #     tok_s = [trunc(s, tokenizer, max_seq_len) for s in trunc_list]
        #     for sentence in tok_s:
        #         result.append([e[0], sentence, train3_label_dict[e[2]]])
        if mode == 'combine':
            result.append([e[0], trunc_list, train3_label_dict[e[2]]])

    return result


# 包含train1的一部分以及train1补充
def read_wiz_html(get_full=False):
    sus_pool = list(csv.reader(open(sus_root, encoding='UTF-8')))[:LIMIT]
    rep_pool = read_train1(rep_root)

    train_data_dict = {}  # 结构：url: (name, label, html_path)

    files = os.listdir(root_path)
    for file in files:
        file_path = os.path.join(root_path, file)
        label = file_path.split('/')[-1]
        html_files = os.listdir(file_path)
        for html in html_files:
            html_path = os.path.join(file_path, html)  # for read
            dataset, num = html.split('.')[0].split('-')
            num = int(num)
            # print(dataset, num)
            if dataset == 'sus':
                # print(sus_pool[num], label)
                assert sus_pool[num][-1] == label
                if sus_pool[num][0] in train_data_dict:
                    Warning('warning, duplicate', label)
                    print(sus_pool[num][0])
                train_data_dict[sus_pool[num][0]] = (html, int(label), html_path)
            elif dataset == 'rep':
                assert rep_pool[num][-1] == int(label)
                if rep_pool[num][0] in train_data_dict:
                    Warning('warning, duplicate')
                    print(sus_pool[num][0], label)
                train_data_dict[rep_pool[num][0]] = (html, int(label), html_path)
            else:
                print('Unknown Dataset')
                return

    if get_full is True:
        for sus_i in range(len(sus_pool)):
            if sus_pool[sus_i][0] not in train_data_dict:
                train_data_dict[sus_pool[sus_i][0]] = (f'nohtml_sus{sus_i}', int(sus_pool[sus_i][-1]), None)

        for rep_i in range(len(rep_pool)):
            if rep_pool[rep_i][0] not in train_data_dict:
                train_data_dict[rep_pool[rep_i][0]] = (f'nohtml_rep{rep_i}', int(sus_pool[rep_i][-1]), None)

    return train_data_dict


def read_test_wiz_html():
    url_pool_test = [i[0] for i in list(csv.reader(open(test_path, encoding='UTF-8')))]
    ip_pool_test = [i[0] for i in list(csv.reader(open(ip_test_path, encoding='UTF-8')))]

    test_data_dict = {}  # 结构：url: (name, html_path)

    files = os.listdir(root_path_test)
    for file in files:
        file_path = os.path.join(root_path_test, file)
        num_in_url = int(file.split('.')[0])
        test_data_dict[url_pool_test[num_in_url]] = ('url-' + file.split('.')[0], file_path)

    files_ip = os.listdir(root_path_ip)
    for f_ip in files_ip:
        file_path_ip = os.path.join(root_path_ip, f_ip)
        num_in_ip = int(f_ip.split('.')[0])
        test_data_dict[ip_pool_test[num_in_ip]] = ('ip-' + f_ip.split('.')[0], file_path_ip)

    # print(test_data_dict)
    return test_data_dict


def extract_tld(url_pool):
    tld_set = collections.defaultdict(int)
    for url in url_pool:
        tld_set[url.split('.')[-1].split(':')[0]] += 1

    tld_set_sorted = sorted(
        tld_set.items(),
        key=lambda item: item[-1],
        reverse=True
    )
    tld_set_top = tld_set_sorted[:27]
    print(tld_set_top)
    print('com' in tld_set)

    with open('./tld_top27_freq_full.txt', 'w') as result:
        for i in tld_set_top:
            result.write(f'{i[0]},{i[1]}\n')


def make_test():
    url_pool_test = set([i[0] for i in list(csv.reader(open(test_path, encoding='UTF-8')))])
    ip_pool_test = set([i[0] for i in list(csv.reader(open(ip_test_path, encoding='UTF-8')))])

    avail_result = list(csv.reader(open('available_result_new.csv', encoding='UTF-8')))

    for addr, name, label in avail_result:
        dset = name.split('-')[0]
        if dset == 'url':
            assert addr in url_pool_test
            url_pool_test.remove(addr)
            url_pool_test.add((addr, label))
        elif dset == 'ip':
            assert addr in ip_pool_test
            ip_pool_test.remove(addr)
            ip_pool_test.add((addr, label))

    # url_pool_test = list(url_pool_test)
    # ip_pool_test = list(ip_pool_test)
    # for ele in tqdm(range(len(url_pool_test))):
    #     if isinstance(url_pool_test[ele], str):
    #         label = stage2_xgb(url_pool_test[ele])


if __name__ == '__main__':
    html_yes_data = []
    with open("whois_mobile.csv", newline='', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for row in reader:
            html_yes_data.append(row)
    # print(html_yes_data)

    have_whois = []
    have_no_whois = []
    for label, filename, url in html_yes_data:
        info = get_whois_info(url)
        print(info)
        if info is not None:
            have_whois.append([label, filename, url])
        else:
            have_no_whois.append([label, filename, url])

    # with open('whois_mobile.csv', 'w', encoding='utf-8', newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerows(have_whois, )
    # with open('no_whois_mobile.csv', 'w', encoding='utf-8', newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerows(have_no_whois, )



    # fdl = ['../../dataset/train1.csv', '../../dataset/train1补充.xlsx']
    # f2 = '../../dataset/train2.csv'
    # x = read_train1(fdl)
    # TDATA = read_wiz_html()

    # url_pool_sus = list(csv.reader(open(sus_root, encoding='UTF-8')))
    # # url_pool_test = [i[0] for i in list(csv.reader(open(test_path, encoding='UTF-8')))]
    # url_pool_rep = read_train1(rep_root)
    # urls = []
    # for i in url_pool_sus:
    #     urls.append(i[0])
    # for j in url_pool_rep:
    #     urls.append(j[0])
    # urls += url_pool_test
    # extract_tld(urls)

    # prepare_html(exp_html_path)
    # read_test_wiz_html()
