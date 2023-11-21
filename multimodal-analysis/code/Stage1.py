# -*- coding: utf-8 -*-
# @Time    : 2023-04-16 0:21
# @Author  : Wei Liu
# @ID      : 2020212172
# @File    : Stage1.py
import collections
import math
import joblib
import sklearn
import numpy as np


def getDomainFeature(domain: str):
    feature = [domain.count('.'), len(domain), sum(c.isdigit() for c in domain), domain.count('-')]

    count_c = 0
    special_characters = (':', ';', '#', '!', '%', '~', '+', '_', '?', '=', '&', '[', ']')
    for c in domain:
        if c in special_characters:
            count_c = count_c + 1
    feature.append(count_c)

    s = len(domain)
    dd = collections.defaultdict(int)
    for c in domain:
        dd[c] += 1

    # 字符随机性
    # H(d) = － ∑lg( P( Xi ) ) * P( Xi )
    feature.append(sum(map(lambda value: -(value / s) * math.log2(value / s), dd.values())))

    # 元音字母比例
    feature.append(sum(map(lambda x: dd[x], ('a', 'e', 'i', 'o', 'u'))) / s)

    # 唯一字符数
    feature.append(len(dd))

    return feature


def stage1(url: str, mode_type="Logistic"):
    """

    :param url:
    :return:
    """
    # 初始化为
    # 模型输出的0为嫌疑网址，1为正常网址

    type_stage1 = 0
    feature = getDomainFeature(url)

    if mode_type == "Logistic":
        Logistic = joblib.load('logistic_regression_model.pkl')
        type_stage1 = Logistic.predict([feature])[0]
    # 返回值中，1为嫌疑网址，0为正常网址，与后续类别一致
    return type_stage1


if __name__ == "__main__":
    url = r"au1.vjst.cc"
    if stage1(url) == 0:
        print("正常网址")
    else:
        print("进行下一步操作")
