# -*- coding: utf-8 -*-
# @Time    : 2023-04-16 0:21
# @Author  : Wei Liu
# @ID      : 2020212172
# @File    : stage2_XGBoost.py

import collections
import math
import joblib
import sklearn
import numpy as np
from xgboost.sklearn import XGBClassifier


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


XGB = XGBClassifier(min_child_weight=6, max_depth=10, scale_pos_weight=5, n_estimators=200,
                    objective='multi:softmax', num_class=13, silent=False)


def stage2_xgb(url: str):
    """

    :param url:
    :return:
    """
    # 初始化为

    feature = getDomainFeature(url)

    XGB.load_model("xgboost_50-part.json")
    type_stage2 = XGB.predict([feature])[0]
    # 返回值中，1为嫌疑网址，0为正常网址，与后续类别一致
    return type_stage2


if __name__ == "__main__":
    url = r"www.baidu.com"
    print(stage2_xgb(url))
