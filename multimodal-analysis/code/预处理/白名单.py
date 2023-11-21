import re
import csv


def getDomain(url):
    domain = re.search(r"(https?://)?(www\.)?(.+?)/.*", url+'/').group(3)
    return domain

def getWriteList(WriteListFile):
    d = {}
    for _ in csv.reader(open(WriteListFile)):
        temp = d
        for i in _[1].split('.')[::-1]:
            if i not in temp:
                temp[i] = {}
            temp = temp[i]
        temp['END'] = None
    return d

def isInWriteList(url,whiteList):
    domain = getDomain(url).split('.')[::-1]
    temp = whiteList
    for i in domain:
        if i not in temp:
            return "END" in temp
        temp = temp[i]
    return "END" in temp



if __name__ == "__main__":
    # 参数为白名单位置
    whiteList = getWriteList("domain/top-1m1.csv")


    TP = 0 #( Ture Positive )真阳性：预测为正，实际也为正
    FP = 0 #( False Positive )假阳性：预测为正，实际为负
    FN = 0 #( False Negative )假阴性：预测为负，实际为正
    N = 0

    print("以下为假阳性网站")
    for _ in csv.reader(open("data/train1.csv")):
        if len(_) >1 and _[1] == '0':
            N += 1
        if isInWriteList(_[0],whiteList):
            if _[1] == '0':
                TP += 1
            else:
                FP += 1
                print(_[0],_[1])
    FN = N - TP
    P = TP /(TP+FP) #( Precision )精确率 P = TP /（TP+FP）
    R = TP /(TP+FN) #( Recall )召回率
    print(f"对于数据集train1来说，一共有{N}个正常网站\n其中白名单判断了{TP}个真阳性网站，判断了{FP}个假阳性网站")
    print(f"仅供参考：\n精确率为{P*100:.2f}%，召回率为{R*100:.2f}%")

    print('\n\n')

    TP = 0 #( Ture Positive )真阳性：预测为正，实际也为正
    FP = 0 #( False Positive )假阳性：预测为正，实际为负
    FN = 0 #( False Negative )假阴性：预测为负，实际为正
    N = 0

    print("以下为假阳性网站")
    for _ in csv.reader(open("data/train2.csv",encoding='utf-8')):
        if not _[0][0].isalpha():
            continue
        if _[2] == '正常网址':
            N += 1
        if isInWriteList(_[0].replace(" ",""), whiteList):
            if _[2] == '正常网址':
                TP += 1
            else:
                FP += 1
                print(_[0], _[2])

    FN = N - TP
    P = TP /(TP+FP) #( Precision )精确率 P = TP /（TP+FP）
    R = TP /(TP+FN) #( Recall )召回率
    print(f"对于数据集train2来说，一共有{N}个正常网站\n其中白名单判断了{TP}个真阳性网站，判断了{FP}个假阳性网站")
    print(f"仅供参考：\n精确率为{P*100:.2f}%，召回率为{R*100:.2f}%")
