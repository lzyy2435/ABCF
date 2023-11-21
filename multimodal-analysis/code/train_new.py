import collections
from sklearn import metrics
from src.homepage2vec.model import WebsiteClassifier, Webpage
# import itertools
# from matplotlib import pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, WeightedRandomSampler
# from Data_prepare import read_wiz_html, feature_path, prepare_train2, prepare_train3, read_test_wiz_html, \
#     feature_path_test
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
# from transformers import BertTokenizer
# from datetime import datetime
import csv

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class KaggleDataset(Dataset):
    def __init__(self):
        print('hh')
        self.kaggle_dataset = []  # num: 0; 1: url; 2:text; 3:label
        with open("../../dataset/website_classification.csv", newline='', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.kaggle_dataset.append(row)

        self.dset = [[int(s[0]), s[1], s[2], s[-1].replace(' ', '_').replace('/', '_').replace('-', '_')] for s in
                     self.kaggle_dataset if s[0] != '']
        # print(self.kaggle_dataset)
        self.H2V = WebsiteClassifier()

    def __getitem__(self, idx):
        num = int(self.dset[idx][0])
        url = self.dset[idx][1]
        label = self.dset[idx][-1]
        te = self.dset[idx][2]
        path = f"../../爬虫/kaggle_set/{label}/{num}.html"
        try:
            with open(path, encoding='utf-8') as f:
                html = f.read()
            # print(html)
            wp = Webpage(url)
            wp.html = html
            wp.is_valid = True
            if not os.path.exists(f'../kaggle_vectors/{label}/{num}.pth'):
                features = self.H2V.train_get_score(wp)
                torch.save(features.clone(), f'../kaggle_vectors/{label}/{num}.pth')
            else:
                features = torch.load(f'../kaggle_vectors/{label}/{num}.pth')
        except:
            # no html
            wp = Webpage(url)
            wp.html = None
            wp.is_valid = False
            # print('no html')
            if not os.path.exists(f'../kaggle_vectors/{label}/{num}.pth'):
                features = self.H2V.train_get_score(wp, text=[te])
                torch.save(features.clone(), f'../kaggle_vectors/{label}/{num}.pth')
            else:
                features = torch.load(f'../kaggle_vectors/{label}/{num}.pth')

        return features, url, label

    def __len__(self):
        return len(self.dset)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, st=False):
        self.st = st
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_max = 0
        self.delta = delta

    def __call__(self, f1, model, path):
        print("f1={}".format(f1))
        score = f1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1, model, path)
            self.counter = 0

    def save_checkpoint(self, f1, model, path):
        if self.verbose:
            print(
                f'F1 increased ({self.f1_max:.6f} --> {f1:.6f}).  Saving model ...model_Mf1_best.pth')
        if self.st:
            model.save(path)
        else:
            torch.save(model.state_dict(), os.path.join(path, 'model_dev.pth'))
        self.f1_max = f1


class KaggleInstructor:
    def __init__(self):
        # self.EntireArc = WebsiteClassifier()
        self.model = WebsiteClassifier(use_my_model=False).model_dev_ka.to(device)
        # self.model_dev = WebsiteClassifier(use_my_model=True).model_dev.to(device)
        # self.url_classifier = RawUrlClassifier(mode='bert')
        self.epochs = 50
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 3e-4
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.class_num = 16

    def main_train(self):
        early_stopping = EarlyStopping(patience=10, verbose=True)

        KD = KaggleDataset()

        train_size = int(len(KD) * 0.7)
        test_size = len(KD) - train_size
        # print(train_size)
        # print(test_size)

        train_dataset, test_dataset = torch.utils.data.random_split(KD, [train_size, test_size])

        # train_sampler = make_weights_for_balanced_classes_split(train_dataset)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=1)
        valid_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=1)

        train_data = []
        valid_data = []
        label_d = {'Adult': 0, 'Business_Corporate': 1, 'Computers_and_Technology': 2, 'E_Commerce': 3,
                   'Education': 4, 'Food': 5, 'Forums': 6, 'Games': 7, 'Health_and_Fitness': 8, 'Law_and_Government': 9,
                   'News': 10, 'Photography': 11, 'Social_Networking_and_Messaging': 12, 'Sports': 13,
                   'Streaming_Services': 14,
                   'Travel': 15}
        print('training')
        writer = SummaryWriter('./log_fix')
        for epoch in range(self.epochs):
            self.model.train()
            train_epoch_loss = []
            loop = tqdm(train_dataloader)

            for_save_list_train = []

if __name__ == "__main__":

    ins = KaggleInstructor()


    # for features, url, label in tqdm(train_dataloader):
    #     print(url[0], label_d[label[0]])
    #     train_data.append([url[0], label_d[label[0]]])
    #
    # for features, url, label in tqdm(valid_dataloader):
    #     print(url[0], label_d[label[0]])
    #     valid_data.append([url[0], label_d[label[0]]])
    #
    # with open('valid_kaggle.csv', 'w', encoding='utf-8', newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerows(valid_data)
    #
    # with open('train_kaggle.csv', 'w', encoding='utf-8', newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerows(train_data)
