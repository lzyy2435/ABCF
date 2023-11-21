import collections
from sklearn import metrics
from src.homepage2vec.model import WebsiteClassifier, Webpage
import itertools
from matplotlib import pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, WeightedRandomSampler
from Data_prepare import read_wiz_html, feature_path, prepare_train2, prepare_train3, read_test_wiz_html, \
    feature_path_test
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from datetime import datetime
import csv

import whois
from model import TransformerClassifier
from torch.nn.utils.rnn import pad_sequence
import time


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


# import os
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

url_err_right = list(csv.reader(open('url-errplace-right.csv', encoding='UTF-8')))


def make_weights_for_balanced_classes_split(dataset):
    N = len(dataset)
    W = [0.] * N
    weight_dict = {0: 0.5, 1: 1., 2: 1., 3: 9., 4: 1., 5: 1., 6: 1., 7: 20., 8: 3., 9: 3., 10: 10., 11: 13., 12: 1.}
    weight_dict_nohtml = {}
    for idx in tqdm(range(N)):
        W[idx] = weight_dict[dataset[idx][1]]
    Sampler = WeightedRandomSampler(W, num_samples=len(dataset))
    return Sampler


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig = plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        # plt.text(j, i, format(cm[i, j], fmt))

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return fig


class webpage_Dataset(Dataset):
    def __init__(self, mix=False, test=False):
        super(webpage_Dataset, self).__init__()
        self.test = test
        if test is False:
            self.fix_dict = {}
            for s in url_err_right:
                url, wrong, right = s
                wrong = int(wrong)
                right = int(right)
                self.fix_dict[url] = (wrong, right)
            self.url_path = list(read_wiz_html().items())  # 结构：url: (name, label, html_path/text)
            self.H2V = WebsiteClassifier()
            self.label_dict = collections.defaultdict(int)
            self.html_label_dict = collections.defaultdict(int)
            self.mix = mix
            if mix is True:
                train23_data = prepare_train2(mode='combine') + prepare_train3(mode='combine')
                text_data = []

                for i, s in enumerate(train23_data):
                    text_data.append((s[0], (f'text-{i}', s[-1], s[1])))
                # print(len(text_data))
                self.text_data = random.sample(text_data, int(len(text_data) * 0.3))
                self.all_data = self.url_path + self.text_data
            else:
                print('no train2,3')
                self.all_data = self.url_path

            for w in self.all_data:
                self.label_dict[w[-1][1]] += 1
            for v in self.url_path:
                self.html_label_dict[v[-1][1]] += 1
        else:
            self.all_data = list(read_test_wiz_html().items())
            self.H2V = WebsiteClassifier()
        # print(self.label_dict)
        # print(self.all_data)

    def __getitem__(self, idx):
        if self.test is False:

            if self.all_data[idx][-1][-1] is not None:  # 有html或者有文本的的
                url = self.all_data[idx][0]
                label = self.all_data[idx][-1][1]
                # print(label)
                try:
                    # print(self.fix_dict[url])
                    new_label = self.fix_dict[url][-1]
                except:
                    new_label = label
                    pass
                    # print('no wrong')
                # print('\n')
                name = self.all_data[idx][-1][0]
                if not os.path.exists(f'{feature_path}/{label}/{name}.pth'):
                    # print(f'get {feature_path}/{label}/{name}.pth')
                    if name.split('-')[0] == 'text':
                        # print(name)
                        wp = Webpage(url)
                        wp.html = None
                        wp.is_valid = False
                        features = self.H2V.train_get_score(wp, self.all_data[idx][-1][-1])  # wp, text_list
                        torch.save(features.clone(), f'{feature_path}/{label}/{name}.pth')
                    else:
                        with open(self.all_data[idx][-1][-1], encoding='utf-8') as f:
                            html = f.read()
                        wp = Webpage(url)
                        wp.html = html
                        wp.is_valid = True
                        features = self.H2V.train_get_score(wp)
                        torch.save(features.clone(), f'{feature_path}/{label}/{name}.pth')
                else:
                    features = torch.load(f'{feature_path}/{label}/{name}.pth')
            else:  # 没有html的
                url = self.url_path[idx][0]
                label = self.url_path[idx][-1][1]
                name = self.url_path[idx][-1][0]
                try:
                    # print(self.fix_dict[url])
                    new_label = self.fix_dict[url][-1]
                except:
                    new_label = label
                    pass
                if not os.path.exists(f'{feature_path}/{label}/{name}.pth'):
                    wp = Webpage(url)
                    wp.is_valid = False
                    features = self.H2V.train_get_score(wp)
                    torch.save(features.clone(), f'{feature_path}/{label}/{name}.pth')
                else:
                    features = torch.load(f'{feature_path}/{label}/{name}.pth')
            return features, new_label, name, url
        else:
            url = self.all_data[idx][0]
            name = self.all_data[idx][-1][0]
            path = self.all_data[idx][-1][-1]
            if not os.path.exists(f'{feature_path_test}/{name}.pth'):
                with open(path, encoding='utf-8') as f:
                    html = f.read()
                wp = Webpage(url)
                wp.html = html
                wp.is_valid = True
                features = self.H2V.train_get_score(wp)
                torch.save(features.clone(), f'{feature_path_test}/{name}.pth')
            else:
                features = torch.load(f'{feature_path_test}/{name}.pth')
            return features, name, url

    def __len__(self):
        return len(self.all_data)


class Url_Dataset(Dataset):
    def __init__(self, w2idx, train_or_val='train'):
        super(Url_Dataset, self).__init__()
        self.flag = train_or_val
        self.raw_data = list(read_wiz_html().items())
        self.train_jiutian = []
        self.valid_jiutian = []

        with open("../../aligned_dataset/jiutian/train_mobile.csv", newline='', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.train_jiutian.append(row)
        with open("../../aligned_dataset/jiutian/valid_mobile.csv", newline='', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.valid_jiutian.append(row)

        self.w2idx = w2idx

    def __getitem__(self, idx):

        if self.flag == 'train':
            url = self.train_jiutian[idx][0]
            label = self.train_jiutian[idx][-1]
            url_idx = [self.w2idx[i] for i in url]
            return url_idx, label
        else:
            url = self.valid_jiutian[idx][0]
            label = self.valid_jiutian[idx][-1]
            url_idx = [self.w2idx[i] for i in url]
            return url_idx, label

    def __len__(self):
        if self.flag == 'train':
            return len(self.train_jiutian)
        else:
            return len(self.valid_jiutian)


def collate_fn(batch_data):
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    data_length = [len(xi[0]) for xi in batch_data]

    sent_seq = [torch.tensor(xi[0], dtype=torch.int32) for xi in batch_data]
    label = [int(xi[-1]) for xi in batch_data]
    padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0)

    return padded_sent_seq, data_length, torch.tensor(label, dtype=torch.int32)


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
            torch.save(model.state_dict(), os.path.join(path, 'model_transformer.pth'))
        self.f1_max = f1


class VanillaInstructor:
    def __init__(self):
        # self.EntireArc = WebsiteClassifier()
        self.model = WebsiteClassifier(use_my_model=False).model.to(device)
        # self.model_dev = WebsiteClassifier(use_my_model=True).model_dev.to(device)
        # self.url_classifier = RawUrlClassifier(mode='bert')
        self.epochs = 100
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 1e-4
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.class_num = 13

    def train_run(self):
        early_stopping = EarlyStopping(patience=10, verbose=True)

        all_data = webpage_Dataset()

        for i in tqdm(range(len(all_data))):
            _ = all_data[i]

        train_size = int(len(all_data) * 0.7)
        test_size = len(all_data) - train_size
        # print(train_size)
        # print(test_size)

        train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size])

        # train_sampler = make_weights_for_balanced_classes_split(train_dataset)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=8)
        valid_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=8)
        train_epochs_loss = []
        valid_epochs_loss = []

        print('training')
        # all_list = []
        # train_dict = collections.defaultdict(int)
        # for i in range(train_size):
        #     train_i = train_dataset[i]
        #     _, label, name, url = train_i
        #     all_list.append([url, label, name])
        #     train_dict[label] += 1
        # with open('train.csv', 'w', encoding='utf-8') as f:
        #     csv_writer = csv.writer(f)
        #     csv_writer.writerows(all_list)

        writer = SummaryWriter('./log_fix')
        for epoch in range(self.epochs):
            # self.model.train()
            self.model.train()
            train_epoch_loss = []
            loop = tqdm(train_dataloader)

            for_save_list_train = []

            for step, (batch_feature, batch_label, batch_names, batch_url) in enumerate(loop):
                batch_feature = batch_feature.to(device)
                bl = batch_label
                batch_label = torch.eye(self.class_num)[batch_label, :].to(device)
                # outputs, _ = self.model_dev(batch_feature)
                outputs, _ = self.model(batch_feature)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, batch_label)
                loss.backward()
                self.optimizer.step()

                train_epoch_loss.append(loss.item())
                for i in range(len(batch_label)):
                    # print([batch_url[i], batch_label[i].item(), outputs[i].item(), batch_names[i]])
                    for_save_list_train.append([batch_url[i], bl[i].item(), batch_names[i]])

                # n_correct += (torch.argmax(outputs, -1) == batch_label).sum().item()
                # n_total += len(outputs)
                if step % (len(train_dataloader) // 2) == 0:
                    print("epoch={}/{},{}/{}of train, loss={}".format(
                        epoch, self.epochs, step, len(train_dataloader), loss.item()))

            with open('train.csv', 'w', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(for_save_list_train)

            writer.add_scalar('epoch loss', np.average(train_epoch_loss), global_step=epoch)
            train_epochs_loss.append(np.average(train_epoch_loss))
            self.model.eval()
            valid_epoch_loss = []
            n_correct, n_total = 0, 0
            t_targets_all, t_outputs_all = None, None

            for_save_list = []

            for idx, (batch_feature, batch_label, batch_names, batch_url) in enumerate(valid_dataloader, 0):
                batch_feature = batch_feature.to(device)
                batch_label_loss = torch.eye(self.class_num)[batch_label, :].to(device)

                # outputs, _ = self.model_dev(batch_feature)
                outputs, _ = self.model(batch_feature)

                loss = self.criterion(outputs, batch_label_loss)

                outputs = torch.argmax(outputs, -1)
                # n_correct += (outputs == batch_label).sum().item()
                # n_total += len(outputs)

                for i in range(len(batch_label)):
                    # print([batch_url[i], batch_label[i].item(), outputs[i].item(), batch_names[i]])
                    for_save_list.append([batch_url[i], batch_label[i].item(), outputs[i].item(), batch_names[i]])
                # print([batch_url, batch_label, outputs, batch_names])
                # print(outputs)
                valid_epoch_loss.append(loss.item())

                if t_targets_all is None:
                    t_targets_all = batch_label
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, batch_label), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)

            # print(t_targets_all)
            # print(t_outputs_all)
            # acc = n_correct / n_total

            Mf1 = metrics.f1_score(t_targets_all.cpu().tolist(), t_outputs_all.cpu().tolist(), labels=[0, 1, 2, 3, 4, 5,
                                                                                                       6, 7, 8, 9, 10,
                                                                                                       11, 12],
                                   average='macro', zero_division=1)
            mf1 = metrics.f1_score(t_targets_all.cpu().tolist(), t_outputs_all.cpu().tolist(), labels=[0, 1, 2, 3, 4, 5,
                                                                                                       6, 7, 8, 9, 10,
                                                                                                       11, 12],
                                   average='micro', zero_division=1)
            m_recall = metrics.recall_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='micro', zero_division=1)
            m_precision = metrics.precision_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='micro',
                                                  zero_division=1)

            M_recall = metrics.recall_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='macro', zero_division=1)
            M_precision = metrics.precision_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='macro',
                                                  zero_division=1)

            cm = metrics.confusion_matrix(t_targets_all.cpu(), t_outputs_all.cpu())

            # cm_plot = plot_confusion_matrix(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            writer.add_scalars('valid acc', {'macro f1': Mf1, 'micro f1': mf1,
                                             'micro recall': m_recall, 'micro precision': m_precision,
                                             'Macro recall': M_recall, 'Macro precision': M_precision},
                               global_step=epoch)
            # writer.add_figure('confusion matrix', cm_plot, global_step=epoch)

            print(f'epoch:{epoch} Macro-f1:{Mf1} micro-f1:{mf1} macro-recall:{M_recall} macro-precision:{M_precision} '
                  f'micro-recall:{m_recall} micro-precision:{m_precision}')

            valid_epochs_loss.append(np.average(valid_epoch_loss))
            early_stopping(Mf1, model=self.model, path='./model_result')

            if Mf1 >= early_stopping.best_score:
                # print(for_save_list)
                with open('valid.csv', 'w', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(for_save_list)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    @staticmethod
    def make_dict():
        print('make dict')
        all_data = webpage_Dataset()
        dictionary = collections.defaultdict(int)
        for _, _label, _names, _url in tqdm(all_data):
            for c in _url:
                dictionary[c] += 1

        vocab_ordered = [i[0] for i in sorted(dictionary.items(), key=lambda x: x[1], reverse=True)]
        vocab2idx = {vocab_ordered[i]: i for i in range(len(vocab_ordered))}
        return vocab2idx, len(vocab2idx.items())

    def try_func(self):
        print('try')
        all_data = webpage_Dataset()

        train_size = int(len(all_data) * 0.7)
        test_size = len(all_data) - train_size

        for i in tqdm(range(len(all_data))):
            _ = all_data[i]
        # print(train_size)
        # print(test_size)

        train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size])

        # train_sampler = make_weights_for_balanced_classes_split(train_dataset)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=1)
        valid_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=1)

        train_data = []
        valid_data = []

        loop = tqdm(train_dataloader)

        for step, (_feature, _label, _names, _url) in enumerate(loop):
            # print(_url[0], _label.item())
            train_data.append([_url[0], _label.item()])

        with open('valid_mobile.csv', 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(train_data, )

    def test_model(self, voc_size, voc2idx):
        # self.model.eval()
        # all_data = webpage_Dataset(mix=True)
        # train_size = int(len(all_data) * 0.7)
        # test_size = len(all_data) - train_size

        model_t = TransformerClassifier(dropout=0.2, vocab_size=voc_size, num_layers=3).to(device)
        model_t.eval()
        train_dataset = Url_Dataset(voc2idx)
        valid_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        #
        # train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size])

        # train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False, num_workers=1)
        # valid_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=1)

        pred_list = []
        ans_list = []
        all_list = []
        T1 = time.time()
        iter = 1000
        for test_i in tqdm(valid_dataloader):
            if iter <= 0:
                break
            batch_url, batch_len, batch_label = test_i
            batch_url = batch_url.to(device)
            # print(label, feature, name)
            pred = model_t(batch_url)
            pred = torch.argmax(pred, -1).item()

            ans_list.append(batch_label)
            pred_list.append(pred)
            iter -= 1

            # all_list.append([url, label, pred, name])
            # print(name)
        T2 = time.time()
        # Mf1 = metrics.f1_score(ans_list, pred_list, labels=[0, 1, 2, 3, 4, 5,
        #                                                     6, 7, 8, 9, 10,
        #                                                     11, 12],
        #                        average='macro', zero_division=1)
        # mf1 = metrics.f1_score(ans_list, pred_list, labels=[0, 1, 2, 3, 4, 5,
        #                                                     6, 7, 8, 9, 10,
        #                                                     11, 12],
        #                        average='micro', zero_division=1)
        print((T2 - T1)*1000)
        # print(Mf1, mf1)
        # with open('valid_origin.csv', 'w', encoding='utf-8') as f:
        #     csv_writer = csv.writer(f)
        #     csv_writer.writerows(all_list)

    def get_result(self):
        self.model.eval()
        D = webpage_Dataset(test=True)
        print(len(D))
        all_list = []
        label_dict = collections.defaultdict(int)
        for i in tqdm(range(len(D))):
            features, name, url = D[i]
            features = features.to(device)

            pred, _ = self.model(features)
            pred = torch.argmax(pred, -1).item()
            all_list.append([url, name, pred])
            label_dict[pred] += 1
        print(label_dict)
        with open('available_result_new_fix.csv', 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(all_list)

    def sort_by_time(self):
        all_data = webpage_Dataset()

        have_whois = []
        have_no_whois = []
        all_list = []
        for _feature, _label, _names, _url in tqdm(all_data):
            # print(_label, _names, _url)
            all_list.append([_label, _names, _url])
            # info = get_whois_info(_url)
            # if info is not None:
            #     have_whois.append([_label, _names, _url])
            # else:
            #     have_no_whois.append([_label, _names, _url])

        with open('whois_mobile.csv', 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(have_whois, )

        with open('no_whois_mobile.csv', 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(have_no_whois, )

    def pure_url_experiment(self, voc2idx, voc_size):
        early_stopping = EarlyStopping(patience=10, verbose=True)

        train_dataset = Url_Dataset(voc2idx)
        test_dataset = Url_Dataset(voc2idx, 'valid')

        # train_sampler = make_weights_for_balanced_classes_split(train_dataset)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        valid_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        model_t = TransformerClassifier(dropout=0.2, vocab_size=voc_size, num_layers=3).to(device)
        self.optimizer = torch.optim.AdamW(model_t.parameters(), lr=self.lr)

        for epoch in range(self.epochs):

            model_t.train()
            train_epoch_loss = []
            loop = tqdm(train_dataloader)

            for step, (batch_url, batch_len, batch_label) in enumerate(loop):
                batch_label = torch.eye(self.class_num)[batch_label, :].to(device)
                batch_url = batch_url.to(device)
                # outputs, _ = self.model_dev(batch_feature)
                outputs = model_t(batch_url)

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, batch_label)
                loss.backward()
                self.optimizer.step()

                train_epoch_loss.append(loss.item())

                # n_correct += (torch.argmax(outputs, -1) == batch_label).sum().item()
                # n_total += len(outputs)
                if step % (len(train_dataloader) // 2) == 0:
                    print("epoch={}/{},{}/{}of train, loss={}".format(
                        epoch, self.epochs, step, len(train_dataloader), loss.item()))

            model_t.eval()
            valid_epoch_loss = []
            n_correct, n_total = 0, 0
            t_targets_all, t_outputs_all = None, None

            for_save_list = []

            for idx, (batch_url, batch_len, batch_label) in enumerate(valid_dataloader, 0):
                batch_label_loss = torch.eye(self.class_num)[batch_label, :].to(device)

                # outputs, _ = self.model_dev(batch_feature)
                batch_url = batch_url.to(device)
                outputs = model_t(batch_url)

                loss = self.criterion(outputs, batch_label_loss)

                outputs = torch.argmax(outputs, -1)
                # n_correct += (outputs == batch_label).sum().item()
                # n_total += len(outputs)

                # print([batch_url, batch_label, outputs, batch_names])
                # print(outputs)
                valid_epoch_loss.append(loss.item())

                if t_targets_all is None:
                    t_targets_all = batch_label
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, batch_label), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)

            # print(t_targets_all)
            # print(t_outputs_all)
            # acc = n_correct / n_total

            Mf1 = metrics.f1_score(t_targets_all.cpu().tolist(), t_outputs_all.cpu().tolist(), labels=[0, 1, 2, 3, 4, 5,
                                                                                                       6, 7, 8, 9, 10,
                                                                                                       11, 12],
                                   average='macro', zero_division=1)
            mf1 = metrics.f1_score(t_targets_all.cpu().tolist(), t_outputs_all.cpu().tolist(), labels=[0, 1, 2, 3, 4, 5,
                                                                                                       6, 7, 8, 9, 10,
                                                                                                       11, 12],
                                   average='micro', zero_division=1)
            m_recall = metrics.recall_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='micro', zero_division=1)
            m_precision = metrics.precision_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='micro',
                                                  zero_division=1)

            M_recall = metrics.recall_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='macro', zero_division=1)
            M_precision = metrics.precision_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='macro',
                                                  zero_division=1)

            cm = metrics.confusion_matrix(t_targets_all.cpu(), t_outputs_all.cpu())

            # cm_plot = plot_confusion_matrix(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            # writer.add_figure('confusion matrix', cm_plot, global_step=epoch)

            print(f'epoch:{epoch} Macro-f1:{Mf1} micro-f1:{mf1} macro-recall:{M_recall} macro-precision:{M_precision} '
                  f'micro-recall:{m_recall} micro-precision:{m_precision}')

            early_stopping(Mf1, model=self.model, path='./model_result')
            if early_stopping.early_stop:
                print("Early stopping")
                break

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    ins = VanillaInstructor()
    c2idx, vocab_size = ins.make_dict()
    ins.test_model(vocab_size, c2idx)

    # targets = []
    # preds = []
    # valid = list(csv.reader(open('valid.csv', encoding='UTF-8')))
    # classes = {0: '正常', 1: '购物消费', 2: '婚恋交友', 3: '假冒身份', 4: '钓鱼网站', 5: '冒充公检法', 6: '平台诈骗',
    #            7: '招聘兼职',
    #            8: '杀猪盘', 9: '博彩赌博', 10: '信贷理财', 11: '刷单诈骗', 12: '中奖诈骗'}
    # for s in valid:
    #     targets.append(classes[int(s[1])])
    #     preds.append(classes[int(s[2])])
    #
    # cm = metrics.confusion_matrix(targets, preds,
    #                               labels=['正常', '购物消费', '婚恋交友', '假冒身份', '钓鱼网站', '冒充公检法',
    #                                       '平台诈骗', '招聘兼职',
    #                                       '杀猪盘', '博彩赌博', '信贷理财', '刷单诈骗', '中奖诈骗'])
    # disp = metrics.ConfusionMatrixDisplay(cm, )
    # disp.plot(cmap=plt.cm.Blues)
    # plt.show()
    #
    # ins.try_func()
    # ins.test_model()
    # ins.train_run()

    # D = webpage_Dataset()
    # s = Url_Dataset()
    # make_weights_for_balanced_classes_split(D)
    # print(D.label_dict)
    # print(len(D))
    # for i in range(len(D)):
    #     print(D[i])
    # print(len(D))
