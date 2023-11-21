import collections
from sklearn import metrics
from src.homepage2vec.model import WebsiteClassifier, Webpage
import itertools
from matplotlib import pyplot as plt
import os, sys
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

# import os
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if len(sys.argv) >= 2:
    MODEL_TYPE = int(sys.argv[1])
else:
    MODEL_TYPE = 1

if len(sys.argv) >= 3:
    BATCH_SIZE = int(sys.argv[2])
else:
    BATCH_SIZE = 4

MODEL_NAME = {1: "纯笛卡尔积  resnet18", 2: "笛卡尔积和线性层  resnet18", 3: "ABC  纯MLP", 4: "2+reLU", 5: "无FEA"}

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


class noWhoisDataset(Dataset):
    def __init__(self):
        print('hh')
        self.no_whois_dataset = []  # num: 0; 1: url; 2:text; 3:label
        with open("no_whois_mobile.csv", newline='', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.no_whois_dataset.append(row)

        # print(self.kaggle_dataset)
        self.H2V = WebsiteClassifier()

    def __getitem__(self, idx):
        label = self.no_whois_dataset[idx][0]
        name = self.no_whois_dataset[idx][1]
        url = self.no_whois_dataset[idx][2]
        path_to_vector = f'../feature_vectors/{label}/{name}.pth'
        features = torch.load(path_to_vector)

        return features, label, name, url

    def __len__(self):
        return len(self.no_whois_dataset)


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
                f'F1 increased ({self.f1_max:.6f} --> {f1:.6f}).  Saving model ...model_new{MODEL_TYPE}_dev.pth')
        if self.st:
            model.save(path)
        else:
            # 存在了这里
            torch.save(model.state_dict(), os.path.join(path, f'model_new{MODEL_TYPE}_dev.pth'))
        self.f1_max = f1


class VanillaInstructor:
    def __init__(self):
        # self.EntireArc = WebsiteClassifier()
        # self.model = WebsiteClassifier(use_my_model=True).model.to(device)
        self.model_dev = WebsiteClassifier(use_my_model=True, try_type=MODEL_TYPE).model_dev.to(device)

        if os.path.exists(f'./model_result/model_new{MODEL_TYPE}_dev.pth'):
            print("已加载训练模型，开始断点续训")
            state_dict = torch.load(f'./model_result/model_new{MODEL_TYPE}_dev.pth')
            if MODEL_TYPE != 3:
                # print(state_dict.keys())
                state_dict.pop('layer1.weight')
                state_dict.pop('layer1.bias')
            self.model_dev.load_state_dict(state_dict, strict=False)
        else:
            print("未找到已训练模型，从头开始训练")

        # self.url_classifier = RawUrlClassifier(mode='bert')
        self.epochs = 100
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = 3e-4
        self.optimizer = torch.optim.Adam(self.model_dev.parameters(), lr=self.lr)
        self.class_num = 13

    def train_run(self):
        early_stopping = EarlyStopping(patience=20, verbose=True)

        all_data = noWhoisDataset()

        for i in tqdm(range(len(all_data))):
            _ = all_data[i]

        train_size = int(len(all_data) * 0.7)
        test_size = len(all_data) - train_size
        # print(train_size)
        # print(test_size)

        train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size])

        # train_sampler = make_weights_for_balanced_classes_split(train_dataset)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        valid_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
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
            self.model_dev.train()
            train_epoch_loss = []
            loop = tqdm(train_dataloader)

            for_save_list_train = []

            for step, (batch_feature, batch_label, batch_names, batch_url) in enumerate(loop):
                batch_feature = batch_feature.to(device)
                bl = batch_label
                batch_label = torch.eye(self.class_num)[batch_label, :].to(device)
                # outputs, _ = self.model_dev(batch_feature)
                outputs = self.model_dev(batch_feature)

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
            self.model_dev.eval()
            valid_epoch_loss = []
            n_correct, n_total = 0, 0
            t_targets_all, t_outputs_all = None, None

            for_save_list = []

            for idx, (batch_feature, batch_label, batch_names, batch_url) in enumerate(valid_dataloader, 0):
                batch_feature = batch_feature.to(device)
                batch_label_loss = torch.eye(self.class_num)[batch_label, :].to(device)

                # outputs, _ = self.model_dev(batch_feature)
                outputs = self.model_dev(batch_feature)

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

            acc = metrics.accuracy_score(t_targets_all.cpu(), t_outputs_all.cpu())
            # cm_plot = plot_confusion_matrix(cm, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            writer.add_scalars('valid acc', {'macro f1': Mf1, 'micro f1': mf1,
                                             'micro recall': m_recall, 'micro precision': m_precision,
                                             'Macro recall': M_recall, 'Macro precision': M_precision},
                               global_step=epoch)
            # writer.add_figure('confusion matrix', cm_plot, global_step=epoch)

            print(
                f'epoch:{epoch} acc:{acc} Macro-f1:{Mf1} micro-f1:{mf1} macro-recall:{M_recall} macro-precision:{M_precision} '
                f'micro-recall:{m_recall} micro-precision:{m_precision}')

            with open(f"./log_res_csv/log_res_{MODEL_NAME[MODEL_TYPE]}.csv", "a", encoding="UTF-8") as f:
                f.write(
                    f'epoch,{epoch},acc,{acc},Macro-f1,{Mf1},micro-f1:{mf1},macro-recall,{M_recall},macro-precisio,{M_precision},'
                    f'micro-recall,{m_recall},micro-precision,{m_precision}' + "\n")

            valid_epochs_loss.append(np.average(valid_epoch_loss))
            early_stopping(Mf1, model=self.model_dev, path='./model_result')

            if Mf1 >= early_stopping.best_score:
                # print(for_save_list)
                with open('valid.csv', 'w', encoding='utf-8') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(for_save_list)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    def try_func(self):
        print('try')
        torch.set_printoptions(profile="full")
        all_data = noWhoisDataset()

        for i in range(len(all_data)):
            # print(all_data[i])
            _ = all_data[i]
            pass

        # print(all_data.label_dict)
        # print(all_data.html_label_dict)
        try_dataloader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

        for step, (batch_feature, batch_label, batch_names) in enumerate(try_dataloader):
            print(self.model_dev(batch_feature))

    def test_model(self):
        self.model_dev.eval()
        all_data = noWhoisDataset()
        train_size = int(len(all_data) * 0.7)
        test_size = len(all_data) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size])

        # train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False, num_workers=1)
        # valid_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=1)

        pred_list = []
        ans_list = []
        all_list = []
        for i in tqdm(range(test_size)):
            test_i = test_dataset[i]
            feature, label, name, url = test_i
            feature = feature.to(device)
            # print(label, feature, name)
            pred, _ = self.model_dev(feature)
            pred = torch.argmax(pred, -1).item()

            ans_list.append(label)
            pred_list.append(pred)

            all_list.append([url, label, pred, name])
            # print(name)
        Mf1 = metrics.f1_score(ans_list, pred_list, labels=[0, 1, 2, 3, 4, 5,
                                                            6, 7, 8, 9, 10,
                                                            11, 12],
                               average='macro', zero_division=1)
        mf1 = metrics.f1_score(ans_list, pred_list, labels=[0, 1, 2, 3, 4, 5,
                                                            6, 7, 8, 9, 10,
                                                            11, 12],
                               average='micro', zero_division=1)
        print(Mf1, mf1)
        with open('valid_origin.csv', 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(all_list)

    def get_result(self):
        self.model_dev.eval()
        D = noWhoisDataset()
        print(len(D))
        all_list = []
        label_dict = collections.defaultdict(int)
        for i in tqdm(range(len(D))):
            features, name, url = D[i]
            features = features.to(device)

            pred, _ = self.model_dev(features)
            pred = torch.argmax(pred, -1).item()
            all_list.append([url, name, pred])
            label_dict[pred] += 1
        print(label_dict)
        with open('available_result_new_fix.csv', 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(all_list)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    ins = VanillaInstructor()
    ins.train_run()

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

'''kaggle_dataset = []  # 1: url; 2:text; 3:label
with open("../../dataset/website_classification.csv", newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        kaggle_dataset.append(row)

print(kaggle_dataset)'''
