import logging
import random

import sys

sys.path.append(
    '/root/autodl-tmp/fraud-website-classification/homepage2vec-master/homepage2vec-master/src/homepage2vec')

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from model_loader import get_model_path
from textual_extractor import TextualExtractor
from visual_extractor import VisualExtractor
from data_collection import access_website, take_screenshot
import uuid
import tempfile
import os
import glob
import json
from importlib.util import find_spec
from sentence_transformers import SentenceTransformer
from transformers import BertModel


class WebsiteClassifier:
    """
    Pretrained Homepage2vec model
    """

    def __init__(self, visual=False, device=None, cpu_threads_count=1, dataloader_workers=1, use_my_model=False,
                 try_type=1):
        self.input_dim = 5177 if visual else 4665
        self.output_dim = 13
        self.output_dim_kaggle = 16
        self.classes = ['正常', '购物消费', '婚恋交友', '假冒身份', '钓鱼网站', '冒充公检法', '平台诈骗', '招聘兼职',
                        '杀猪盘', '博彩赌博', '信贷理财', '刷单诈骗', '中奖诈骗']

        if find_spec('selenium') is None and visual is True:
            raise RuntimeError('Visual flag requires the installation of the selenium package')
        self.with_visual = visual

        self.model_home_path, self.model_path = get_model_path(visual)

        self.temporary_dir = tempfile.gettempdir() + "/homepage2vec/"
        os.makedirs(self.temporary_dir + "/screenshots", exist_ok=True)
        # clean screen shorts
        files = glob.glob(self.temporary_dir + "/screenshots/*")
        for f in files:
            os.remove(f)

        self.device = device
        self.dataloader_workers = dataloader_workers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if not device:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
                torch.set_num_threads(cpu_threads_count)

        # features used in training
        self.features_order = []
        self.features_dim = {}
        logging.debug("Loading features from {}".format(self.model_path + '/features.txt'))
        with open(self.model_path + '/features.txt', 'r') as file:
            for f in file:
                name = f.split(' ')[0]
                dim = int(f.split(' ')[1][:-1])
                self.features_order.append(name)
                self.features_dim[name] = dim

        # load pretrained model
        self.model_inner = SimpleClassifier(self.input_dim, self.output_dim, features_order=self.features_order,
                                            features_dim=self.features_dim)
        self.model_inner_ka = SimpleClassifier(self.input_dim, self.output_dim_kaggle,
                                               features_order=self.features_order,
                                               features_dim=self.features_dim)

        if use_my_model:
            print('my_model')
            model_tensor = torch.load("./model_result/model_maybe_best.pth", map_location=torch.device(self.device))
            self.model_inner.load_state_dict(model_tensor, strict=False)
        else:
            model_tensor = torch.load(self.model_path + "/model.pt", map_location=torch.device(self.device))
            model_tensor.pop('fc.weight')
            model_tensor.pop('fc.bias')
            # self.model_inner.load_state_dict(model_tensor, strict=False)
        self.te = TextualExtractor(self.device, use_my=False)

        self.model = Classifier(self.model_inner, self.input_dim, self.output_dim, features_order=self.features_order,
                                features_dim=self.features_dim)
        self.model_ka = Classifier(self.model_inner_ka, self.input_dim, self.output_dim_kaggle,
                                   features_order=self.features_order,
                                   features_dim=self.features_dim)

        self.model_dev = DevClassifier(self.model_inner, self.input_dim, self.output_dim,
                                       features_order=self.features_order,
                                       features_dim=self.features_dim, try_type=try_type)
        self.model_dev_ka = DevClassifier(self.model_inner, self.input_dim, self.output_dim_kaggle,
                                          features_order=self.features_order,
                                          features_dim=self.features_dim, try_type=try_type)
        self.model_dev_bi = DevClassifier(self.model_inner, self.input_dim, 2,
                                          features_order=self.features_order,
                                          features_dim=self.features_dim, try_type=try_type)

    def get_scores(self, x):
        with torch.no_grad():
            self.model.eval()
            return self.model.forward(x)

    def fetch_website(self, url):
        logging.debug("Fetching website: {}".format(url))
        response = access_website(url)
        w = Webpage(url)
        if response is not None:
            html, get_code, content_type = response
            w.http_code = get_code
            if self.is_valid(get_code, content_type):
                w.is_valid = True
                w.html = html
        if self.with_visual:
            logging.debug("Generating screenshot: {}".format(url))
            out_path = self.temporary_dir + "/screenshots/" + str(w.uid)
            w.screenshot_path = take_screenshot(w.url, out_path)
            logging.debug("Screenshot for {} ready in {}".format(url, w.screenshot_path))
        return w

    def get_features(self, url, html, screenshot_path, text=None):
        # te = TextualExtractor(self.device, use_my=True)
        features = self.te.get_features(url, html, text=text)
        if self.with_visual:
            ve = VisualExtractor(self.device)
            visual_features = ve.get_features(screenshot_path)
            features['f_visual'] = visual_features
        return features

    def predict(self, website):
        website.features = self.get_features(website.url, website.html, website.screenshot_path)
        all_features = self.concatenate_features(website)
        input_features = torch.FloatTensor(all_features)
        scores, embeddings = self.get_scores(input_features)
        return dict(zip(self.classes, torch.sigmoid(scores).tolist())), embeddings.tolist()

    def train_get_score(self, website, text=None):
        website.features = self.get_features(website.url, website.html, website.screenshot_path, text=text)
        all_features = self.concatenate_features(website)
        input_features = torch.FloatTensor(all_features)
        return input_features

    def concatenate_features(self, w, use_noise=False):
        """
        Concatenate the features attributes of webpage instance, with respect to the features order in h2v
        """

        v = np.zeros(self.input_dim)

        ix = 0

        for f_name in self.features_order:
            f_dim = self.features_dim[f_name]
            f_value = w.features[f_name]
            if f_value is None:

                # f_value = np.random.normal(0, 1, f_dim)
                f_value = f_dim * [0.]  # if no feature, replace with zeros
                if use_noise is True:
                    for i in range(len(f_value)):
                        f_value[i] = random.gauss(0, 1)
            v[ix:ix + f_dim] = f_value
            ix += f_dim

        return v

    def is_valid(self, get_code, content_type):
        valid_get_code = get_code == 200
        valid_content_type = content_type.startswith('text/html')
        return valid_get_code and valid_content_type


class ImportanceLearning(nn.Module):
    def __init__(self, feature_type_num, features_order, features_dim):
        super(ImportanceLearning, self).__init__()
        self.ftn = feature_type_num
        self.fc = nn.Linear(feature_type_num, feature_type_num)
        self.sigmoid = nn.Sigmoid()
        self.dim_num = []
        for name in features_order:
            self.dim_num.append(features_dim[name])

    def forward(self, features_vector):
        res = features_vector
        fe_list = torch.split(features_vector, self.dim_num, dim=-1)
        max_pool = [torch.max(fe, dim=-1)[0] for fe in fe_list]
        max_tensor = torch.stack(max_pool, dim=-1)
        x = self.fc(max_tensor)
        scale = self.sigmoid(x)

        scale_split = torch.split(scale, [1] * self.ftn, dim=-1)
        scaled_features = [fe_list[i] * scale_split[i] for i in range(len(fe_list))]
        scaled_features_t = torch.cat(scaled_features, dim=-1)

        return res + scaled_features_t


class Classifier(nn.Module):
    def __init__(self, simple_classifier, input_dim, output_dim, features_order, features_dim):
        super(Classifier, self).__init__()
        self.CLS = simple_classifier
        self.important_layer = ImportanceLearning(len(features_order), features_order, features_dim)
        self.att_layer = AttentionLayer(input_dim)

    def forward(self, x):
        x = self.important_layer(x)
        x = self.att_layer(x)

        out, emb = self.CLS(x)
        return out, emb


class SimpleClassifier(nn.Module):
    """
    Model architecture of Homepage2vec
    """

    def __init__(self, input_dim, output_dim, features_order, features_dim, dropout=0.4):
        super(SimpleClassifier, self).__init__()

        self.layer1 = torch.nn.Linear(input_dim, 1000)
        self.layer2 = torch.nn.Linear(1000, 100)
        self.fc = torch.nn.Linear(100, output_dim)

        self.drop = torch.nn.Dropout(dropout)  # dropout of 0.5 before each layer

        self.dim_num = []
        for name in features_order:
            self.dim_num.append(features_dim[name])

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(self.drop(x))

        emb = self.layer2(x)
        x = F.relu(self.drop(emb))

        x = self.fc(x)

        return x, emb


from new_model import FA, FEA, FA2
from resnet import resnet18, resnet10, resnet152, resnet101, resnet50, resnet34


class DevClassifier(nn.Module):
    """
    Model architecture of Homepage2vec
    """

    def __init__(self, mlp, input_dim, output_dim, features_order, features_dim, try_type=1, dropout=0.4):
        super(DevClassifier, self).__init__()
        self.try_type = try_type

        # self.layer1 = torch.nn.Linear(5476, 1000)
        self.layer1 = torch.nn.Linear(2361, 1000)
        self.layer2 = torch.nn.Linear(1000, 100)
        self.fc = torch.nn.Linear(100, output_dim)

        self.drop = torch.nn.Dropout(dropout)  # dropout of 0.5 before each layer

        self.fc_1 = torch.nn.Linear(768 * 4, 768)

        if try_type == 1:
            self.fa = FA()
        elif try_type == 2 or try_type == 4 or try_type == 5:
            self.fa = FA(2)

        self.fea = FEA()
        # self.fa2 = FA2()

        # self.flat = nn.Flatten()
        # self.changeD = torch.nn.Linear(3*265*265, input_dim)
        # self.mlp = mlp

        self.important_layer = ImportanceLearning(len(features_order), features_order, features_dim)
        self.att_layer = AttentionLayer(input_dim)

        self.resnet = resnet50(pretrained=True, num_classes=output_dim)
        self.dim_num = []
        for name in features_order:
            self.dim_num.append(features_dim[name])

    def forward(self, x):
        fe_list = torch.split(x, self.dim_num, dim=-1)
        fe_html_xlm = torch.cat(fe_list[3: 7], dim=1, out=None)
        fe_meta = fe_list[2]

        fe_html_t = self.fc_1(fe_html_xlm)  # B
        if self.try_type == 4:
            fe_html_t = F.relu(self.drop(fe_html_t))

        fe_html = torch.cat([fe_meta, fe_html_t], dim=1, out=None)
        fe_url = torch.cat(fe_list[0: 2], dim=1, out=None)  # A
        fe_raw_text = torch.cat(fe_list[7:], dim=1, out=None)

        if self.try_type == 3:
            x = torch.cat([fe_url, fe_html, fe_raw_text], dim=1)
            x = self.layer1(x)
            x = F.relu(self.drop(x))
            emb = self.layer2(x)
            x = F.relu(self.drop(emb))
            out = self.fc(x)
        else:
            x = self.fa(A=fe_url, B=fe_html, C=fe_raw_text)
            if self.try_type != 5:
                x = self.fea(x)

            # x = self.fa2(x)
            # x = self.flat(x)
            # x = self.changeD(x)
            # out, emb = self.mlp(x)

            out = self.resnet(x)

        # return x, emb
        return out


import math


class AttentionLayer(nn.Module):
    def __init__(self, length):
        super(AttentionLayer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(length, length, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(length, length, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(length, length, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x
        # print(x.size())
        d_k = x.size(-1)
        # print(d_k)
        if d_k != x.size(0):
            x = x.mean([0])
        q = self.fc1(x)
        q = q.resize(1, d_k)
        k = self.fc2(x)
        k = k.resize(1, d_k)
        v = self.fc3(x)
        v = v.resize(1, d_k)

        scores = torch.matmul(q.T, k) / (d_k ** 0.5)
        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(v, scores)

        return output + residual


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, Y, X):
        return self.ln(self.dropout(Y) + X)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid=None, dropout=0.):
        super(PositionwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0.):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head

        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_v = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q, v):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        if len(v.shape) == 2:  # k_len missing
            v = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # batch_size-
        k_len = k.shape[1]
        q_len = q.shape[1]
        v_len = v.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        vx = self.w_v(v).view(mb_size, v_len, self.n_head, self.hidden_dim)
        vx = vx.permute(2, 0, 1, 3).contiguous().view(-1, v_len, self.hidden_dim)

        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')

        # score = score.view(-1, k_len * q_len).contiguous()
        score = F.softmax(score, dim=-1)
        # score = score.view(-1, k_len, q_len)
        output = torch.bmm(score, vx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim,
                 hidden_dim=None,
                 out_dim=None, n_head=1,
                 score_function='dot_product',
                 dropout=0.):
        super(TransformerEncoder, self).__init__()
        self.attention = Attention(embed_dim, hidden_dim=None, out_dim=out_dim, n_head=n_head,
                                   score_function=score_function, dropout=dropout)
        self.ffn = PositionwiseFeedForward(out_dim, hidden_dim, dropout)
        self.add_norm1 = AddNorm([out_dim], dropout)
        self.add_norm2 = AddNorm([out_dim], dropout)

    def forward(self, input):
        # 改成kqv也行，适应不同叠加的结构
        x = input
        output = self.attention(x, x, x)
        output = self.add_norm1(output, x)
        Y = self.ffn(output)
        Y = self.add_norm2(Y, output)
        return Y


class TransformerClassifier(nn.Module):
    def __init__(self, output_dim=13, dropout=0.3, vocab_size=100, num_layers=3):
        super(TransformerClassifier, self).__init__()

        # if mode == 'sentence-transformer':
        #     self.embedding = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
        # elif mode == 'bert':
        #     self.bert = BertModel.from_pretrained("bert-base-cased")
        self.encoder = nn.Sequential()
        for i in range(num_layers):
            self.encoder.add_module("block" + str(i), TransformerEncoder(
                embed_dim=50,
                hidden_dim=50,
                out_dim=50,
                dropout=dropout,
                score_function='scaled_dot_product',
                n_head=1
            ))

        self.embed = Embeddings(d_model=50, vocab=vocab_size)

        self.layer1 = torch.nn.Linear(50, 50)
        self.fc = torch.nn.Linear(50, output_dim)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)

        x = torch.max(x, dim=1)[0]
        # print(x.shape)
        x = self.layer1(x)
        x = F.relu(self.drop(x))
        x = self.fc(x)
        return x


class Webpage:
    """
    Shell for a webpage query
    """

    def __init__(self, url):
        self.url = url
        self.uid = uuid.uuid4().hex
        self.is_valid = False
        self.http_code = False
        self.html = None
        self.screenshot_path = None
        self.features = None
        self.embedding = None
        self.scores = None

    def __repr__(self):
        return json.dumps(self.__dict__)
