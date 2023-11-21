from src.homepage2vec.model import WebsiteClassifier
from sklearn import metrics
from torch import nn
import torch
from Data_prepare import *
from train_full import EarlyStopping
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
from sentence_transformers import models, losses
from transformers import AutoTokenizer, AutoModel
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1', device=device)

class_num = 13

# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_weights_for_balanced_classes_split(dataset):
    N = len(dataset)
    W = [0.] * N
    for idx in range(N):
        if dataset[idx][-1] == 10:
            W[idx] = 6.52
        elif dataset[idx][-1] == 11:
            W[idx] = 17.09
        elif dataset[idx][-1] == 2:
            W[idx] = 1.27
    Sampler = WeightedRandomSampler(W, num_samples=len(dataset))
    return Sampler


class Text_Url_dataset(Dataset):
    def __init__(self, mode='text'):
        self.data_all = prepare_train2() + prepare_train3()
        self.label_dict = collections.defaultdict(int)
        if mode == 'text':
            self.data_out = [[w[1], w[-1]] for w in self.data_all]
        elif mode == 'url':
            self.data_out = [[w[0], w[-1]] for w in self.data_all]
        else:
            raise RuntimeError('戳啦')
        for w in self.data_all:
            self.label_dict[w[-1]] += 1
        print(self.label_dict)

    def __getitem__(self, ind):
        return self.data_out[ind][0], self.data_out[ind][-1]

    def __len__(self):
        return len(self.data_all)


class Finetune_LLM(nn.Module):
    def __init__(self):
        super(Finetune_LLM, self).__init__()
        # self.emb_model = models.Transformer('paraphrase-xlm-r-multilingual-v1')
        # self.max_pooling = models.Pooling(self.model.get_sentence_embedding_dimension(),
        #                                   pooling_mode_max_tokens=True,
        #                                   pooling_mode_mean_tokens=False)
        self.linear = nn.Linear(768, 13)
        self.embedding = Model

    def forward(self, text_input):
        text_embedding = self.embedding.encode(text_input, convert_to_numpy=False, convert_to_tensor=True)
        out = self.linear(text_embedding)
        return out


if __name__ == '__main__':

    model = Finetune_LLM().to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    all_data = Text_Url_dataset('text')

    early_stopping = EarlyStopping(patience=5, verbose=True, st=True)
    train_size = int(len(all_data) * 0.8)
    test_size = len(all_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size])

    sampler_train = make_weights_for_balanced_classes_split(train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=False, sampler=sampler_train,
                                  num_workers=8)
    valid_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=8)

    train_epochs_loss = []
    valid_epochs_loss = []

    for epoch in range(20):
        model.train()
        train_epoch_loss = []
        loop = tqdm(train_dataloader)

        for step, (batch_text, batch_label) in enumerate(loop):
            batch_label = torch.eye(class_num)[batch_label, :].to(device)
            batch_output = model(batch_text)

            optimizer.zero_grad()
            loss = loss_function(batch_output, batch_label)
            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.item())
            if step % (len(train_dataloader) // 2) == 0:
                print("epoch={}/{},{}/{}of train, loss={}".format(
                    epoch, 20, step, len(train_dataloader), loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))

        model.eval()
        valid_epoch_loss = []
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        for idx, (batch_text, batch_label) in enumerate(valid_dataloader, 0):
            batch_label = batch_label.to(device)
            outputs = model(batch_text)
            n_correct += (torch.argmax(outputs, -1) == batch_label).sum().item()
            n_total += len(outputs)
            loss = loss_function(outputs, batch_label)

            outputs = torch.argmax(outputs, -1)
            valid_epoch_loss.append(loss.item())
            if t_targets_all is None:
                t_targets_all = batch_label
                t_outputs_all = outputs
            else:
                t_targets_all = torch.cat((t_targets_all, batch_label), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), labels=[0, 1, 2, 3, 4, 5,
                                                                                6, 7, 8, 9, 10,
                                                                                11, 12],
                              average='macro')
        print(f'epoch:{epoch}   acc:{acc}   f1:{f1}')
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        early_stopping(f1, model=model.embedding, path='./xlm_text')
        if early_stopping.early_stop:
            print("Early stopping")
            break
