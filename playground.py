# from transformers import BertTokenizer, BertLMHeadModel, BertConfig
import random

import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch as tc
from torch.utils.data import Dataset, DataLoader
import re
from ast import literal_eval
import time

device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')

def gen_fore_labels(sentences,word_list):
    fore_labels = []
    for sen in sentences:
        fore_label = []
        for word in word_list:
            match_word = re.finditer(pattern=word,string=sen)
            for e_mat in match_word:
                fore_label.append(e_mat.span())
        fore_labels.append(fore_label)

    return fore_labels

class WordVecDataset(Dataset):
    def __init__(self,sentences,fore_labels,fore_mask_sz=5):
        self.sentences = sentences
        self.fore_labels = fore_labels
        self.fore_mask_sz = fore_mask_sz
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        org_sen = self.sentences[item]
        fore_mask_pos = []
        fore_mask_sen = []
        len_of_fores = len(self.fore_labels[item])
        random.shuffle(self.fore_labels[item])
        for fore_st,fore_ed in self.fore_labels[item][:min(len_of_fores,self.fore_mask_sz)]:
            sen = list(org_sen)
            span = fore_ed - fore_st
            sen[fore_st:fore_ed] = ['[MASK]'] * span
            fore_mask_sen.append(''.join(sen))
            fore_mask_pos.append(list(range(fore_st,fore_ed)))
        part_mask_pos = []
        part_mask_sen = []
        len_of_fores = len(self.fore_labels[item])
        random.shuffle(self.fore_labels[item])
        for fore_st, fore_ed in self.fore_labels[item][:min(len_of_fores, self.fore_mask_sz)]:
            span = fore_ed - fore_st
            if span <= 1:
                continue
            sen = list(org_sen)
            idx_lst = list(range(fore_st,fore_ed))
            chs_idx = random.choice(idx_lst)
            sen[chs_idx] = '[MASK]'
            part_mask_sen.append(''.join(sen))
            part_mask_pos.append([chs_idx])
        # len_pad_part = len(part_mask_sen)
        # len_pad_fore = len(fore_mask_sen)
        # if len_pad_part < self.fore_mask_sz:
        #     sub = self.fore_mask_sz - len_pad_part
        #     part_mask_pos.extend([(0,0)]*sub)
        #     part_mask_sen.extend(['']*sub)
        # if len_pad_fore < self.fore_mask_sz:
        #     sub = self.fore_mask_sz - len_pad_fore
        #     fore_mask_pos.extend([(0,0)] * sub)
        #     fore_mask_sen.extend([''] * sub)
        return org_sen,fore_mask_sen,fore_mask_pos,part_mask_sen,part_mask_pos

if  __name__ == '__main__':

    epochs = 4
    batch_sz = 16

    data = pd.read_csv('./data/.tmp/word-sep.data.csv')
    sentences = [ele for ele in list(data['chat_txt']) if ele is not None and type(ele) is str]

    words = data['chat_words']
    word_lst = []
    for val in list(words.values):
        lst = literal_eval(val)
        for ele in lst:
            ele = ele.strip()
            if ele not in [r'\u3000', '']:
                try:
                    _ = float(ele)
                except ValueError:
                    word_lst.append(ele)
    word_dic = {}
    for word in word_lst:
        word_dic[word] = word_dic.get(word, 0) + 1
    word_list = list(word_dic.keys())
    random.shuffle(word_list)
    word_list = word_list[:1]
    fore_labels = gen_fore_labels(sentences=sentences,word_list=word_list)
    trainset = WordVecDataset(sentences=sentences,fore_labels=fore_labels,fore_mask_sz=6)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_sz, shuffle=True,collate_fn=lambda x:x)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

    loss = tc.nn.CrossEntropyLoss()
    optim = tc.optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        st_time = time.time()
        train_loss = 0.
        cnt = 0
        for idx,batch_data in enumerate(train_loader):
            optim.zero_grad()
            org_sen, fore_mask_sen, fore_mask_pos, part_mask_sen, part_mask_pos = [],[],[],[],[]
            comp_org_ids_fore = []
            for idy,(e_org_sen,e_fore_mask_sen,e_fore_mask_pos,e_part_mask_sen,e_part_mask_pos) in enumerate(batch_data):
                org_sen.append(e_org_sen)
                if len(e_fore_mask_sen) > 0:
                    fore_mask_sen.extend(e_fore_mask_sen)
                    fore_mask_pos.extend(e_fore_mask_pos)
                    comp_org_ids_fore.append(idy)
                if len(e_part_mask_sen) > 0:
                    part_mask_sen.extend(e_part_mask_sen)
                    part_mask_pos.extend(e_part_mask_pos)

            org_tok = tokenizer(org_sen, padding='max_length', return_tensors="pt",max_length=128)
            org_tok = org_tok.to(device)
            out_org = model(**org_tok)
            org_batch_loss = loss(out_org.logits.view(-1,21128),org_tok['input_ids'].view(-1))

            fore_tok = tokenizer(fore_mask_sen, padding='max_length', return_tensors="pt",max_length=128)
            fore_tok = fore_tok.to(device)
            out_fore = model(**fore_tok)
            fore_batch_loss = loss(out_fore.logits.view(-1, 21128), org_tok['input_ids'][comp_org_ids_fore].view(-1))

            batch_loss = org_batch_loss + fore_batch_loss
            batch_loss.backward()
            optim.step()
            train_loss += batch_loss.item()
            cnt += len(org_tok['input_ids'].view(-1))
            print('--epoch {} | iter {} | train loss:{:.4f} time:{:.3f}'.format(epoch,idx, batch_loss.item() / batch_sz, time.time() - st_time))
        print('--epoch {} | train loss:{:.4f} time:{:.3f}'.format(epoch,train_loss/cnt,time.time() - st_time))

            # fore_tok = tokenizer(fore_mask_sen, padding=True, return_tensors="pt")
            # fore_tok = fore_tok.to(device)
            # out_fore = model(**fore_tok)
            # fore_mask_pos_sq = []
            # for mask_pos in fore_mask_pos:
            #     fore_mask_pos_sq.append(mask_pos)
            # out_fore = out_fore[:,:]
            # fore_batch_loss = loss(out_org.logits.view(batch_sz * 22, -1), org_tok['input_ids'].view(-1))


