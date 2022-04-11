import jieba
import numpy as np
import pandas as pd

from anal import *
import jieba as jb
import tqdm
import re
from dataload.virtual_loader import *
from ast import literal_eval
import torch as tc
import tqdm
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertLMHeadModel, BertConfig,AutoTokenizer, AutoModelForMaskedLM
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
from util.util import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import *
from sklearn.preprocessing import StandardScaler


device = tc.device('cuda:0' if tc.cuda.is_available() else 'cpu')


class WordSepAnal(DoAnal):
    '''
        Things be done by the current module:
            1. common stopwords detection.
            2. picture ref detection & extraction
            3. fill null in chat_txt
            4. add stop words.
    '''
    def __init__(self,word_dict='../data/word_dict.log',stop_words='../data/stop_words.log',**kwargs):
        super(WordSepAnal, self).__init__(**kwargs)
        self.word_dict = word_dict
        self.stop_words = []
        with open(stop_words,'r',encoding='utf-8') as f:
            self.stop_words.extend(f.readlines())
        self.stop_words = [line.strip() for line in self.stop_words]

    def __str__(self):
        return 'WordSepAnal:'+self.anal_name

    def fit_transform(self, dloader: dataloader.DataLoader):
        dloader.load()
        jieba.load_userdict(self.word_dict)
        def __detect_pic__(x):
            x = str(x)
            cnt = 0
            if x is not None and x != 'nan':
                res = re.findall(pattern='双击查看原图',string=x)
                if res is not None:
                    cnt += len(res)
            else:
                cnt = 1 # empty chat txt indicates a picture.
            return cnt

        def __trans_chat__(x):
            x = str(x)
            if x is None or x == 'nan':
                return []
            match_pic = re.finditer(pattern='双击查看原图', string=x)
            xx = ''
            pnt = 0
            for e_mat in match_pic:
                span_mat = e_mat.span()
                xx += x[pnt:span_mat[0]]
                pnt = span_mat[1]
            xx += x[pnt:]
            return xx

        def __word_split__(x):
            x = str(x)
            ret_x = []
            for word in jieba.cut_for_search(x, HMM=True):
                if word in self.stop_words:
                    continue
                else:
                    ret_x.append(word)
            return ret_x

        data = dloader.data.copy(deep=True)
        data['pic_used'] = data['chat_txt'].transform(__detect_pic__)
        data['chat_txt'] = data['chat_txt'].transform(__trans_chat__)
        data['chat_words'] = data['chat_txt'].transform(__word_split__)

        wl = WordLoader(loader_name=self.anal_name)
        wl.register(data_name='data',data=data,data_type='pd')
        wl.push()
        return wl


class WordLabelingAnal(DoAnal):
    '''
        focus on aspects of word feature instead of users or chat text.
        Things be done by the current module:
            1. label each word within their sentences.
            (2. find the clusters of vectors.)
    '''
    def __init__(self,word_ratio=0.6,**kwargs):
        super(WordLabelingAnal, self).__init__(**kwargs)
        self.word_ratio = word_ratio

    def __str__(self):
        return 'WordLabelingAnal:'+self.anal_name

    def fit_transform(self,dloader: WordLoader):
        dloader.load()
        words = dloader.data['data']['chat_words']
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
        word_lst = [k for k,v in sorted(word_dic.items(), key=lambda x: x[1], reverse=True)]
        sentences = [ele for ele in list(dloader.data['data']['chat_txt']) if ele is not None and type(ele) is str]
        word_lst = word_lst[:max(1,int(len(word_lst)*self.word_ratio))]
        # word_lst = word_lst[:10]
        fore_labels = self.gen_fore_labels(sentences=sentences, word_list=word_lst)
        fsl = ForeSpanLoader(loader_name=self.anal_name)
        fsl.register(data_name='fore_labels',data=fore_labels,data_type='inner')
        fsl.register(data_name='sentences',data=sentences,data_type='inner')
        fsl.push()

    def gen_fore_labels(self,sentences, word_list):
        fore_labels = []
        for sen in tqdm.tqdm(sentences):
            fore_label = []
            for word in word_list:
                match_word = re.finditer(pattern=word, string=sen)
                for e_mat in match_word:
                    fore_label.append(e_mat.span())
            fore_labels.append(fore_label)
        return fore_labels


class CharVecAnal(DoAnal):
    '''
        focus on aspects of word feature instead of users or chat text.
        Things be done by the current module:
            1. interpret characters into vectors.
    '''

    def __init__(self, epochs=1,lr=1e-4,batch_sz=4,fore_mask_sz=6, **kwargs):
        super(CharVecAnal, self).__init__(**kwargs)
        self.epochs = epochs
        self.lr = lr
        self.fore_mark_sz = fore_mask_sz
        self.batch_sz=batch_sz

    def __str__(self):
        return 'CharVecAnal:' + self.anal_name

    def fit_transform(self, dloader: ForeSpanLoader):
        dloader.load()
        sentences = dloader.data['sentences']
        fore_labels = dloader.data['fore_labels']

        trainset = WordVecDataset(sentences=sentences, fore_labels=fore_labels, fore_mask_sz=self.fore_mark_sz)
        train_loader = DataLoader(dataset=trainset, batch_size=self.batch_sz, shuffle=True, collate_fn=lambda x: x)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

        loss = tc.nn.CrossEntropyLoss()
        optim = tc.optim.Adam(model.parameters(), lr=self.lr)
        train_log = []
        model.to(device)
        for epoch in range(self.epochs):
            model.train()
            st_time = time.time()
            train_loss = 0.
            cnt = 0
            for idx, batch_data in enumerate(train_loader):
                optim.zero_grad()
                org_sen, fore_mask_sen, fore_mask_pos, part_mask_sen, part_mask_pos = [], [], [], [], []
                comp_org_ids_fore = []
                for idy, (e_org_sen, e_fore_mask_sen, e_fore_mask_pos, e_part_mask_sen, e_part_mask_pos) in enumerate(
                        batch_data):
                    org_sen.append(e_org_sen)
                    if len(e_fore_mask_sen) > 0:
                        fore_mask_sen.extend(e_fore_mask_sen)
                        fore_mask_pos.extend(e_fore_mask_pos)
                        comp_org_ids_fore.extend([idy]*len(e_fore_mask_sen))
                    if len(e_part_mask_sen) > 0:
                        part_mask_sen.extend(e_part_mask_sen)
                        part_mask_pos.extend(e_part_mask_pos)

                try:
                    org_tok = tokenizer(org_sen, padding='max_length', return_tensors="pt",truncation=True, max_length=128)
                except IndexError:
                    optim.zero_grad()
                org_tok = org_tok.to(device)
                out_org = model(**org_tok)
                org_batch_loss = loss(out_org.logits.view(-1, 21128), org_tok['input_ids'].view(-1))

                try:
                    fore_tok = tokenizer(fore_mask_sen, padding='max_length', return_tensors="pt",truncation=True, max_length=128)
                except IndexError:
                    optim.zero_grad()
                    continue
                fore_tok = fore_tok.to(device)
                out_fore = model(**fore_tok)
                fore_batch_loss = loss(out_fore.logits.view(-1, 21128),
                                       org_tok['input_ids'][comp_org_ids_fore].view(-1))

                batch_loss = org_batch_loss + fore_batch_loss
                batch_loss.backward()
                optim.step()
                train_loss += batch_loss.item()
                cnt += len(org_tok['input_ids'].view(-1))
                print('--epoch {} | iter {}/{} | train loss:{:.6f} | time:{:.3f}'.format(epoch, idx,len(sentences)/self.batch_sz,
                                                                                    train_loss / cnt,
                                                                                    time.time() - st_time))
                train_log.append(float(train_loss / cnt))

            print('--epoch {} | train loss:{:.6f} | time:{:.3f}'.format(epoch, train_loss / cnt, time.time() - st_time))
            # tc.cuda.empty_cache()
        tc.save(model,self.pwd(is_tmp=True)+'.encoder')
        enc_pth = self.pwd(is_tmp=True)+'.encoder'
        eml = EmbModelLoader(loader_name=self.anal_name)
        eml.register('enc_pth',enc_pth,'inner')
        eml.register('train_log',train_log,'inner')
        eml.push()

        # fore_tok = tokenizer(fore_mask_sen, padding=True, return_tensors="pt")
        # fore_tok = fore_tok.to(device)
        # out_fore = model(**fore_tok)
        # fore_mask_pos_sq = []
        # for mask_pos in fore_mask_pos:
        #     fore_mask_pos_sq.append(mask_pos)
        # out_fore = out_fore[:,:]
        # fore_batch_loss = loss(out_org.logits.view(batch_sz * 22, -1), org_tok['input_ids'].view(-1))

class WordVecAnal(DoAnal):
    '''
        focus on aspects of word feature instead of users or chat text.
        Things be done by the current module:
            1. interpret character vec into word vec.
    '''

    def __init__(self,word_loader, **kwargs):
        super(WordVecAnal, self).__init__(**kwargs)
        self.word_loader = word_loader

    def __str__(self):
        return 'WordVecAnal:' + self.anal_name

    def fit_transform(self, dloader: EmbModelLoader):
        dloader.load()
        enc_path = dloader.data['enc_pth']
        model = tc.load(enc_path,map_location=tc.device('cpu'))

        self.word_loader.load()
        words = self.word_loader.data['data']['chat_words']
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
        word_lst = [k for k, v in sorted(word_dic.items(), key=lambda x: x[1], reverse=True)]

        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        word_tok = tokenizer(word_lst,padding='max_length', return_tensors="pt",truncation=True, max_length=128)
        word_emb = model.base_model.embeddings(word_tok['input_ids'])

        emb_lst = []
        emb_lst_pnt = 0
        word2id = {}
        id2word = {}
        for emb,input_id,word in zip(word_emb,word_tok['input_ids'],word_lst):
            st_id = 1
            ed_id = 1
            for i in range(st_id,128):
                if int(input_id[i]) == 102:
                    ed_id = i
                    break
            emb_lst.append(tc.mean(emb[st_id:ed_id],dim=0).tolist())
            word2id[word] = emb_lst_pnt
            id2word[emb_lst_pnt] = word
            emb_lst_pnt += 1
        embs = np.array(emb_lst)
        wel = WordEmbLoader(loader_name=self.anal_name)
        wel.register('embs',embs,'np')
        wel.register('word2id',word2id)
        wel.register('id2word',id2word)
        wel.push()

class VecDistAnal(Anal):
    '''
        focus on aspects of word feature instead of users or chat text.
        Things be done by the current module:
            1. determine the sufficiently-important cluster of vectors.
    '''

    def __init__(self,dloader: WordEmbLoader,num_cls=50,samples_per_cls=25, **kwargs):
        super(VecDistAnal, self).__init__(**kwargs)
        self.dloader = dloader
        self.num_cls = num_cls
        self.samples_per_cls = samples_per_cls

    def __str__(self):
        return 'VecDistAnal:' + self.anal_name

    def _print(self, to_screen=True):
        self.dloader.load()
        word2id = self.dloader.data['word2id']
        id2word = self.dloader.data['id2word']
        embs = self.dloader.data['embs']
        for emb in embs:
            emb[np.isnan(emb)] = 0
        print('embs is nan:{}'.format(np.isnan(embs).any()))

        pca = PCA(n_components=64)

        embs_low = pca.fit_transform(embs)

        print('PCA transform completed.')

        ss = StandardScaler()
        embs_low = ss.fit_transform(embs_low)

        print('Scaling transform completed.')

        # bw = estimate_bandwidth(embs_low, n_samples=len(embs_low), quantile=0.1)
        # model = MeanShift(bandwidth=bw, bin_seeding=True)
        # model.fit(embs_low)
        # pred_y = model.predict(embs_low)
        # centers = model.cluster_centers_
        # sc = SpectralClustering(n_clusters=self.num_cls)
        # pred_y = sc.fit_predict(embs_low)

        km = KMeans(n_clusters=self.num_cls)
        pred_y = km.fit_predict(embs_low)
        print('clustering completed.')

        tsne = TSNE(n_components=2,learning_rate=200)
        embs_2d = tsne.fit_transform(embs_low)
        print('t-sne transform completed.')
        plt.scatter(embs_2d[:,0],embs_2d[:,1],c=pred_y,s=1.5,alpha=0.4)
        plt.savefig(self.pwd(False)+'.org_embs.svg')
        plt.show()

        eig_word = []
        for cls in range(self.num_cls):
            idx = np.array(list(range(embs_low.shape[0])))
            idx_lst = idx[pred_y == cls].tolist()
            random.shuffle(idx_lst)
            idx_lst = idx_lst[:self.samples_per_cls]
            eig_word.append([id2word[str(ele)] for ele in idx_lst])

        with open(self.pwd(True)+'.eig_word.txt','w',encoding='utf-8') as f:
            for e_eig_word in eig_word:
                cur_line = ','.join(e_eig_word)
                f.write(cur_line+'\n')

        vdl = VecDistLoader(loader_name='vec-dist')
        vdl.register('embs',embs_low,'np')
        vdl.register('labels',pred_y,'np')
        vdl.register('word2id',word2id)
        vdl.register('id2word', id2word)
        vdl.register('eig_word',eig_word)
        vdl.push()


class VecLabelingAnal(Anal):
    '''
        focus on aspects of word feature instead of users or chat text.
        Things be done by the current module:
            1. create continuous labels for each vectorized feature.
    '''

    def __init__(self, dloader: WordEmbLoader, **kwargs):
        super(VecLabelingAnal, self).__init__(**kwargs)
        self.dloader = dloader

    def __str__(self):
        return 'VecDistAnal:' + self.anal_name

    def _fit(self, dloader: VecDistLoader):
        dloader.load()
        embs = dloader.data['embs']
        labels = dloader.data['labels']
        word2id = dloader.data['word2id']
        id2word = dloader.data['id2word']
        eig_word = dloader.data['eig_word']

        self.dloader.load()
        eig_word2person = self.dloader.data['word2person']

        emb_lst = []
        person_lst = []
        for word in eig_word2person:
            emb_lst.append(embs[word2id[word]].tolist())
            person_lst.append(eig_word2person[word])

        emb_lst = np.array(emb_lst)
        person_lst = np.array(emb_lst)

        assert emb_lst.shape[0] == person_lst.shape[0]

        persons = np.zeros(shape=(embs.shape[0],person_lst.shape[1]))
        for word in word2id:
            if word not in eig_word2person:
                persons[word2id[word]] = np.array((np.sum(np.exp( - (embs[word2id[word]] - emb_lst) ** 2 / 2),axis = 1) @ person_lst).tolist())
            else:
                persons[word2id[word]] = np.array(eig_word2person[word])

        embs = dloader.data['embs']
        labels = dloader.data['labels']
        word2id = dloader.data['word2id']
        id2word = dloader.data['id2word']
        eig_word = dloader.data['eig_word']

        pel = PersonEmbLoader(loader_name=self.anal_name)
        pel.register('embs',embs,'np')
        pel.register('labels',labels,'np')
        pel.register('word2id',word2id)
        pel.register('id2word', id2word)
        pel.register('eig_word',eig_word)
        pel.register('eig_word2per',eig_word2person)
        pel.register('persons',persons,'np')
        pel.push()

if __name__ == '__main__':
    # wsa = WordSepAnal(anal_name='word-sep')
    # cdl1 = dataloader.ChatDataLoader(loader_name='cdl1', local_file='chat-data.log')
    # wsa.fit_transform(cdl1)

    # wla = WordLabelingAnal(anal_name='word-lab',word_ratio=0.8)
    # wla.fit_transform(WordLoader(loader_name='word-sep'))

    # cva = CharVecAnal(anal_name='char-vec')
    # cva.fit_transform(ForeSpanLoader(loader_name='word-lab'))

    # wva = WordVecAnal(anal_name='word-vec',word_loader=WordLoader(loader_name='word-sep'))
    # wva.fit_transform(EmbModelLoader(loader_name='char-vec'))

    vda = VecDistAnal(anal_name='word-vec',dloader=WordEmbLoader(loader_name='word-vec'))
    vda.print()


    print('hello anal words.')
