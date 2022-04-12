import random

import matplotlib.pyplot as plt

from anal import *
import numpy as np
import pandas as pd
import dgl
from dataload.virtual_loader import *
import torch as tc
import networkx as nx
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import tqdm
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


class UserGraphAnal(Anal):
    def __init__(self,**kwargs):
        super(UserGraphAnal, self).__init__(**kwargs)

    def fit_transform(self, dloader: UserComplLoader):
        dloader.load()
        pd_data = dloader.data['data']
        per_types = dloader.data['person_type']
        user_data = dloader.data['user_data']
        uid2nid = {val:idx for idx,val in enumerate(list(user_data['user_id'].values))}
        nid2uid = {idx:val for idx, val in enumerate(list(user_data['user_id'].values))}

        def func(x):
            return per_types[int(np.argmax(np.array([x[ele+'@nor'] for ele in per_types])))]
        user_data['person_type@nor'] = user_data.apply(func,axis=1)

        g = dgl.DGLGraph()
        g.add_nodes(len(user_data))
        per_ids = [per_types.index(str(val)) for val in user_data['person_type@nor'].values]
        g.ndata['per_ids'] = tc.LongTensor(per_ids)

        cs = ['coral','dodgerblue','brown','red','pink']

        print('start to construct graph...')

        uid_lst = list(pd_data['user_id'].values)
        interval = 3
        for idx in tqdm.tqdm(range(interval,len(uid_lst))):
            if idx < interval:
                continue
            for idy in range(idx-interval,idx):
                g.add_edge(uid2nid[uid_lst[idx]],uid2nid[uid_lst[idy]])

        G = dgl.to_networkx(g)
        plt.clf()
        nx.draw(G,node_color=[cs[int(ele)] for ele in g.ndata['per_ids']],node_size=2)
        plt.savefig(self.pwd(False)+'.user-per-graph.svg')
        plt.show()

        print('start to walk...')

        walks = []
        walk_len = 64
        for nid in tqdm.tqdm(range(g.num_nodes())):
            walk = [nid]
            cur_len = 0
            visited = set([])
            cur_nid = nid
            while cur_len < walk_len:
                next_lst = list(g.successors(cur_nid))
                random.shuffle(next_lst)
                is_hit = False
                for nnid in next_lst:
                    nnid = int(nnid)
                    if nnid not in visited:
                        visited.add(nnid)
                        walk.append(nnid)
                        cur_nid = nnid
                        cur_len += 1
                        is_hit=True
                        break
                if not is_hit:
                    break
            walk = [str(int(g.ndata['per_ids'][ele])) for ele in walk]
            walks.append(walk)

        print('start to train emb...')
        emb_sz = 16
        model = Word2Vec(sentences=walks,size=emb_sz,window=5,min_count=0,sg=1,hs=0,negative=3,workers=8)
        model.wv.save_word2vec_format(self.pwd()+'.per.emb')

        print('emb saved.')

        per_emb = [None]*len(per_types)
        with open(self.pwd()+'.per.emb','r') as f:
            for idx,line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                else:
                    lst = line.strip().split()
                    per_emb[int(lst[0])] = [float(ele) for ele in lst[1:]]
        per_emb = np.array(per_emb)
        pca = PCA(n_components=2)
        per_emb_2d = pca.fit_transform(per_emb)
        plt.clf()
        for i in range(per_emb_2d.shape[0]):
            plt.scatter([per_emb_2d[i,0]],[per_emb_2d[i,1]],c=cs[i],label=per_types[i])
        plt.legend()
        plt.title('不同人格特征交互距离')
        plt.savefig(self.pwd(True)+'.emb-per-dist.svg')
        plt.show()

if __name__ == '__main__':
    uga = UserGraphAnal(anal_name='user-graph')
    uga.fit_transform(dloader=UserComplLoader(loader_name='user-compl'))
