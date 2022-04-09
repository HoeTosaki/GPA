import json
import os
import time
import numpy as np
import pandas as pd
import re
import copy

class DataLoader:
    '''
    load data within a regular format as pandas.DataFrame, including:
        data_mat: [n × m], n - samples, m - feat dim
        feat_name: [m], a list of corresponding feature names
    '''
    def __init__(self,loader_name='anony',data_dir='../data',tmp_dir='../data/.tmp',**kwargs):
        self.loader_name = loader_name
        self.data_dir = data_dir
        self.tmp_dir = tmp_dir

        self.data = None
        self.feat_name = []

    def fetch(self,force=False):
        if not force and self._fetch_checked():
            print('[{}] fetch file checked.'.format(self))
            return
        st_time = time.time()
        self._fetch()
        print('--{}-- file fetched by {:.3f}s.'.format(self,time.time()-st_time))

    def parse(self,force=False):
        if not force and self._parse_checked():
            print('[{}] parse file checked.'.format(self))
            return
        st_time = time.time()
        self._parse()
        print('--{}-- file parsed by {:.3f}s.'.format(self,time.time()-st_time))

    def load(self,force=False):
        if force:
            self.fetch(force=force)
            self.parse(force=force)
        else:
            if not self._fetch_checked():
                self.fetch(force=force)
            if not self._parse_checked():
                self.parse(force=force)
        st_time = time.time()
        self._load()
        print('--{}-- file loaded by {:.3f}s.'.format(self, time.time() - st_time))

    def _fetch_checked(self):
        return False

    def _parse_checked(self):
        return False

    def _load_checked(self):
        return self.data_mat is None or len(self.feat_name) <= 0

    def _fetch(self):
        raise NotImplementedError

    def _parse(self):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    def pwd(self):
        '''
        return the reserved path for data loading and parsing.
        '''
        return os.path.join(self.tmp_dir,self.loader_name)

    def __str__(self):
        return 'DataLoader:' + self.loader_name


class ChatDataLoader(DataLoader):
    def __init__(self,local_file,**kwargs):
        '''
            Chat data requires direct loading from local files.
        '''
        super(ChatDataLoader, self).__init__(**kwargs)
        self.local_file = local_file

    def __str__(self):
        return 'ChatDataLoader:'+self.loader_name

    def _load(self):
        self.data = pd.read_csv(self.pwd()+'.csv')
        self.feat_name = list(self.data.columns)

    def _fetch(self):
        print('[{}] Empty Routine Executed since chat data requires loading from local files.'.format(self))

    def _fetch_checked(self):
        return os.path.exists(os.path.join(self.data_dir,self.local_file))

    def _parse(self):
        cur_date = None
        cur_time = None
        user_name = None
        user_id = None
        chat_lv = None
        chat_txt = None
        date_pat = r'[0-9]{4}-[0-9]{2}-[0-9]{2}'
        time_pat = r'[0-9]{2}:[0-9]{2}:[0-9]{2}'
        chat_lv_pat = r'【.*】'
        user_id_pat = r'\([0-9]{4}[0-9]+\)|<.*\.com.*>' # id at least more than 5.
        at_pat = r'@[^@ ]+ '
        cur_informs = []
        is_mul_line = False # avoid add duplicate chat line to one user.

        data_lst = []
        name2id = {}
        feat_names = ['user_id','user_name','chat_lv','date','time','chat_txt','informs']
        is_date_parsed = False
        is_withdrawed = False
        with open(os.path.join(self.data_dir,self.local_file),'r',encoding='utf-8') as f:
            for idx,line in enumerate(f.readlines()):
                if line is not None:
                    org_line = line
                    is_date_parsed = False
                    if is_withdrawed:
                        is_withdrawed = False
                        print('[{}] line {} withdraw occurred end at {}'.format(self, idx, org_line))
                        continue
                    line = line.strip()
                    match_date = re.finditer(pattern=date_pat,string=line)
                    match_time = re.finditer(pattern=time_pat,string=line)
                    # assert match_date is None or match_time is None,print('[{}] parse line {} with error, line text:{}'.format(self,idx,line))
                    match_date = next(match_date,None)
                    if match_date is not None:
                        span = match_date.span()
                        assert span[0] == 0 and span[1] == len(line),print('[{}] parse line {} with error, line text:{}'.format(self,idx,line))
                        cur_date = line[span[0]:span[1]]
                        is_date_parsed = True
                    match_time = next(match_time,None)
                    if match_time is not None:
                        is_mul_line = False
                        span = match_time.span()
                        if span[0] == 0:
                            is_withdrawed = True
                            print('[{}] line {} withdraw occurred start at {}'.format(self,idx,org_line))
                            continue # bad line since msg withdraw occurred.
                        cur_time = line[span[0]:span[1]]
                        line = line[:span[0]]
                        match_chat_lv = re.match(pattern=chat_lv_pat,string=line)
                        if match_chat_lv is not None:
                            span_chat_lv = match_chat_lv.span()
                            chat_lv = line[span_chat_lv[0] + 1:span_chat_lv[1]-1]
                            line = line[span_chat_lv[1]:]
                        else:
                            chat_lv = 'unk'
                        match_id = re.finditer(pattern=user_id_pat,string=line)
                        span_id = next(match_id, None)
                        assert span_id is not None,print('[{}] parse line {} with error, line text:{}'.format(self,idx,line))
                        span_id = span_id.span()
                        user_id = line[span_id[0]+1:span_id[1]-1]
                        line = line[:span_id[0]]
                        user_name = line if line is not None else ''
                        if user_name != '':
                            if user_name in name2id:
                                # assert user_id == name2id[user_name],print('[{}] parse line {} with error, line text:{}'.format(self,idx,line))
                                pass
                            name2id[user_name] = user_id
                    if match_date is None and match_time is None:
                        chat_txt = copy.deepcopy(line)
                        if chat_txt == '':
                            if is_mul_line:
                                continue # ignore cur line.
                            else:
                                is_mul_line = True
                        else:
                            is_mul_line = False
                        match_at = re.finditer(pattern=at_pat,string=line)
                        res_line = ''
                        pnt = 0
                        if match_at is not None:
                            cur_informs = []
                            for e_mat in match_at:
                                span_e_mat = e_mat.span()
                                cur_inform = line[span_e_mat[0]+1:span_e_mat[1]-1]
                                for name in name2id:
                                    # if re.match(name,cur_inform) is not None:
                                    if cur_inform[:len(name)] == name:
                                        cur_informs.append(name2id[name])
                                        res_line += line[pnt:span_e_mat[0]]
                                        pnt = span_e_mat[1]
                            res_line += line[pnt:]
                            chat_txt = res_line
                        cur_informs = list(set(cur_informs))
                        data_lst.append([user_id,user_name,chat_lv,cur_date,cur_time,chat_txt,'-'.join([str(ele) for ele in cur_informs])])
                        # add new chat lint to our log.
        pd_data = pd.DataFrame(data=data_lst,columns=feat_names)
        pd_data.to_csv(self.pwd()+'.csv',index=False)

    def _parse_checked(self):
        return os.path.exists(self.pwd()+'.csv')


if __name__ == '__main__':
    print('hello dataloader.')
    cdl1 = ChatDataLoader(loader_name='cdl1',local_file='chat-data.log')
    cdl1.load(force=True)
    print(cdl1.data)


