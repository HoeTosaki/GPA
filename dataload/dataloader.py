import json
import os
import time

class DataLoader:
    '''
    load data within a regular format as pandas.DataFrame, including:
        data_mat: [n × m], n - samples, m - feat dim
        feat_name: [m], a list of corresponding feature names
    '''
    def __init__(self,loader_name='anony',data_dir='./data',**kwargs):
        self.loader_name = loader_name
        self.data_dir = data_dir

        self.data_mat = None
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
        return os.path.join(self.data_dir,self.loader_name)

    def __str__(self):
        return 'DataLoader:' + self.loader_name


class ChatDataLoader(DataLoader):
    