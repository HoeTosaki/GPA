import json
import numpy as np
import pandas as pd
from dataload import dataloader


class VirtualLoader(dataloader.DataLoader):
    '''
        functionally-similar loader as Dataloader, transformed from DoAnal.
    '''
    def __init__(self,**kwargs):
        super(VirtualLoader, self).__init__(**kwargs)

        self.data_dict = {'inner':{},'np':{},'pd':{}}
        self.data = None

    def __str__(self):
        return 'VirtualLoader:'+self.loader_name

    def register(self,data_name,data,data_type='inner'):
        '''
            register sub-data for storage and loading.
        '''
        assert data_type in ['inner','np','pd']
        assert data_name not in self.data_dict[data_type]
        self.data_dict[data_type][data_name] = data

    def push(self):
        data_dict = {}
        data_dict.update(self.data_dict['inner'])
        data_dict.update({'.' + name + '.npy':name for name in self.data_dict['np']})
        data_dict.update({'.' + name + '.csv':name for name in self.data_dict['pd']})
        with open(self.pwd()+'.json','w') as f:
            json.dump(data_dict,f)
        for name in self.data_dict['np']:
            np.save(self.pwd()+'.'+name+'.npy',self.data_dict['np'][name])
        for name in self.data_dict['pd']:
            self.data_dict['pd'][name].to_csv(self.pwd()+'.'+name+'.csv')

    def _fetch(self):
        pass

    def _parse(self):
        pass

    def _load(self):
        with open(self.pwd()+'.json','r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('[{}] load failed at path:{}'.format(self,self.pwd()+'.json'))
        self.data = {}
        for name in load_dict:
            if name[:1] == '.':
                if name[-4:] == '.npy':
                    self.data[load_dict[name]] = np.load(self.pwd()+ name)
                    continue
                if name[-4:] == '.csv':
                    self.data[load_dict[name]] = pd.read_csv(self.pwd() + name)
                    continue
                assert False
            else:
                self.data[name] = load_dict[name]

    def _load_checked(self):
        # NOT perform load checked since a time-costly operation.
        pass


class WordLoader(VirtualLoader):
    pass

class ForeSpanLoader(VirtualLoader):
    pass

class EmbModelLoader(VirtualLoader):
    pass

class WordEmbLoader(VirtualLoader):
    pass

class VecDistLoader(VirtualLoader):
    pass

class EigWordLoader(VirtualLoader):
    pass

class PersonEmbLoader(VirtualLoader):
    pass
