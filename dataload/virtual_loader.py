import json
import numpy as np
import pandas as pd
from dataload import dataloader


class VirtualLoader(dataloader.DataLoader):
    '''
        functionally-similar loader as Dataloader, transformed from DoAnal.
    '''

    def __init__(self, **kwargs):
        super(VirtualLoader, self).__init__(**kwargs)

        self.data_dict = {'inner': {}, 'np': {}, 'pd': {}}
        self.data = None

    def __str__(self):
        return 'VirtualLoader:' + self.loader_name

    def register(self, data_name, data, data_type='inner'):
        '''
            register sub-data for storage and loading.
        '''
        assert data_type in ['inner', 'np', 'pd']
        assert data_name not in self.data_dict[data_type]
        self.data_dict[data_type][data_name] = data

    def push(self):
        data_dict = {}
        data_dict.update(self.data_dict['inner'])
        data_dict.update({'.' + name + '.npy': name for name in self.data_dict['np']})
        data_dict.update({'.' + name + '.csv': name for name in self.data_dict['pd']})
        with open(self.pwd() + '.json', 'w') as f:
            json.dump(data_dict, f)
        for name in self.data_dict['np']:
            np.save(self.pwd() + '.' + name + '.npy', self.data_dict['np'][name])
        for name in self.data_dict['pd']:
            self.data_dict['pd'][name].to_csv(self.pwd() + '.' + name + '.csv')

    def _fetch(self):
        pass

    def _parse(self):
        pass

    def _load(self):
        with open(self.pwd() + '.json', 'r') as f:
            load_dict = json.load(f)
        assert load_dict is not None, print('[{}] load failed at path:{}'.format(self, self.pwd() + '.json'))
        self.data = {}
        for name in load_dict:
            if name[:1] == '.':
                if name[-4:] == '.npy':
                    self.data[load_dict[name]] = np.load(self.pwd() + name)
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


if __name__ == '__main__':
    ewl1 = EigWordLoader()

    ewl1.register('person_type', ['神经质', '外向性', '开放性', '顺同性', '严谨性'])

    # ewl1.register('色图', [26, 19, 27, 16, 13])
    # ewl1.register('拉尼', [21, 32, 30, 30, 37])
    # ewl1.register('肝帝', [26, 26, 30, 16, 34])
    # ewl1.register('强不强', [25, 41, 23, 5, 30])
    # ewl1.register('锁血', [15, 34, 42, 22, 39])
    # ewl1.register('冲击', [16, 37, 38, 16, 27])
    # ewl1.register('点满', [30, 40, 26, 8, 28])
    # ewl1.register('没闪', [27, 30, 33, 11, 41])
    # ewl1.register('越玩越', [26, 26, 30, 16, 34])
    # ewl1.register('龙鳞刀', [20, 26, 33, 18, 32])
    # ewl1.register('暴捶', [28, 42, 30, 8, 22])
    # ewl1.register('过瘾', [25, 41, 34, 11, 24])
    # ewl1.register('神服', [36, 28, 28, 4, 16])
    # ewl1.register('开黑', [17, 40, 36, 18, 26])
    # ewl1.register('加', [22, 31, 34, 16, 32])
    # ewl1.register('好惨', [27, 37, 36, 18, 34])
    # ewl1.register('绕圈', [22, 30, 33, 11, 26])
    # ewl1.register('哥们', [17, 40, 36, 18, 26])
    # ewl1.register('死去', [25, 41, 23, 5, 30])

    ewl1.register('dic', {
        '色图': [26, 19, 27, 16, 13]
        , '拉尼': [21, 32, 30, 30, 37]
        , '肝帝': [26, 26, 30, 16, 34]
        , '强不强': [25, 41, 23, 5, 30]
        , '锁血': [15, 34, 42, 22, 39]
        , '冲击': [16, 37, 38, 16, 27]
        , '点满': [30, 40, 26, 8, 28]
        , '没闪': [27, 30, 33, 11, 41]
        , '越玩越': [26, 26, 30, 16, 34]
        , '龙鳞刀': [20, 26, 33, 18, 32]
        , '暴捶': [28, 42, 30, 8, 22]
        , '过瘾': [25, 41, 34, 11, 24]
        , '神服': [36, 28, 28, 4, 16]
        , '开黑': [17, 40, 36, 18, 26]
        , '加': [22, 31, 34, 16, 32]
        , '好惨': [27, 37, 36, 18, 34]
        , '绕圈': [22, 30, 33, 11, 26]
        , '哥们': [17, 40, 36, 18, 26]
        , '死去': [25, 41, 23, 5, 30]
    })

    print(ewl1.data_dict)
