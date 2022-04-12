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


class UserComplLoader(VirtualLoader):
    pass

if __name__ == '__main__':
    ewl1 = EigWordLoader(loader_name='eig-word')

    ewl1.register('person_type', ['神经质', '外向性', '开放性', '顺同性', '严谨性'])

    ewl1.register('word2person', {
        '色图': [26, 19, 27, 16, 13],
        '白捡': [25, 28, 38, 32, 34],
        '石坑': [24, 24, 30, 24, 27],
        '魔法': [27, 28, 34, 24, 24],
        '拉尼': [21, 32, 30, 30, 37],
        '肝帝': [26, 26, 30, 16, 34],
        '肉块': [24, 26, 30, 26, 24],
        '圣地': [24, 28, 34, 26, 27],
        '我淦': [28, 36, 28, 22, 16],
        'NB': [18, 35, 25, 33, 22],
        '延时': [20, 32, 34, 24, 32],
        '趁手': [24, 31, 35, 22, 27],
        '强不强': [25, 41, 23, 5, 30],
        '锁血': [15, 34, 42, 22, 39],
        '全图': [28, 30, 36, 24, 30],
        '多个': [27, 24, 34, 24, 24],
        '冲击': [16, 37, 38, 16, 27],
        '小葛': [26, 27, 33, 30, 22],
        '二十多': [27, 28, 33, 24, 32],
        '天上': [24, 24, 30, 24, 24],
        '点满': [30, 40, 26, 8, 28],
        '十几万': [30, 32, 28, 16, 27],
        'ok': [21, 28, 26, 32, 24],
        '一週目': [27, 31, 40, 24, 33],
        '教教': [19, 40, 28, 34, 20],
        '没闪': [27, 30, 33, 11, 41],
        '好玩': [20, 34, 30, 28, 24],
        '越玩越': [26, 26, 30, 16, 34],
        '龙鳞刀': [20, 26, 33, 18, 32],
        '非杀不可': [30, 38, 28, 6, 20],
        '盾太强': [26, 37, 34, 12, 24],
        '暴捶': [28, 42, 30, 8, 22],
        '过瘾': [25, 41, 34, 11, 24],
        '癫火': [27, 32, 32, 16, 27],
        '打土龙': [22, 31, 31, 20, 24],
        '神服': [36, 28, 28, 4, 16],
        '踹': [23, 31, 29, 20, 25],
        '三狗': [24, 29, 32, 24, 25],
        '开黑': [17, 40, 36, 18, 26],
        '加': [22, 31, 34, 16, 32],
        '情愿': [20, 34, 33, 36, 20],
        '双子': [24, 29, 31, 24, 24],
        '好惨': [27, 37, 36, 18, 34],
        '能削': [26, 38, 36, 14, 27],
        '身躯': [22, 24, 31, 24, 24],
        '桥下': [22, 24, 31, 24, 26],
        '绕圈': [22, 30, 33, 11, 26],
        '装逼': [30, 37, 30, 16, 20],
        '哥们': [17, 40, 36, 18, 26],
        '旧作': [24, 36, 37, 20, 34],
        '恶魔': [25, 27, 30, 22, 23],
        '所有人': [23, 27, 30, 22, 24],
        '死去': [25, 41, 23, 5, 30],
    })

    ewl1.push()
    print(ewl1.data_dict)
