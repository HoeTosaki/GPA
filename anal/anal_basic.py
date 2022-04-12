import random

from anal import *
import numpy as np
import pandas as pd
from dataload.virtual_loader import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from ast import literal_eval
from wordcloud import WordCloud
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import MinMaxScaler


class UserAnal(DoAnal):
    def __init__(self,word_loader:WordLoader,**kwargs):
        super(UserAnal, self).__init__(**kwargs)
        self.word_loader = word_loader

    def fit_transform(self, pemb_loader:PersonEmbLoader):
        self.word_loader.load()
        pemb_loader.load()
        pd_data = self.word_loader.data['data']
        word2id = pemb_loader.data['word2id']
        persons = pemb_loader.data['persons']
        person_type = pemb_loader.data['person_type']
        def __chat_word_to_person__(x):
            x = list(x)
            cur_person = np.zeros(shape=(len(person_type),))
            cnt = 0
            for xx in x:
                if xx in word2id:
                    cur_person += persons[word2id[xx]]
                    cnt += 1
            if cnt == 0:
                cur_person = np.ones(shape=(len(person_type),))*0.5
            else:
                cur_person /= cnt
            return cur_person.tolist()

        # def __gen_per_type__(x):
        #     x = np.array(list(x))
        #     return person_type[int(x.argmax())]

        pd_data['personality'] = pd_data['chat_words'].apply(__chat_word_to_person__)
        # pd_data['person_type'] = pd_data['personality'].apply(__gen_per_type__)
        user_person = []
        users = list(pd_data['user_id'].unique())
        for user in users:
            lst = np.zeros(shape=(len(person_type),))
            cnt = 0
            for val in list(pd_data[pd_data['user_id'] == user]['personality'].values):
                lst += np.array(val)
                cnt += 1
            if cnt > 0:
                lst /= cnt
            cur_row = [user]
            [cur_row.append(float(ele)) for ele in lst]
            cur_row.append(person_type[lst.argmax()])
            user_person.append(cur_row)
        cols = ['user_id']
        [cols.append(ele) for ele in person_type]
        cols.append('person_type')
        user_data = pd.DataFrame(user_person,columns=cols)

        for per_type in person_type:
            vals = np.array(list(user_data[per_type].values))
            vals /= np.median(vals)
            mm = MinMaxScaler()
            vals = mm.fit_transform(vals.reshape(-1,1))
            user_data[per_type+'@nor'] = pd.Series(data=vals.reshape(-1))
        ucl = UserComplLoader(loader_name=self.anal_name)
        ucl.register('person_type',person_type)
        ucl.register('data',pd_data,'pd')
        ucl.register('user_data', user_data, 'pd')

        ucl.push()

class SinFactorAnal(UoAnal):
    def __init__(self,user_compl_loader:UserComplLoader,**kwargs):
        super(SinFactorAnal, self).__init__(**kwargs)
        self.user_compl_loader = user_compl_loader

    def _print(self, to_screen):
        self.user_compl_loader.load()
        pd_data = self.user_compl_loader.data['data']
        self.per_types = self.user_compl_loader.data['person_type']
        user_data = self.user_compl_loader.data['user_data']

        user2per = {}
        for idx in range(len(user_data)):
            user2per[str(user_data['user_id'][idx])] = []
            for pertype in self.per_types:
                user2per[str(user_data['user_id'][idx])].append((float(user_data[pertype][idx]),float(user_data[pertype+'@nor'][idx])))
        for idx, pertype in enumerate(self.per_types):
            pd_data[pertype] = pd_data.apply(lambda x: user2per[x['user_id']][idx][0], axis=1)
            pd_data[pertype+'@nor'] = pd_data.apply(lambda x: user2per[x['user_id']][idx][1], axis=1)

        self.stop_words = list(pd_data['chat_lv'].unique())

        # # 游戏讨论热度变化趋势
        # self._game_hot(pd_data)
        #
        # # 游戏讨论周期性变化趋势
        # self._game_hot_interval(pd_data)
        #
        # # 游戏截图动态发布趋势
        # self._game_hot_pic(pd_data)
        #
        # # 游戏截图动态周期性趋势
        # self._game_hot_pic_interval(pd_data)
        #
        # # 一日内游戏热度统计
        # self._game_hot_time(pd_data)
        #
        # # 一日内游戏截图动态统计
        # self._game_hot_time_pic(pd_data)
        #
        # # 玩家整体大五人格分布
        # self._user_total_per(pd_data,user_data)

        # # 不同人格发言热度归一化趋势
        # self._per_hot(pd_data,user_data)

        # # 不同人格发言时段归一化趋势
        # self._per_hot_day(pd_data,user_data)
        #
        # # 整体游戏讨论词云
        # self._word_cloud_total(pd_data)
        #
        # # 不同人格的词云分布
        # self._word_cloud_per(pd_data,user_data)

    def _word_cloud_per(self,pd_data,user_data):
        glb_personal_actived = 0.5
        for per_type in self.per_types:
            self._gen_word_cloud(data=pd_data[pd_data[per_type+'@nor'] > glb_personal_actived],add_name='.per-{}-wcloud'.format(per_type))

    def _word_cloud_total(self,pd_data):
        self._gen_word_cloud(data=pd_data,add_name='.total-wcloud')

    def _gen_word_cloud(self,data,add_name):
        words = data['chat_words']
        word_lst = []

        for val in list(words.values):
            lst = literal_eval(val)
            for ele in lst:
                ele = ele.strip()
                if (ele not in [r'\u3000', '']) and (ele not in self.stop_words):
                    try:
                        _ = float(ele)
                    except ValueError:
                        word_lst.append(ele)
        word_dic = {}
        for word in word_lst:
            word_dic[word] = word_dic.get(word, 0) + 1
        word_dic_r = {}
        for word in word_dic:
            if 5 > len(word) > 1:
                word_dic_r[word] = word_dic[word]
        wordcloud = WordCloud(font_path='msyh.ttc',background_color="white",scale=4,min_font_size=4).generate_from_frequencies(word_dic_r,max_font_size=128)
        wordcloud.to_file(self.pwd(False)+add_name+'.jpg')

    def _per_hot(self,pd_data,user_data):
        glb_personal_actived = 0.5
        dates = list(pd_data['date'].unique())
        per_cnts = []
        for date in dates:
            data = pd_data[pd_data['date'] == date]
            cur_cnts = []
            for per_type in self.per_types:
                cur_cnts.append(len(data[data[per_type+'@nor'] > glb_personal_actived]))
            sum_cur_cnts = sum(cur_cnts)
            per_cnts.append([ele/sum_cur_cnts for ele in cur_cnts])
        per_cnts = np.array(per_cnts).T
        cs = ['coral','dodgerblue','brown','red','pink']

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        for idx in range(len(per_cnts)):
            ax.plot(dates,per_cnts[idx,:],c=cs[idx],label=self.per_types[idx])
        plt.title('不同人格发言热度按人格归一化趋势')
        plt.xlabel('日期')
        plt.ylabel('当日发言比例/条')
        plt.legend()
        plt.xticks(rotation=30)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        plt.tight_layout()
        plt.savefig(self.pwd(False) + '.per_hot@per.svg')
        plt.show()

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        for irow in range(per_cnts.shape[0]):
            per_cnts[irow,:] /= np.sum(per_cnts[irow,:])
            per_cnts[irow,:][np.isnan(per_cnts[irow,:])] = 0
        for idx in range(len(per_cnts)):
            ax.plot(dates,per_cnts[idx,:],c=cs[idx],label=self.per_types[idx])
        plt.title('不同人格发言热度按发言数归一化趋势')
        plt.xlabel('日期')
        plt.ylabel('当日发言比例/条')
        plt.legend()
        plt.xticks(rotation=30)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        plt.tight_layout()
        plt.savefig(self.pwd(False) + '.per_hot@chat.svg')
        plt.show()


    def _per_hot_day(self,pd_data,user_data):
        glb_personal_actived = 0.5
        times = list(pd_data['time'].unique())
        xs = ['{}时'.format(ele) for ele in range(24)]
        time_chat_cnt = [[] for _ in range(len(xs))]

        per_cnts = []
        for idx in range(len(xs)):
            data = pd_data[pd_data['time'].transform(lambda x: int(str(x).split(':')[0]) == idx)]
            cur_cnts = []
            for per_type in self.per_types:
                cur_cnts.append(len(data[data[per_type+'@nor'] > glb_personal_actived]))
            sum_cur_cnts = sum(cur_cnts)
            per_cnts.append([ele/sum_cur_cnts if sum_cur_cnts > 0 else 0 for ele in cur_cnts])
        per_cnts = np.array(per_cnts).T
        per_cnts = per_cnts[:,9:]
        xs = xs[9:]
        cs = ['coral','dodgerblue','brown','red','pink']

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        for idx in range(len(per_cnts)):
            ax.plot(xs,per_cnts[idx,:],c=cs[idx],label=self.per_types[idx])
        plt.title('不同人格发言热度按人格归一化趋势')
        plt.xlabel('日期')
        plt.ylabel('当日发言比例/条')
        plt.legend()
        plt.xticks(rotation=30)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        plt.tight_layout()
        plt.savefig(self.pwd(False) + '.per_hot@per.svg')
        plt.show()

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        for irow in range(per_cnts.shape[0]):
            per_cnts[irow,:] /= np.sum(per_cnts[irow,:])
            per_cnts[irow,:][np.isnan(per_cnts[irow,:])] = 0
        for idx in range(len(per_cnts)):
            ax.plot(xs,per_cnts[idx,:],c=cs[idx],label=self.per_types[idx])
        plt.title('不同人格发言热度按发言数归一化趋势')
        plt.xlabel('日期')
        plt.ylabel('当日发言比例/条')
        plt.legend()
        plt.xticks(rotation=30)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        plt.tight_layout()
        plt.savefig(self.pwd(False) + '.per_hot@chat.svg')
        plt.show()


    def _user_total_per(self,pd_data,user_data):
        glb_personal_actived = 0.5
        user_cnt = [0] * len(self.per_types)

        for idx,per_type in enumerate(self.per_types):
            user_cnt[idx] = len(user_data[user_data[per_type+'@nor'] >= glb_personal_actived])

        plt.figure(figsize=(6, 6))
        label = self.per_types
        explode = [0.01]*len(self.per_types)
        patches, l_text, p_text = plt.pie(user_cnt, explode=explode, labels=label, autopct='%1.2f%%')

        plt.suptitle('玩家群体大五人格总体分布', fontsize=16, y=0.93)
        plt.legend(bbox_to_anchor=(-0.04, 1), borderaxespad=0, frameon=False)
        plt.savefig(self.pwd(False)+'.pie-total-person.svg')
        plt.show()

    def _game_hot_interval(self,pd_data):
        dates = list(pd_data['date'].unique())
        xs = ['周六','周日','周一','周二','周三','周四','周五']
        date_chat_cnt = [[] for _ in range(len(xs))]
        for idx,date in enumerate(dates):
            date_chat_cnt[idx%len(xs)].append(len(pd_data[pd_data['date'] == date]))
        date_chat_min = []
        date_chat_max = []
        date_chat_mean = []
        for idx in range(len(xs)):
            date_chat_min.append(min(date_chat_cnt[idx]))
            date_chat_max.append(max(date_chat_cnt[idx]))
            date_chat_mean.append(sum(date_chat_cnt[idx]) / len(date_chat_cnt[idx]))

        fig, ax = plt.subplots(1, 1)
        plt.fill_between(xs,date_chat_min,date_chat_max,alpha=0.5)
        ax.plot(xs, date_chat_mean,c='coral')
        plt.title('游戏讨论热度周期性变化')
        plt.xlabel('周内日期')
        plt.ylabel('当日发言总数/条')
        plt.xticks(rotation=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.tight_layout()
        plt.savefig(self.pwd(False) + '.game_hot_interval.svg')
        plt.show()

    def _game_hot(self,pd_data):
        dates =  list(pd_data['date'].unique())
        date_chat_cnt = []
        for date in dates:
            date_chat_cnt.append(len(pd_data[pd_data['date'] == date]))

        fig, ax = plt.subplots(1, 1)
        ax.plot(dates,date_chat_cnt)
        plt.title('发售日起游戏讨论热度趋势')
        plt.xlabel('日期')
        plt.ylabel('当日发言总数/条')
        plt.xticks(rotation=30)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        plt.tight_layout()
        plt.savefig(self.pwd(False)+'.game_hot.svg')
        plt.show()

    def _game_hot_pic(self,pd_data):
        dates = list(pd_data['date'].unique())
        date_chat_cnt = []
        for date in dates:
            data = pd_data[pd_data['date'] == date]
            date_chat_cnt.append(data['pic_used'].sum())

        fig, ax = plt.subplots(1, 1)
        ax.plot(dates, date_chat_cnt)
        plt.title('发售日起游戏截图动态趋势')
        plt.xlabel('日期')
        plt.ylabel('当日截图动态/张')
        plt.xticks(rotation=30)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        plt.tight_layout()
        plt.savefig(self.pwd(False) + '.game_hot_pic.svg')
        plt.show()

    def _game_hot_pic_interval(self,pd_data):
        dates = list(pd_data['date'].unique())
        xs = ['周六','周日','周一','周二','周三','周四','周五']
        date_chat_cnt = [[] for _ in range(len(xs))]
        for idx,date in enumerate(dates):
            date_chat_cnt[idx%len(xs)].append(pd_data[pd_data['date'] == date]['pic_used'].sum())
        date_chat_min = []
        date_chat_max = []
        date_chat_mean = []
        for idx in range(len(xs)):
            date_chat_min.append(min(date_chat_cnt[idx]))
            date_chat_max.append(max(date_chat_cnt[idx]))
            date_chat_mean.append(sum(date_chat_cnt[idx]) / len(date_chat_cnt[idx]))

        fig, ax = plt.subplots(1, 1)
        plt.fill_between(xs,date_chat_min,date_chat_max,alpha=0.5)
        ax.plot(xs, date_chat_mean,c='coral')
        plt.title('游戏截图动态周期性变化')
        plt.xlabel('周内日期')
        plt.ylabel('当日截图动态/张')
        plt.xticks(rotation=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.tight_layout()
        plt.savefig(self.pwd(False) + '.game_hot_pic_interval.svg')
        plt.show()

    def _game_hot_time(self,pd_data):
        times = list(pd_data['time'].unique())
        xs = ['{}时'.format(ele) for ele in range(24)]
        time_chat_cnt = [[] for _ in range(len(xs))]

        for idx in range(len(xs)):
            data = pd_data[pd_data['time'].transform(lambda x: int(str(x).split(':')[0]) == idx)]
            dates = list(data['date'].unique())
            for date in dates:
                time_chat_cnt[idx].append(len(data[data['date'] == date]))
        time_chat_min = []
        time_chat_max = []
        time_chat_mean = []
        for idx in range(len(xs)):
            if len(time_chat_cnt[idx]) == 0:
                time_chat_min.append(0)
                time_chat_mean.append(0)
                time_chat_max.append(0)
            else:
                time_chat_min.append(min(time_chat_cnt[idx]))
                time_chat_max.append(max(time_chat_cnt[idx]))
                time_chat_mean.append(sum(time_chat_cnt[idx]) / len(time_chat_cnt[idx]))
        xs = xs[9:]
        time_chat_min = time_chat_min[9:]
        time_chat_max = time_chat_max[9:]
        time_chat_mean = time_chat_mean[9:]

        fig, ax = plt.subplots(1, 1)
        plt.fill_between(xs,time_chat_min,time_chat_max,alpha=0.5)
        ax.plot(xs, time_chat_mean,c='coral')
        plt.title('一日内游戏热度周期性变化')
        plt.xlabel('小时')
        plt.ylabel('当前小时发言/条')
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.tight_layout()
        plt.savefig(self.pwd(False) + '.game_hot_day.svg')
        plt.show()

    def _game_hot_time_pic(self, pd_data):
        times = list(pd_data['time'].unique())
        xs = ['{}时'.format(ele) for ele in range(24)]
        time_chat_cnt = [[] for _ in range(len(xs))]

        for idx in range(len(xs)):
            data = pd_data[pd_data['time'].transform(lambda x: int(str(x).split(':')[0]) == idx)]
            dates = list(data['date'].unique())
            for date in dates:
                time_chat_cnt[idx].append(data[data['date'] == date]['pic_used'].sum())
        time_chat_min = []
        time_chat_max = []
        time_chat_mean = []
        for idx in range(len(xs)):
            if len(time_chat_cnt[idx]) == 0:
                time_chat_min.append(0)
                time_chat_mean.append(0)
                time_chat_max.append(0)
            else:
                time_chat_min.append(min(time_chat_cnt[idx]))
                time_chat_max.append(max(time_chat_cnt[idx]))
                time_chat_mean.append(sum(time_chat_cnt[idx]) / len(time_chat_cnt[idx]))
        xs = xs[9:]
        time_chat_min = time_chat_min[9:]
        time_chat_max = time_chat_max[9:]
        time_chat_mean = time_chat_mean[9:]

        fig, ax = plt.subplots(1, 1)
        plt.fill_between(xs, time_chat_min, time_chat_max, alpha=0.5)
        ax.plot(xs, time_chat_mean, c='coral')
        plt.title('一日内截图动态周期性变化')
        plt.xlabel('小时')
        plt.ylabel('当前小时截图/张')
        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.tight_layout()
        plt.savefig(self.pwd(False) + '.game_hot_day_pic.svg')
        plt.show()


if __name__ == '__main__':
    print('hello anal basic.')
    # ua = UserAnal(anal_name='user-compl',word_loader=WordLoader(loader_name='word-sep'))
    # ua.fit_transform(pemb_loader=PersonEmbLoader(loader_name='vec-lab'))

    sfa = SinFactorAnal(anal_name='sin-factor',user_compl_loader=UserComplLoader(loader_name='user-compl'))
    sfa.print()

    # print(random.randint(0,48))
