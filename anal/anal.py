import dataload.dataloader as dataloader
import os

class Anal:
    '''
        a general basic & abstract class that every analyzer is required to extend.
        4.10 (hoe) design for two types of usage situation:
            - type 1: dataloader-oriented anal, impl. fit/transform/fit_transform(by default)
                    , to generate new virtual dataloader.
            - type 2: user-oriented anal, impl. fit/print, to summarize valuable figures or tables.
        feel free to extend DoAnal for just type1 considered, and UoAnal for type2 as well.
    '''

    def __init__(self, anal_name='anony', out_dir='../outputs/', tmp_dir='../outputs/.tmp'):
        self.anal_name = anal_name
        self.out_dir = out_dir
        self.tmp_dir = tmp_dir

        self.__FORBIDDEN_OVERRIDE_SET__ = set()

    def __str__(self):
        return 'Anal:' + self.anal_name

    def pwd(self, is_tmp=True):
        return os.path.join(self.tmp_dir if is_tmp else self.out_dir, self.anal_name)

    def fit(self, dloader: dataloader.DataLoader):
        try:
            if '_fit' in self.__FORBIDDEN_OVERRIDE_SET__:
                raise NotImplementedError
            self._fit(dloader)
        except NotImplementedError:
            print('[{}] Warning no direct impl. of fit, use fit_transform instead.'.format(self))
            self.fit_transform()

    def transform(self, dloader: dataloader.DataLoader):
        try:
            if '_transform' in self.__FORBIDDEN_OVERRIDE_SET__:
                raise NotImplementedError
            return self._transform(dloader)
        except NotImplementedError:
            print('[{}] Warning no direct impl. of fit, use fit_transform instead.'.format(self))
            return self.fit_transform()

    def fit_transform(self, dloader: dataloader.DataLoader):
        self._fit(dloader)
        return self._transform(dloader)

    def print(self, to_screen=True):
        '''
            print anal output data to out_dir & optionally print to screen.
        '''
        if '_print' in self.__FORBIDDEN_OVERRIDE_SET__:
            raise NotImplementedError
        self._print(to_screen)

    def _print(self, to_screen):
        raise NotImplementedError

    def _fit(self, dloader: dataloader.DataLoader):
        raise NotImplementedError

    def _transform(self, dloader: dataloader.DataLoader):
        raise NotImplementedError


class DoAnal(Anal):
    def __init__(self, data_dir='../data', tmp_dir='../data/.tmp', **kwargs):
        '''
            dataloader-oriented anal, impl. fit/transform/fit_transform(by default), to generate new virtual dataloader.
            recommend tmp_dir to be included in data_dir.
        '''
        kwargs['tmp_dir'] = tmp_dir
        super(DoAnal, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.__FORBIDDEN_OVERRIDE_SET__ = {'_print'}


class UoAnal(Anal):
    def __init__(self, **kwargs):
        '''
            user-oriented anal, impl. fit/print, to summarize valuable figures or tables.
        '''
        super(UoAnal, self).__init__(**kwargs)
        self.__FORBIDDEN_OVERRIDE_SET__ = {'_transform'}


if __name__ == '__main__':
    print('hello anal.')
