from torch.utils.data import Dataset, DataLoader
import random

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
