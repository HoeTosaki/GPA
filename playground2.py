import re
import torch
import math
import numpy as np
from random import *
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thank you Romeol\n'  # J
    'Where are you going today?\n'  # R
    'I am going shopping. What about you?\n'  # J
    'I am going to visit my grandmother. she is not very well'  # R
)
"""
下面这段代码很有智慧，做了数据数据预处理，构建词表，构建双向字典（由idx向word，由word向idx）
在短短7行代码中完成
预处理思路：
肯定要先构建词典，(词对索引，索引对词，词典长度)
再根据词典，对应句子中(不管是一维还是二维)的词找其所对应的索引
"""
sentences = re.sub('[.,?!]', '', text.lower()).split('\n')
word_list = list(set(' '.join(sentences).split()))
word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

# 做token_list
token_list = []
for sen in sentences:
    token_list.append([word2idx[i] for i in sen.split()])  # [[29, 6, 32, 37, 33, 31, 13], [29, 13, 16, 22, 21...

# bert parameters
maxlen = 30
batch_size = 6
max_pred = 5  # max mask
n_layers = 1
n_heads = 12
d_model = 768
d_ff = 768 * 4  # 全连接神经网络的维度
d_k = d_v = 64
n_segments = 2  # 每一行由多少句话构成


# IsNext和NotNext的个数得一样
def make_data():
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15% of tokens in one sentence
        # 被mask的值不能是cls和sep：
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            # 将被mask位置的正确的位置和值保存下来
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            # 在原数据（input_ids）上替换为mask或者其他词
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']
            elif random() > 0.9:
                index = randint(4, vocab_size - 1)
                # 用词库中的词来替换，但是不能用cls，sep，pad，mask来替换
                # while index<4:
                #     index=randint(0,vocab_size-1)
                input_ids[pos] = index  # replace

        # Zero Padding
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # 给保存mask位置的值的列表补零，使之能参与运算
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # 需要确保正确样本数和错误样本数一样
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1
    """
    此时batch长度为6，每个长度中有四个量，分别是：
    input_ids：含有两句话以及cls，sep的id（其中的值已经被mask替换过了）。因为做了pad，此时batch中不同input_ids的长度相同
    segment_ids：0表示第一句话，1表示第二句话
    masked_tokens:保存的被替换的词，用于做loss
    masked_pos：保存的被替换的词在input_ids中的位置。
    """
    return batch


batch = make_data()
input_ids, segment_ids, masked_token, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_token, masked_pos, isNext = torch.LongTensor(input_ids), \
                                                           torch.LongTensor(segment_ids), torch.LongTensor(
    masked_token), torch.LongTensor(masked_pos), \
                                                           torch.LongTensor(isNext)


class MyDataSet(Dataset):
    def __init__(self, input_ids, segment_ids, masked_token, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_token = masked_token
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.segment_ids[item], self.masked_token[item], self.masked_pos[item], \
               self.isNext[item]


trainset = MyDataSet(input_ids, segment_ids, masked_token, masked_pos, isNext)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)


# 这里seq_k不用吗？？
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batchsize,1,seq_len]
    # eq(0)表示和0相等的返回True，不相等返回False。unsqueeze(1)在第一维上
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batchsize,seq_len,seq_len]


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(torch.nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = torch.nn.Embedding(vocab_size, d_model)
        self.pos_embed = torch.nn.Embedding(maxlen, d_model)
        self.seg_embed = torch.nn.Embedding(n_segments, d_model)
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(device)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size,seq_len]

        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


# 没有特别懂，需要结合下边的代码
class ScaleDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # matmul做内积，transpose做转置
        scores.masked_fill_(attn_mask, -1e9)  # fills elements of self tensor with value where mask is one.
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_K = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_V = torch.nn.Linear(d_model, d_k * n_heads)
        self.linear = torch.nn.Linear(n_heads * d_v, d_model)
        self.LN = torch.nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        # (B,S,D) --proj-> (B,S,D) --split-> (B,S,H,W) --trans-> (B,H,S,W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size,n_heads,seq_len,d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size,n_heads,seq_len,d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size,n_heads,seq_len,d_k]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # 将扩展的第二维扩张到n_heads。[batch_size,n_heads,seq_len,seq_len]

        # context:[batch_size,n_heads,seq_len,d_v],attn:[batch_size,n_heads,seq_len,seq_len]
        context = ScaleDotProductAttention().to(device)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.LN(output + residual)  # 这边应该是用到了残差吧
        # output: [batch_size,seq_len,d_model]


class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention().to(device)
        self.pos_ffn = PoswiseFeedForwardNet().to(device)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs


class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding().to(device)  # 这是上边定义的Embedding，self.embedding就是一个类对象
        self.layers = torch.nn.ModuleList([EncoderLayer().to(device) for _ in range(n_layers)])
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.Dropout(0.5),
            torch.nn.Tanh(),
        )
        self.classifier = torch.nn.Linear(d_model, 2)
        self.linear = torch.nn.Linear(d_model, d_model)
        self.activ2 = gelu  # 这是上边定义的gelu，self.activ2就是gelu函数
        # 这里为什么要用embed_weight的维度，回头再看一下
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)  # [batch_size,seq_len,d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size,maxlen,maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0])  # [batch_size,d_model]。且对于三维tensor，[:,0]和[:,0,:]效果是一样的
        # 且[]中出现了数字，他就是降维操作，所以output由三维降到两维。
        logits_clsf = self.classifier(h_pooled)
        # 得到判断两句话是不是上下句关系的结果

        # 得到被mask位置的词，准备与正确词进行比较
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size,max_pred,d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size,max_pred,d_model]
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size,max_pred,d_model]
        logits_lm = self.fc2(h_masked)  # [batch_size,max_pred,vocab_size]
        return logits_lm, logits_clsf  # 预测的被mask地方的值，预测两句话是否为上下句的结果


model = BERT()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.to(device)

for epoch in range(50):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in trainloader:
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        masked_pos = masked_pos.to(device)
        masked_tokens = masked_tokens.to(device)
        isNext = isNext.to(device)
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))
        loss_lm = (loss_lm.float()) / 5
        loss_clsf = criterion(logits_clsf, isNext)
        loss = loss_clsf + loss_lm
        if (epoch + 1) % 10 == 0:
            print('Epoch: %04d' % (epoch + 1), 'loss=', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
input_word = [idx2word[i] for i in input_ids]
input_ids = torch.LongTensor([input_ids]).to(device)
segment_ids = torch.LongTensor([segment_ids]).to(device)
masked_pos = torch.LongTensor([masked_pos]).to(device)
masked_tokens = torch.LongTensor(masked_tokens).to(device)
isNext = torch.LongTensor([isNext]).to(device)
print(text)
print('================================')
print([idx2word[w.item()] for w in input_ids[0] if idx2word[w.item()] != '[PAD]'])

logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
logits_lm = logits_lm.data.max(2)[1][0].data
# 最大值总是在第0个，代码应该是有问题的。且5个mask的值完全一样，所以明天好好看看代码
date = [i.item() for i in logits_lm]
for i in date:
    if i:
        print(f'被mask掉的词是{idx2word[i]}')
"""
max(2):再第2维上取最大值，取完后维度：[batch,max_pred],[batch,max_pred]。此时有两个维度值。第一个是具体的值，第二个是位置。（值不需要，就是一个经过了softmax后比其他值大的数）
[1]:取到最大值对应的位置。维度是：[batch,max_pred]
[0]:因为batch为1（只取了第一组值），所以此时维度是：[max_pred]
注：max会使tensor降维
"""
print('masked token list:', [pos.item() for pos in masked_tokens if pos != 0])
print('predict masked tokens list:', [pos.item() for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data[0]
print('isNext:', True if isNext.item() else False)
print('predict isNext:', True if logits_clsf else False)

if __name__ == '__main__':
    print('hello playground2')