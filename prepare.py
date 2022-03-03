from collections import defaultdict
import numpy as np
import json
import argparse
#这段代码应该是将NELL转换成NELL-one
#而我自己需要的是，如何将自己和NELL-one相同的数据转换成为NELL同类的数据。应该是一个逆操作
#BG：In-Train设定的理解：
#GMatching这个工作一开始会将所有样本放进去训练，然后学习到一个初始的embedding。然后在使用train、dev、test进行训练。
#这个所有样本，就是train + path_graph中的数据
args = argparse.ArgumentParser()
args.add_argument("-path", "--dataset_path", default="./NELL", type=str)  # ./Wiki
args.add_argument("-data", "--dataset_name", default="NELL-One", type=str)  # Wiki-One
params = args.parse_args()

dire = params.dataset_path
data = params.dataset_name

path = {
    'train_tasks': '/train_tasks.json',
    'test_tasks': '/test_tasks.json',
    'dev_tasks': '/dev_tasks.json',
    'rel2candidates': '/rel2candidates.json',
    'e1rel_e2': '/e1rel_e2.json',
    'path_graph': '/path_graph', #这个是原本的Wiki和NELL数据中有的
    'ent2emb': '/entity2vec.TransE'
}

print('Start')
print('Process {} in {}'.format(data, dire))

print("Loading jsons ... ...")
train_tasks = json.load(open(dire+path['train_tasks']))
test_tasks = json.load(open(dire+path['test_tasks']))
dev_tasks = json.load(open(dire+path['dev_tasks']))
e1rel_e2 = json.load(open(dire+path['e1rel_e2']))
path_graph_lines = open(dire+path['path_graph']).readlines()
rel2candidates = json.load(open(dire+path['rel2candidates']))
ent2emb = np.loadtxt(dire+path['ent2emb'], dtype=np.float32)

# convert entity2vec to .npy
np.save('ent2vec.npy', ent2emb) #TODO：逆操作，这里我需要将自己的ent2vec.npy转换成ent2emb文件。应该是每个实体的embedidng。如果我想随机的话，应该自己随机生成一个

entity = set()
path_graph = []
for line in path_graph_lines:
    triple = line.strip().split()
    entity.add(triple[0])
    entity.add(triple[2])
    path_graph.append(triple)
json.dump(path_graph, open(dire+'/path_graph.json', 'w')) #TODO：这里保存了path_graph，但是好像metaR的模型中不需要这个数据

# train_tasks_in_train
print("Writing train_tasks_in_train.json ... ...")
path_graph_tasks = defaultdict(list) #这里是构造一个path_graph_tasks字典，当通过这个字典查找值不存在的时候，返回一个空的列表（list）
for p in path_graph:
    path_graph_tasks[p[1]].append(p) #通过将关系p[1]为key,该关系下每个三元组组合成为列表
train_tasks_in_train = {**train_tasks, **path_graph_tasks} #这里完全是结构了，但是tranin和这个合并有什么特殊的意义吗？
json.dump(train_tasks_in_train, open(dire+'/train_tasks_in_train.json', 'w'))

# rel2candidates_in_train
if data == 'NELL-One':
    print("Writing rel2candidates_in_train.json ... ...")
    entity_dict = defaultdict(list)
    for ent in entity:
        s = ent.split(':')
        if len(s) != 3:
            entity_dict['num'].append(ent)
        else:
            entity_dict[s[1]].append(ent)

    rel2candidates_in_train = defaultdict(list)

    for rel, task in path_graph_tasks.items():
        types = []
        cands = []
        for i in task:
            e1, r, e2 = i
            s = e2.split(':')
            if len(s) != 3:
                types.append('num')
            else:
                types.append(s[1])
        types = set(types)
        for t in types:
            cands.extend(entity_dict[t])
        cands = list(set(cands))
        rel2candidates_in_train[rel] = cands

    rel2candidates_in_train = {**rel2candidates, **rel2candidates_in_train}
else:
    print("Writing rel2candidates_in_train.json ... ...")
    rel2candidates_in_train = defaultdict(list)
    for k, v in path_graph_tasks.items():
        cands = []
        for tri in v:
            cands.append(tri[2])
        cands = list(set(cands))
        rel2candidates_in_train[k] = cands
    rel2candidates_in_train = {**rel2candidates, **rel2candidates_in_train}

    for rel, cands in rel2candidates_in_train.items():
        if len(cands) == 1:
            one_cand = cands[0]
            for k, v in train_tasks_in_train.items():
                for tri in v:
                    h, r, t = tri
                    if t == one_cand:
                        cands.extend(rel2candidates_in_train[r])
                        break
                if len(cands) > 1:
                    break
            rel2candidates_in_train[rel] = list(set(cands))

json.dump(rel2candidates_in_train, open(dire + '/rel2candidates_in_train.json', 'w'))


# e1rel_e2_in_train
print("Writing e1rel_e2_in_train.json ... ...")
e1rel_e2_in_train = defaultdict(list)
for k, v in path_graph_tasks.items():
    for triple in v:
        e1, r, e2 = triple
        e1rel_e2_in_train[e1+r].append(e2)

e1rel_e2_in_train = {**e1rel_e2, **e1rel_e2_in_train}
json.dump(e1rel_e2_in_train, open(dire+'/e1rel_e2_in_train.json', 'w'))

print('End')


