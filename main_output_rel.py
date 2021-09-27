from trainer import *
from params import *
from data_loader import *
import json
import random
import numpy as np

'''
这个主函数的作用主要是根据已经训练好的模型，输入test_task数据集，然后输出分数，以便用于分析。中间设置了很多变量用于检查数据的形式
'''

if __name__ == '__main__':
    params = get_params()

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path']+v

    tail = ''
    if params['data_form'] == 'In-Train':
        tail = '_in_train'

    dataset = dict()
    print("loading train_tasks{} ... ...".format(tail))
    dataset['train_tasks'] = json.load(open(data_dir['train_tasks'+tail]))
    print("loading test_tasks ... ...")
    dataset['test_tasks'] = json.load(open(data_dir['test_tasks']))
    print("loading dev_tasks ... ...")
    dataset['dev_tasks'] = json.load(open(data_dir['dev_tasks']))
    print("loading rel2candidates{} ... ...".format(tail))
    dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates'+tail]))
    print("loading e1rel_e2{} ... ...".format(tail))
    dataset['e1rel_e2'] = json.load(open(data_dir['e1rel_e2'+tail]))
    print("loading ent2id ... ...")
    dataset['ent2id'] = json.load(open(data_dir['ent2ids']))

    if params['data_form'] == 'Pre-Train':
        print('loading embedding ... ...')
        dataset['ent2emb'] = np.load(data_dir['ent2vec'])

    print("----------------------------")

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]

    # trainer
    trainer = Trainer(data_loaders, dataset, params)

    if params['step'] == 'train':
        trainer.train()
        print("test") 
        print(params['prefix'])
        trainer.reload()
        trainer.eval(istest=True)
    elif params['step'] == 'test':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=True)
        else:
            trainer.eval(istest=True)
        
        #获取到所有的关系的embedding all_rel_dic:{key:value}
        all_rel_dic = trainer.metaR.rel_q_sharing

        for key_num, _ in enumerate(all_rel_dic.keys()):
            print('----------------')
            print(_)
            print('----------------')
            triples = dataset['test_tasks'][_] #导入当前rel的三元组
            #生成负样本
            negative = []
            e1rel_e2, curr_cand = dataset['e1rel_e2'], dataset['rel2candidates'][_]

            #把当前关系下的节点进行分类，方便排列组合
            compounds, genes = [], []
            for cand in curr_cand:
                if cand[0] == 'C':
                    compounds.append(cand)
                if cand[0] == 'G':
                    genes.append(cand)
            
            #进行排列组合
            #TODO:源代码进行h+r-t的时候，有没有考虑到绝对值的问题？
            for c in compounds:
                for g in genes:
                    if ([c, _, g] or [g, _, c]) in triples:
                        continue
                    else:
                        negative.append([c, _, g])
                
            print('正样本的个数', len(triples))
            print(triples[0])
            print('负样本的个数', len(negative))
            print(negative[0])
            #把正负样本传给meta.embedding,得到对应的embedding
            # triples_emb, negative_emb = [trainer.metaR.embedding(t) for t in [triples, negative]
            triples_emb = trainer.metaR.embedding([triples])
            negative_emb = trainer.metaR.embedding([negative])

            #把正样本和负样本的h和t用split.concat()函数连接在一起
            head_emb, tail_emb = trainer.metaR.split_concat(triples_emb, negative_emb)

            #获取关系rel的embedding
            rel_emb = all_rel_dic[_].expand(-1,(len(triples) + len(negative)), -1, -1)
            print('----------------------')

            print('head_emb:', head_emb.shape)
            print('tail_emb:', tail_emb.shape)
            print('rel_emb:', rel_emb.shape)
            #计算分数，p_score是正样本的分数，n_score是负样本的分数
            p_score, n_score = trainer.metaR.embedding_learner(head_emb, tail_emb, rel_emb, len(triples))

            #把分数进行排名，并且组合成我要的输出形式
            yes = [1 for i in range(len(triples))] #这里的1是正样本，但是不是所有的正样本都放进去训练了,放进去训练的是test样本中的前few个
            no = [0 for j in range(len(negative))] #这里的0是负样本，全部都是我自己生成的

            all_triples = triples + negative
            print('p_score:', p_score.shape)
            print('n_score:', n_score.shape)
            
            input('stop a moment')
            all_scores = (torch.cat((p_score, n_score),1).squeeze()).cpu().detach().numpy().tolist()
            all_imfo = yes + no
            zipped = zip(all_triples, all_scores, all_imfo)
            my_sort = list(sorted(zipped, key=lambda x: x[1], reverse=True))

            #写成文件
            with open('./test_result/new_result_{}.txt'.format(key_num), 'a', encoding='utf-8') as output:
                for row in my_sort:
                    #下面的5个信息分别表示 反应的类型、药物名字、基因名字、分数、真值label
                    rowtext = '{} {} {} {} {}'.format(row[0][1], row[0][0], row[0][2], row[1], row[2])
                    output.write(rowtext)
                    output.write('\n')
                # output.write('\n')
    elif params['step'] == 'dev':
        print(params['prefix'])
        if params['eval_by_rel']:
            trainer.eval_by_relation(istest=False)
        else:
            trainer.eval(istest=False)

