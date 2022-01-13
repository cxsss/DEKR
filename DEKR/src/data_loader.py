import torch
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataLoader:

    def __init__(self, dataset_name):
        self.cfg = {
            'MachineLearning': {
                'item2entity_id_path': '../data/MachineLearning/method_index2entity_id_rehashed.txt',
                'kg_path': '../data/MachineLearning/mlkg_rehashed.txt',
                'desc_path': '../data/MachineLearning/entity_desc_embed_vector.npy',
                'rating_path': '../data/MachineLearning/dataset-method_ratings.csv',
                'rating_sep': ',',
                'threshold': 0.0
            }
        }
        self.data = dataset_name

        df_item2entity_id = pd.read_csv(self.cfg[dataset_name]['item2entity_id_path'], sep='\t', header=None, names=['item_id', 'entity_id'])
        df_kg = pd.read_csv(self.cfg[dataset_name]['kg_path'], sep='\t', header=None, names=['head', 'relation', 'tail'])
        df_rating = pd.read_csv(self.cfg[dataset_name]['rating_path'], sep=self.cfg[dataset_name]['rating_sep'],
                                names=['user', 'item', 'rating'], skiprows=1)

        # df_rating['itemID'] and df_item2id['item'] both represents old entity ID
        df_rating = df_rating[df_rating['item'].isin(df_item2entity_id['item_id'])]
        df_rating.reset_index(inplace=True, drop=True)

        self.df_item2entity_id = df_item2entity_id
        self.df_kg = df_kg
        self.df_rating = df_rating

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()

        self.user_old2new_dict, self.item_old2new_dict, self.entity_old2new_dict, self.entity_cnt, self.relation_cnt = self._reindex()

    def _reindex(self):

        self.user_encoder.fit(self.df_rating['user'])
        self.item_encoder.fit(self.df_item2entity_id['item_id'])
        self.entity_encoder.fit(pd.concat([self.df_item2entity_id['item_id'], self.df_rating['user']]))
        # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
        user_old2new_dict = dict(zip(self.df_rating['user'], self.user_encoder.transform(self.df_rating['user'])))
        item_old2new_dict = dict(zip(self.df_item2entity_id['item_id'], self.item_encoder.transform(self.df_item2entity_id['item_id'])))
        entity_old2new_dict = dict(zip((pd.concat([self.df_item2entity_id['item_id'], self.df_rating['user']])),
                                       (self.entity_encoder.transform(pd.concat([self.df_item2entity_id['item_id'], self.df_rating['user']])))))

        entity_id2index_dict = dict(zip(self.df_item2entity_id['entity_id'], self.item_encoder.transform(self.df_item2entity_id['item_id'])))
        entity_cnt = len(self.entity_encoder.classes_)
        relation_cnt = 0
        relation_id2index_dict = dict()
        # print(entity_id2index_dict)
        head_list = []
        relation_list = []
        tail_list = []
        for line in self.df_kg.index:
            triple = self.df_kg.loc[line].values
            head_old = triple[0]
            relation_old = triple[1]
            tail_old = triple[2]

            if head_old not in entity_old2new_dict:
                entity_old2new_dict[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_old2new_dict[head_old]

            if tail_old not in entity_old2new_dict:
                entity_old2new_dict[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_old2new_dict[tail_old]

            if relation_old not in relation_id2index_dict:
                relation_id2index_dict[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index_dict[relation_old]
            head_list.append(head)
            relation_list.append(relation)
            tail_list.append(tail)
        kg_final = pd.DataFrame({'head': head_list, 'relation': relation_list, 'tail': tail_list})
        self.df_kg = kg_final
        return user_old2new_dict, item_old2new_dict, entity_old2new_dict, entity_cnt, relation_cnt
        # print(self.df_kg)

    def _prepare_dataset(self):

        print('preparing rating data ...', end=' ')
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['user'] = self.entity_encoder.transform(self.df_rating['user'])

        # update to new id
        item2entity_dict = dict(zip(self.df_item2entity_id['item_id'], self.df_item2entity_id['entity_id']))
        self.df_rating['item'] = self.df_rating['item'].apply(lambda x: item2entity_dict[x])
        df_dataset['item'] = self.entity_encoder.transform(self.df_rating['item'])
        df_dataset['label'] = self.df_rating['rating'].apply(lambda x: 0 if x < self.cfg[self.data]['threshold'] else 1)

        # negative sampling
        df_dataset = df_dataset[df_dataset['label'] == 1]
        # df_dataset requires columns to have new entity ID
        full_item_set = set(range(len(self.item_encoder.classes_)))
        user_list = []
        item_list = []
        label_list = []
        for user, group in df_dataset.groupby(['user']):
            item_set = set(group['item'])
            negative_set = full_item_set - item_set
            negative_sampled = random.sample(negative_set, len(item_set))
            user_list.extend([user] * len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0] * len(negative_sampled))
        negative = pd.DataFrame({'user': user_list, 'item': item_list, 'label': label_list})
        df_dataset = pd.concat([df_dataset, negative])
        df_dataset = df_dataset.sort_values('user')
        df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset

    def _construct_kg(self):

        print('preparing knowledge graph data...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
        print('Done')
        return kg

    def get_entity_desc_embed_vector(self):
        entity_desc_embed_vector = np.load(self.cfg[self.data]['desc_path'], allow_pickle=True)
        return entity_desc_embed_vector

    def load_dataset(self):
        return self._prepare_dataset()

    def load_kg(self):
        return self._construct_kg()

    def get_num(self):
        return len(self.user_encoder.classes_), len(self.item_encoder.classes_), self.entity_cnt, self.relation_cnt


class ModelDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['user'])
        item_id = np.array(self.df.iloc[idx]['item'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label

