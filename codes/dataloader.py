#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import pandas as pd
import torch
import random

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, ego_network_data, 
                 do_similarity_injection=False, do_similarity_corruption=True, top_x_percent=0.0, 
                 simmat_drugs=None, simmat_prots=None, entity2id=None, relation2id=None, seed=42):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.entity2id = entity2id
        self.relation2id = relation2id
        if entity2id is not None: self.id2entity = {v: k for k, v in entity2id.items()}
        if entity2id is not None: self.id2relation = {v: k for k, v in relation2id.items()}
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.ego_network_data = ego_network_data
        if self.ego_network_data is not None:
            self.true_neg_samples_head, self.true_neg_samples_tail = self.get_true_neg_samples(self.triples, self.ego_network_data, self.true_head, self.true_tail)
        
        # Build similarity dict and use for corruption and injection if requested
        if (entity2id is not None) and (relation2id is not None):
            # 1. Build similarity dict
            if do_similarity_corruption or do_similarity_injection:
                # Dictionary of similarities for drugs AND proteins. Structure stated in build_similarity_dict()
                print("--- Start Building Similarity Dictionary ---")
                self.simdict = {}
                if simmat_drugs is not None: self.build_similarity_dict(simmat_drugs, self.calc_thresh(simmat_drugs, top_x_percent))
                if simmat_prots is not None: self.build_similarity_dict(simmat_prots, self.calc_thresh(simmat_prots, top_x_percent))
                print(self.simdict)
                print("--- End Building Similarity Dictionary ---")
            # 2. Corruption
            if do_similarity_corruption:
                print("------- Start Similarity Corruption -------")
                random.seed(seed)
                self.similarity_head_tail_corruption()
                print("------- End Similarity Corruption -------")
            # 3. Injection
            if do_similarity_injection:
                print("------- Start Similarity Injection -------")
                if self.simdict: self.inject_similarity_triples(self.simdict)
                print("------- End Similarity Injection -------")
        self.triple_set = set(triples)


    @staticmethod
    def calc_thresh(simmat: pd.DataFrame, top_x_percent: float):
        """
        Calculates and returns similarity threshold for similar entities.

        :param simmat:
            Similarity score matrix of entities.
        :param top_x_percent:
            The top X percent of matrix values that we choose to mark similar entities.
        :return:
            A threshold value in the matrix that a given percent of all other values in the matrix are higher than.
        """
        if top_x_percent <= 0 or top_x_percent > 100:
            raise ValueError("Please, provide a number between 0 and 100 for top_x_percent.")

        a = simmat.to_numpy()
        a[a >= 1] = None    # Remove '1's to avoid cases where we only get values on the main diagonal

        percentile = 100 - top_x_percent
        thresh = np.nanpercentile(a, percentile)
        return thresh
    

    def build_similarity_dict(self, simmat: pd.DataFrame, thresh: float):
        """
        Extracts all entity pairs from the matrix, that have a similarity score higher than thresh, in the form of a nested dictionary:
        {
            "entity_A": {
                "similar_entity_1": <score>,
                "similar_entity_2": <score>
            },
            "entity_B": ...
        }

        :param matrix:
            Similarity score matrix of entities.
        :param thresh:
            Similarity score, above which an entity pair is viewed as similar.
        """
        # Iterate through matrix columns
        for col in simmat.columns:
            if col in self.entity2id.keys():
                # Find indexes where values are greater than thresh
                indexes = simmat.index[simmat[col] > thresh].tolist()
                # Remove indexes same as column
                if col in indexes: indexes.remove(col)
                # Remove indexes not present in entities.dict
                for index in indexes:
                    if not index in self.entity2id.keys(): indexes.remove(index)
                
                if indexes:
                    # Create inner-dictionary of indexes and similarity values
                    index_value_dict = {index: value for index, value in zip(indexes, simmat[col][indexes])}
                    # Put inner-dicationary into outer-dictionary of columns
                    self.simdict[col] = index_value_dict
    

    def inject_similarity_triples(self, simdict: pd.DataFrame):
        """
        Injects similarity triples (entity_A, sameAs, entity_B) into self.triples
        """
        # List of tuples of similar entities
        pairs = []

        # Iterate through simdict, extract entities and inject (and append to pairs list for optional reasons)
        for col, indexes in simdict.items():
            for index in indexes.keys():
                pairs.append((col, index))
                self.triples.append((self.entity2id[col], self.relation2id['sameAs'], self.entity2id[index]))
        
        # Convert list of tuples to DataFrame and insert relation
        df = pd.DataFrame.from_records(pairs, columns=['head', 'tail'])
        df.insert(loc=1, column='relation', value="sameAs")
        print("\nDataframe of all similar entity pairs:\n", df)
        df.to_csv("./TEST_injected_similarity_triples.txt", sep='\t', header=False, index=False)

    
    def similarity_head_tail_corruption(self):
        """
        For each triple (protein_A, interactsWith, drug_A) two positive triples are added - one with a corrupted head 
        and the other with corrupted tail. During corruption head/tail is replaced by randomly picking one entity 
        from the similar entities of that head/tail. The distribution, the random pick is based on, is made from the 
        similarity values between the corrupting and the corrupted entity. Higher similarity values lead to a higher
        chance for a entity to corrupt the head/tail.
        """
        # List of corrupted triples
        new_triples = []

        for count, triple in enumerate(self.triples):
            print("Triple number:", count)
            # head (h) and tail (t) to be corrupted
            corrupted_h = self.id2entity[triple[0]]
            corrupted_t = self.id2entity[triple[2]]
            print("corrupted_h:", corrupted_h)
            print("corrupted_t:", corrupted_t)
            # head
            if corrupted_h in self.simdict: 
                corruptors_h = list(self.simdict[corrupted_h].keys())               # List of similar entities to corrupt the head (h)
                corruptors_h_weights = list(self.simdict[corrupted_h].values())     # List of similarity values (=weights for distribution) belonging to the potential corruptors
                corruptor_h = random.choices(corruptors_h, weights=corruptors_h_weights, k=1)[0]    # Choose the actual corruptor based on weights distribution
                new_triples.append((self.entity2id[corruptor_h], triple[1], triple[2]))             # Add new triple to triples list
            # tail
            if corrupted_t in self.simdict:                                         # The same for the tail (t)
                corruptors_t = list(self.simdict[corrupted_t].keys())
                corruptors_t_weights = list(self.simdict[corrupted_t].values())
                corruptor_t = random.choices(corruptors_t, weights=corruptors_t_weights, k=1)[0]
                new_triples.append((triple[0], triple[1], self.entity2id[corruptor_t]))
            
        self.triples.extend(new_triples)
        print("New triples:\n", new_triples)
    

    @staticmethod
    def get_true_neg_samples(triples, ego_network_data, true_head, true_tail):
        true_neg_samples_head = {}
        true_neg_samples_tail = {}
        for head, relation, tail in triples:
            head_corruption_candidates = ego_network_data[str(head)]
            t_heads = true_head[(relation, tail)]

            tail_corruption_candidates = ego_network_data[str(tail)]
            t_tails = true_tail[(head, relation)]

            true_neg_samples_head[(head, relation, tail)] = [(val, key) for [val, key] in head_corruption_candidates if val not in t_heads]

            true_neg_samples_tail[(head, relation, tail)] = [(val, key) for [val, key] in tail_corruption_candidates if val not in t_tails]

        return true_neg_samples_head, true_neg_samples_tail


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        if self.ego_network_data is not None:
            if self.mode == "tail-batch":
                ego_network_candidates = self.true_neg_samples_tail[(head, relation, tail)] # copy.deepcopy(self.ego_network_data[str(tail)])
            elif self.mode == "head-batch":
                ego_network_candidates = self.true_neg_samples_head[(head, relation, tail)] # copy.deepcopy(self.ego_network_data[str(head)])

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)

            if self.ego_network_data is not None:
                if len(ego_network_candidates) > 0:
                    scores = np.asarray([x[1] for x in ego_network_candidates], 'float64')
                    scores_probs = scores / np.sum(scores)
                    if np.sum(scores_probs) != 1:
                        scores_probs[0] = scores_probs[0] + (1 - np.sum(scores_probs))
                    ents = [x[0] for x in ego_network_candidates]
                    hard_negatives_selected = np.random.choice(ents, len(ents), p=scores_probs)
                    #print('hard_negatives:', hard_negatives_selected.shape)
                    #print('unique hard_negatives:', np.unique(hard_negatives_selected).shape)
                    # print(len(hard_negatives_selected))
                    if len(negative_sample) > len(hard_negatives_selected):
                        negative_sample[:len(hard_negatives_selected)] = hard_negatives_selected
                    else:
                        negative_sample = hard_negatives_selected[:len(negative_sample)]

            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=False,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=False,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            # print(f'Number of true negatives found: {mask.sum()}', self.mode)
            # print('\n')
            # print(len(ego_network_candidates), self.mode)
            # if self.ego_network_data is not None:
            #     ego_network_candidates = self.filter_candidates(ego_network_candidates, negative_sample[~mask])
            # print(len(ego_network_candidates), self.mode)
            # print('\n')
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def filter_candidates(ego_network_candidates, false_negatives):
        # print(false_negatives)
        return [(val, key) for (val, key) in ego_network_candidates if val not in false_negatives]

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
