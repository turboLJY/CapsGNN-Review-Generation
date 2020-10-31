import torch
import re
import os
import unicodedata
import pickle

# depends on the word_vocab file
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
EXTEND_SIZE = 50
WINDOW_SIZE = 5


class Vocabulary:
    def __init__(self, name, save_dir):
        self.name = name
        # user, item (rating or not)
        with open(os.path.join(save_dir, 'context.pkl'), 'rb') as fp:
            self.context2idx = pickle.load(fp)
        with open(os.path.join(save_dir, 'context_rev.pkl'), 'rb') as fp:
            self.idx2context = pickle.load(fp)
        self.n_context = len(self.context2idx)

        # aspect
        with open(os.path.join(save_dir, 'aspect.pkl'), 'rb') as fp:
            self.aspect2idx = pickle.load(fp)
        with open(os.path.join(save_dir, 'aspect_rev.pkl'), 'rb') as fp:
            self.idx2aspect = pickle.load(fp)
        self.n_aspects = len(self.aspect2idx)

        # entity, words
        with open(os.path.join(save_dir, 'token.pkl'), 'rb') as fp:
            self.token2idx = pickle.load(fp)
        with open(os.path.join(save_dir, 'token_rev.pkl'), 'rb') as fp:
            self.idx2token = pickle.load(fp)
        self.n_tokens = len(self.token2idx)

        # node
        with open(os.path.join(save_dir, 'node.pkl'), 'rb') as fp:
            self.node2idx = pickle.load(fp)
        with open(os.path.join(save_dir, 'node_rev.pkl'), 'rb') as fp:
            self.idx2node = pickle.load(fp)
        self.n_nodes = len(self.node2idx)

        # relation
        with open(os.path.join(save_dir, 'relation.pkl'), 'rb') as fp:
            self.relation2idx = pickle.load(fp)
        with open(os.path.join(save_dir, 'relation_rev.pkl'), 'rb') as fp:
            self.idx2relation = pickle.load(fp)
        self.n_relations = len(self.relation2idx)

        # neighbors of each node
        with open(os.path.join(save_dir, 'node_2_neighbor.pkl'), 'rb') as fp:
            self.node2neighbor = pickle.load(fp)

        # names of each node
        with open(os.path.join(save_dir, 'node_2_name.pkl'), 'rb') as fp:
            self.node2name = pickle.load(fp)

        # user item graph
        with open(os.path.join(save_dir, 'user_item_graph.pkl'), 'rb') as fp:
            self.user_item2graph = pickle.load(fp)


def tokenize(path, vocabs):
    print("Reading {}".format(path))
    pairs = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data = eval(line)
            user = data['reviewerID']
            item = data['productID']
            rating = data['overall']
            
            user_id = vocabs.context2idx[user]
            item_id = vocabs.context2idx[item]
            rating_id = vocabs.context2idx[rating]

            graph = vocabs.user_item2graph[(user, item)]

            nodes = graph["Nodes"]
            edges = graph["Edges"]
            types = graph["Edge_types"]
            
            aspect = ["<sos>"] + data['aspect'].split() + ["<eos>"]

            review = data["review"].split("||")
            review = [["<sos>"] + sen.split() + ["<eos>"] for sen in review]
            
            context = [user_id, item_id, rating_id]
            pair = [context, aspect, review, nodes, edges, types]
            pairs.append(pair)
    return pairs


def prepareData(vocabs, save_dir):
    train_pairs = tokenize(os.path.join(save_dir, 'train_tok.json'), vocabs)
    valid_pairs = tokenize(os.path.join(save_dir, 'valid_tok.json'), vocabs)
    test_pairs = tokenize(os.path.join(save_dir, 'test_tok.json'), vocabs)

    torch.save(train_pairs, os.path.join(save_dir, '{!s}.tar'.format('train_pairs')))
    torch.save(valid_pairs, os.path.join(save_dir, '{!s}.tar'.format('valid_pairs')))
    torch.save(test_pairs, os.path.join(save_dir, '{!s}.tar'.format('test_pairs')))
    return train_pairs, valid_pairs, test_pairs


def loadPrepareData(corpus_name, save_dir):
    try:
        print("Start loading training data ...")
        vocabs = Vocabulary(corpus_name, save_dir)
        train_pairs = torch.load(os.path.join(save_dir, 'train_pairs.tar'))
        valid_pairs = torch.load(os.path.join(save_dir, 'valid_pairs.tar'))
        test_pairs = torch.load(os.path.join(save_dir, 'test_pairs.tar'))
        
    except FileNotFoundError:
        print("Saved data not found, start preparing training data ...")
        vocabs = Vocabulary(corpus_name, save_dir)
        train_pairs, valid_pairs, test_pairs = prepareData(vocabs, save_dir)
    return vocabs, train_pairs, valid_pairs, test_pairs


