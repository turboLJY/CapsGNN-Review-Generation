import torch
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
import itertools
import random
import math
import sys
import os
from collections import deque
from tqdm import tqdm
from load import loadPrepareData, PAD_ID, UNK_ID, SOS_ID, EOS_ID, EXTEND_SIZE, WINDOW_SIZE
import pickle
import logging
logging.basicConfig(level=logging.INFO)
    
#############################################
# Prepare Training Data
#############################################


def indicesFromAspect(vocabs, sentence):
    ids = []
    for word in sentence:
        word = word
        ids.append(vocabs.aspect2idx[word])
    return ids


def indicesFromReview(vocabs, sentence):
    ids = []
    for word in sentence:
        word = word
        if word in vocabs.token2idx:
            ids.append(vocabs.token2idx[word])
        else:
            ids.append(UNK_ID)
    return ids


def returnHistory(vocabs, sentence, window_size=WINDOW_SIZE):
    ban = ["<sos>", "<eos>", "<unk>", ",", ".", "!", "?"]
    his_sequence = deque([PAD_ID] * window_size)
    history = []
    for wid in sentence:
        if vocabs.idx2token[wid] not in ban:
            his_sequence.popleft()
            his_sequence.append(wid)
            history.append(list(his_sequence))
        else:
            history.append(list(his_sequence))
    return history


def returnExtend(vocabs, history, extend_size=EXTEND_SIZE):
    ext_inp = []
    # adding extend entity and word
    for his in history:
        inp = set()
        for wid in his:
            if vocabs.idx2token[wid] in vocabs.node2neighbor:
                for nei in vocabs.node2neighbor[vocabs.idx2token[wid]]:
                    if nei in vocabs.token2idx:
                        inp.add(vocabs.token2idx[nei])
        inp = list(inp)
        if len(inp) <= extend_size:
            inp = inp + [PAD_ID] * (extend_size - len(inp))
        else:
            inp = random.sample(inp, extend_size)
        ext_inp.append(inp)
    return ext_inp


def sequencePadding(data, fillvalue=PAD_ID):
    max_len = max([len(d) for d in data])
    new_seq = []
    for d in data:
        new_seq.append(d + [fillvalue] * (max_len - len(d)))
    return new_seq


def extendPadding(data, filllen=EXTEND_SIZE, fillvalue=PAD_ID):
    max_len = max([len(d) for d in data])
    new_seq = []
    for d in data:
        pad = [[fillvalue] * filllen] * (max_len - len(d))
        new_seq.append(d + pad)
    return new_seq


def InputVar(data):
    context = [[d[0], d[1], d[2]] for d in data]
    contextVar = torch.LongTensor(context)
    return contextVar


def GraphVar(node_data, edge_data, type_data, vocabs):
    max_node = max([len(d) for d in node_data])

    node_input = []
    for nodes in node_data:
        node_input.append(nodes + [nodes[0]] * (max_node - len(nodes)))

    max_edge = max([len(d[0]) for d in edge_data])

    edge_input = []
    type_input = []
    for bi in range(len(edge_data)):
        head = edge_data[bi][0] + [0] * (max_edge - len(edge_data[bi][0]))
        tail = edge_data[bi][1] + [1] * (max_edge - len(edge_data[bi][1]))
        edge_input.append([head, tail])
        type_input.append(type_data[bi] + [vocabs.relation2idx["<interact>"]] * (max_edge - len(type_data[bi])))

    nodeVar = torch.LongTensor(node_input)
    edgeVar = torch.LongTensor(edge_input)
    typeVar = torch.LongTensor(type_input)

    return nodeVar, edgeVar, typeVar


def AspectOutputVar(l, vocabs):
    aspect_input = [indicesFromAspect(vocabs, sentence[:-1]) for sentence in l]
    aspect_output = [indicesFromAspect(vocabs, sentence[1:]) for sentence in l]
    inpadList = sequencePadding(aspect_input)
    outpadList = sequencePadding(aspect_output)
    inpadVar = torch.LongTensor(inpadList)
    outpadVar = torch.LongTensor(outpadList)

    return inpadVar, outpadVar


def ReviewOutputVar(asp_data, rev_data, vocabs):
    aspect = [indicesFromAspect(vocabs, sentence) for sentence in asp_data]

    aspect_input = []
    review_input = []
    review_output = []
    extend_input = []
    for bi in range(len(asp_data)):
        asp = []
        inp = []
        out = []
        for sj in range(len(asp_data[bi])):
            sentence = rev_data[bi][sj]
            asp += [aspect[bi][sj]] * len(sentence[:-1])
            inp += indicesFromReview(vocabs, sentence[:-1])
            out += indicesFromReview(vocabs, sentence[1:])
        aspect_input.append(asp)
        review_input.append(inp)
        review_output.append(out)

        his = returnHistory(vocabs, inp)
        ext_inp = returnExtend(vocabs, his)
        extend_input.append(ext_inp)  # seq_len x extend_size

    aspectList = sequencePadding(aspect_input)
    inpadList = sequencePadding(review_input)
    outpadList = sequencePadding(review_output)
    extendList = extendPadding(extend_input)

    aspectVar = torch.LongTensor(aspectList)
    inpadVar = torch.LongTensor(inpadList)
    outpadVar = torch.LongTensor(outpadList)
    extendVar = torch.LongTensor(extendList)

    return aspectVar, inpadVar, outpadVar, extendVar


def AspectBatch2Data(vocabs, pair_batch):
    pair_batch.sort(key=lambda x: len(x[1]), reverse=True)
    context_batch, node_batch, edge_batch, type_batch, aspect_batch = [], [], [], [], []
    for i in range(len(pair_batch)):
        context_batch.append(pair_batch[i][0])
        aspect_batch.append(pair_batch[i][1])
        node_batch.append(pair_batch[i][3])
        edge_batch.append(pair_batch[i][4])
        type_batch.append(pair_batch[i][5])
    context_input = InputVar(context_batch)
    node_input, edge_input, type_input = GraphVar(node_batch, edge_batch, type_batch, vocabs)
    aspect_input, aspect_output = AspectOutputVar(aspect_batch, vocabs)
    return context_input, node_input, edge_input, type_input, aspect_input, aspect_output


def ReviewBatch2Data(vocabs, pair_batch):
    pair_batch.sort(key=lambda x: len(x[1]), reverse=True)
    context_batch, aspect_batch, review_batch = [], [], []
    for i in range(len(pair_batch)):
        context_batch.append(pair_batch[i][0])
        aspect_batch.append(pair_batch[i][1][1:-1])  # remove <sos> and <eos>
        review_batch.append(pair_batch[i][2])
    context_input = InputVar(context_batch)
    aspect_input, review_input, review_output, extend_input = ReviewOutputVar(aspect_batch, review_batch, vocabs)
    return context_input, aspect_input, review_input, review_output, extend_input

