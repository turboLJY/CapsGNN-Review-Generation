import torch
import random
from model import *
from util import *
import sys
import os
from loss import *
import itertools
import random
import math
from tqdm import tqdm
from collections import deque
import pickle
import logging

sys.path.append("..")
from model import AspectModel, ReviewModel
from load import loadPrepareData, PAD_ID, UNK_ID, SOS_ID, EOS_ID, EXTEND_SIZE, WINDOW_SIZE

MAX_TOPIC = 10
MIN_TOPIC = 3

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Hypothesis(object):
    def __init__(self, tokens, his_tokens, probs, state):
        self.tokens = tokens
        self.his_tokens = deque(his_tokens)
        self.probs = probs
        self.state = state

    def extend(self, token, his_token, prob, state):
        if his_token:
            self.his_tokens.popleft()
            self.his_tokens.append(his_token)
        return Hypothesis(tokens=self.tokens + [token],
                          his_tokens=self.his_tokens,
                          probs=self.probs + [prob],
                          state=state)

    @property
    def history_tokens(self):
        return list(self.his_tokens)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def prob(self):
        return sum(self.probs)

    @property
    def avg_prob(self):
        return self.prob / len(self.tokens)


def sort_hyps(hyps):
    return sorted(hyps, key=lambda h: h.prob, reverse=True)


def beam_decode(aspect_model, review_model, aspect_hidden, review_hidden, aspect_context_embed, graph_embed,
                review_context_embed, vocabs, beam_size, max_length, min_length):
    aspect_input = torch.LongTensor([[SOS_ID]]).to(device)

    decoded_aspects = []
    decoded_reviews = []

    for ai in range(MAX_TOPIC):
        aspect_output, aspect_hidden = aspect_model(aspect_context_embed, aspect_hidden, aspect_input, graph_embed)
        topv, topi = aspect_output.data.topk(5)
        topi = topi.squeeze(0)
        nti = topi[0][0].item()
        if nti == EOS_ID:
            decoded_aspects.append("<eos>")
            break
        else:
            decoded_aspects.append(vocabs.idx2aspect[nti])
            aspect_input = torch.LongTensor([[nti]]).to(device)

            aspect = torch.LongTensor([[nti]]).to(device)

            hyps = [Hypothesis(tokens=[SOS_ID], his_tokens=[PAD_ID] * WINDOW_SIZE,
                               probs=[0.0], state=review_hidden) for _ in range(beam_size)]
            results = []
            steps = 0
            while steps < max_length and len(results) < beam_size:
                new_hiddens = []
                topk_ids = []
                topk_probs = []
                for hyp in hyps:
                    review_input = torch.LongTensor([[hyp.latest_token]]).to(device)
                    review_hidden = hyp.state

                    extend = set()
                    for wid in hyp.history_tokens:
                        if vocabs.idx2token[wid] in vocabs.node2neighbor:
                            for nei in vocabs.token2neighbor[vocabs.idx2token[wid]]:
                                if nei in vocabs.token2idx:
                                    extend.add(vocabs.token2idx[nei])
                    extend = list(extend)
                    if len(extend) <= EXTEND_SIZE:
                        extend = extend + [PAD_ID] * (EXTEND_SIZE - len(extend))
                    else:
                        extend = random.sample(extend, EXTEND_SIZE)
                    extend_input = torch.LongTensor([[extend]]).to(device)

                    review_output, review_hidden = review_model(review_context_embed, review_hidden, review_input,
                                                                aspect, extend_input)
                    new_hiddens.append(review_hidden)

                    topv, topi = review_output.data.topk(beam_size)
                    topv = topv.squeeze(0)
                    topi = topi.squeeze(0)

                    topk_ids.extend(topi)
                    topk_probs.extend(topv)

                all_hyps = []
                num_orig_hyps = 1 if steps == 0 else len(hyps)
                for i in range(num_orig_hyps):
                    hyp, new_hidden = hyps[i], new_hiddens[i]
                    for j in range(beam_size):
                        # adding to history sequence or not
                        ht = None
                        if vocabs.idx2token[topk_ids[i][j].item()] not in ["<sos>", "<eos>", "<unk>", ",", ".", "!", "?"]:
                            ht = topk_ids[i][j].item()
                        new_hyp = hyp.extend(token=topk_ids[i][j], his_token=ht,
                                             prob=topk_probs[i][j], state=new_hidden)
                        all_hyps.append(new_hyp)

                hyps = []
                for h in sort_hyps(all_hyps):
                    if h.latest_token == EOS_ID:
                        if steps >= min_length:
                            results.append(h)
                    else:
                        hyps.append(h)

                    if len(hyps) == beam_size or len(results) == beam_size:
                        break

                steps += 1

            if len(results) == 0:
                results = hyps

            hyps_sorted = sort_hyps(results)
            best_hyp = hyps_sorted[0]

            sentence_tokens = best_hyp.tokens
            if sentence_tokens[-1] != EOS_ID:
                sentence_tokens.append(EOS_ID)
            for t in sentence_tokens[1:]:
                if vocabs.idx2token[int(t)] in vocabs.node2name:
                    name = vocabs.node2name[vocabs.idx2token[int(t)]]
                    decoded_reviews.append(name)
                else:
                    decoded_reviews.append(vocabs.idx2token[int(t)])

    return decoded_aspects, decoded_reviews


def decode(aspect_model, review_model, aspect_hidden, review_hidden, aspect_context_embed, graph_embed,
           review_context_embed, vocabs, max_length, min_length):
    aspect_input = torch.LongTensor([[SOS_ID]]).to(device)

    history = deque([PAD_ID] * WINDOW_SIZE)
    extend = [PAD_ID] * EXTEND_SIZE

    decoded_aspects = []
    decoded_reviews = []

    for ai in range(MAX_TOPIC):
        aspect_output, aspect_hidden = aspect_model(aspect_context_embed, aspect_hidden, aspect_input, graph_embed)
        topv, topi = aspect_output.data.topk(4)
        topi = topi.squeeze(0)
        nti = topi[0][0]
        if nti == EOS_ID:
            decoded_aspects.append("<eos>")
            break
        else:
            decoded_aspects.append(vocabs.idx2aspect[nti])
            aspect_input = torch.LongTensor([[nti]]).to(device)

            aspect = torch.LongTensor([[nti]]).to(device)
            review_input = torch.LongTensor([[SOS_ID]]).to(device)
            for si in range(max_length):
                extend_input = torch.LongTensor([[extend]]).to(device)
                review_output, review_hidden = review_model(review_context_embed, review_hidden, review_input,
                                                            aspect, extend_input)
                topv, topi = review_output.data.topk(4)
                topi = topi.squeeze(0)
                npi = topi[0][0]
                if npi == EOS_ID:
                    decoded_reviews.append("<eos>")
                    break
                else:
                    if vocabs.idx2token[t] in vocabs.node2name:
                        decoded_reviews.append(vocabs.node2name[vocabs.idx2token[t]])
                    else:
                        decoded_reviews.append(vocabs.idx2token[t])
                    review_input = torch.LongTensor([[npi]]).to(device)
                    # judge word
                    if vocabs.idx2token[npi] not in ["<sos>", "<eos>", "<unk>", ",", ".", "!", "?"]:
                        history.popleft()
                        history.append(npi)
                        ext = set()
                        for wid in list(history):
                            if vocabs.idx2token[wid] in vocabs.node2neighbor:
                                for nei in vocabs.node2neighbor[vocabs.idx2token[wid]]:
                                    if nei in vocabs.token2idx:
                                        ext.add(vocabs.token2idx[nei])
                        ext = list(ext)
                        if len(ext) <= EXTEND_SIZE:
                            ext = ext + [PAD_ID] * (EXTEND_SIZE - len(ext))
                        else:
                            ext = random.sample(ext, EXTEND_SIZE)
                        extend = ext

            if decoded_reviews[-1] != "<eos>":
                decoded_reviews.append("<eos>")

    if decoded_aspects[-1] != "<eos>":
        decoded_aspects.append("<eos>")

    return decoded_aspects, decoded_reviews


def evaluate(aspect_model, review_model, vocabs, context, nodes, edges, types, beam_size, max_length, min_length):
    context_input = InputVar([context]).to(device)
    node_input, edge_input, type_input = GraphVar([nodes], [edges], [types], vocabs)
    node_input = node_input.to(device)
    edge_input = edge_input.to(device)
    type_input = type_input.to(device)

    # attribute encoder
    aspect_context_embed, aspect_hidden = aspect_model.forward_context(context_input)
    graph_embed = aspect_model.forward_graph(node_input, edge_input, type_input)

    review_context_embed, review_hidden = review_model.forward_context(context_input)

    if beam_size == 1:
        return decode(aspect_model, review_model, aspect_hidden, review_hidden,
                      aspect_context_embed, graph_embed, review_context_embed, vocabs, max_length, min_length)
    else:
        return beam_decode(aspect_model, review_model, aspect_hidden, review_hidden,
                           aspect_context_embed, graph_embed, review_context_embed, vocabs, beam_size, max_length,
                           min_length)


def evaluateRandomly(aspect_model, review_model, vocabs, pairs, n_pairs, beam_size, max_length, min_length, save_dir):
    path = os.path.join(save_dir, 'decode')
    if not os.path.exists(path):
        os.makedirs(path)
    f1 = open(path + "/decoded.txt", 'w')
    for i in range(n_pairs):

        pair = pairs[i]

        user = pair[0][0]
        item = pair[0][1]
        rating = pair[0][2]
        aspects = " ".join(pair[1][1:-1])
        reviews = []
        for review in pair[2]:
            words = []
            for wd in review[1:-1]:
                if wd in vocabs.node2name:
                    words.append(vocabs.node2name[wd])
                else:
                    words.append(wd)
            reviews.append(" ".join(words))
        reviews = "||".join(reviews)
        print("=============================================================")
        print('Attribute > ',
              '\t'.join([vocabs.idx2context[user], vocabs.idx2context[item], vocabs.idx2context[rating]]))
        print('Aspect > ', aspects)
        print('Review > ', reviews)

        f1.write('Attribute: ' + '\t'.join(
            [vocabs.idx2context[user], vocabs.idx2context[item], vocabs.idx2context[rating]]) + '\n'
                 + 'Aspects: ' + aspects + '\n'
                 + 'Reviews: ' + reviews + '\n')
        if beam_size >= 1:
            output_aspects, output_reviews = evaluate(aspect_model, review_model,
                                                      vocabs, pair[0], pair[3], pair[4], pair[5],
                                                      beam_size, max_length, min_length)
            aspect_sentence = ' '.join(output_aspects[:-1])
            review_words = []
            for wd in output_reviews:
                if wd == "<eos>":
                    review_words.append("||")
                else:
                    review_words.append(wd)
            review_sentence = ' '.join(review_words[:-1])
            print('Generation aspect < ', aspect_sentence)
            print('Generation review < ', review_sentence)
            f1.write('Generation aspect: ' + aspect_sentence + "\n")
            f1.write('Generation review: ' + review_sentence + "\n")
    f1.close()


def runTest(corpus, rnn_layers, hidden_size, embed_size, node_size, capsule_size, gcn_layers, gcn_filters, capsule_num,
            saved_aspect_model, saved_review_model, beam_size, max_length, min_length, save_dir):
    vocabs, train_pairs, valid_pairs, test_pairs = loadPrepareData(corpus, save_dir)

    print('Building aspect model ...')
    aspect_model = AspectModel(vocabs, embed_size, node_size, hidden_size, capsule_size,
                               gcn_layers, gcn_filters, rnn_layers, capsule_num).to(device)

    print('Building review model ...')
    review_model = ReviewModel(vocabs, embed_size, node_size, hidden_size, rnn_layers).to(device)

    checkpoint = torch.load(saved_aspect_model)
    aspect_model.load_state_dict(checkpoint['aspect_model'])

    checkpoint = torch.load(saved_review_model)
    review_model.load_state_dict(checkpoint['review_model'])

    # train mode set to false, effect only on dropout, batchNorm
    aspect_model.train(False)
    review_model.train(False)

    evaluateRandomly(aspect_model, review_model, vocabs, test_pairs, len(test_pairs), beam_size,
                     max_length, min_length, save_dir)
