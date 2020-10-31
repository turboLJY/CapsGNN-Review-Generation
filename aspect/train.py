import torch
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import numpy as np
import itertools
import random
import math
import sys
import os
import pickle
import time
from tqdm import tqdm, trange

sys.path.append("..")
from model import AspectModel
from load import loadPrepareData, PAD_ID, UNK_ID, SOS_ID, EOS_ID
from util import AspectBatch2Data
from loss import margin_loss

cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################
# Training
#############################################


def adjust_learning_rate(optimizer, epoch, learning_rate, lr_decay_epoch, lr_decay_ratio):
    lr = learning_rate * (lr_decay_ratio ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def from_pretrained(embeddings, freeze=True):
    assert embeddings.dim() == 2, 'Embeddings parameter is expected to be 2-dimensional'
    rows, cols = embeddings.shape
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embeddings)
    embedding.weight.requires_grad = not freeze
    return embedding


def train(context_input, node_input, edge_input, type_input, aspect_input, aspect_output,
          aspect_model, aspect_optimizer):
    aspect_optimizer.zero_grad()

    context_input = context_input.to(device)
    node_input = node_input.to(device)
    edge_input = edge_input.to(device)
    type_input = type_input.to(device)
    aspect_input = aspect_input.to(device)
    aspect_output = aspect_output.to(device)

    # context and graph encoder
    context_embed, hidden = aspect_model.forward_context(context_input)

    graph_embed = aspect_model.forward_graph(node_input, edge_input, type_input)

    decoder_generate, decoder_hidden = aspect_model(context_embed, hidden, aspect_input, graph_embed)

    mask = torch.ne(aspect_output, PAD_ID)
    loss = margin_loss(decoder_generate, aspect_output, mask)
    loss.backward()

    clip = 5.0
    mc = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, aspect_model.parameters()), clip)

    aspect_optimizer.step()

    return loss.item()


def evaluate(context_input, node_input, edge_input, type_input, aspect_input, aspect_output, aspect_model):
    aspect_model.eval()

    context_input = context_input.to(device)
    node_input = node_input.to(device)
    edge_input = edge_input.to(device)
    type_input = type_input.to(device)
    aspect_input = aspect_input.to(device)
    aspect_output = aspect_output.to(device)

    # context and graph encoder
    context_embed, hidden = aspect_model.forward_context(context_input)

    graph_embed = aspect_model.forward_graph(node_input, edge_input, type_input)

    decoder_generate, decoder_hidden = aspect_model(context_embed, hidden, aspect_input, graph_embed)

    mask = torch.ne(aspect_output, PAD_ID)
    loss = margin_loss(decoder_generate, aspect_output, mask)

    return loss.item()


def batchify(pairs, bsz, vocabs):
    nbatch = len(pairs) // bsz + 1
    add_num = nbatch * bsz - len(pairs)
    add_sam = random.sample(pairs, add_num)
    pairs.extend(add_sam)
    assert len(pairs) % bsz == 0
    data = []
    for i in range(nbatch):
        data.append(AspectBatch2Data(vocabs, pairs[i * bsz: i * bsz + bsz]))
    return data


def trainIters(corpus, learning_rate, lr_decay_epoch, lr_decay_ratio, weight_decay, batch_size, rnn_layers, hidden_size,
               embed_size, node_size, capsule_size, epochs, gcn_layers, gcn_filters, capsule_num, save_dir, load_file=None):
    print('load data...')
    vocabs, train_pairs, valid_pairs, test_pairs = loadPrepareData(corpus, save_dir)  # load data
    print('load data finish...')

    data_path = os.path.join(save_dir, "batches")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    corpus_name = corpus
    try:
        training_batches = torch.load(os.path.join(data_path, '{}_{}.tar'.format('training_batches', batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = batchify(train_pairs, batch_size, vocabs)
        print('Complete building training pairs ...')
        torch.save(training_batches, os.path.join(data_path, '{}_{}.tar'.format('training_batches', batch_size)))

    # validation/test data
    eval_batch_size = 10
    try:
        val_batches = torch.load(os.path.join(data_path, '{}_{}.tar'.format('val_batches', eval_batch_size)))
    except FileNotFoundError:
        print('Validation pairs not found, generating ...')
        val_batches = batchify(valid_pairs, eval_batch_size, vocabs)
        print('Complete building validation pairs ...')
        torch.save(val_batches, os.path.join(data_path, '{}_{}.tar'.format('val_batches', eval_batch_size)))

    print('Building aspect model ...')
    aspect_model = AspectModel(vocabs, embed_size, node_size, hidden_size, capsule_size,
                               gcn_layers, gcn_filters, rnn_layers, capsule_num).to(device)

    print('Building optimizers ...')
    aspect_optimizer = optim.Adam(aspect_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print('Initializing hyper-parameter...\n')
    global_step = 1
    last_epoch = 1
    perplexities = []
    losses = []
    best_val_loss = None

    log_path = os.path.join('ckpt/' + corpus_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    if load_file:
        checkpoint = torch.load(load_file)
        aspect_model.load_state_dict(checkpoint['aspect_model'])
        global_step = checkpoint['global_step']
        last_epoch = checkpoint['epoch'] + 1
        perplexities = checkpoint['perplexity']
        losses = checkpoint['loss']
        for i in range(len(losses)):
            writer.add_scalar("Train/loss", losses[i], i)
            writer.add_scalar("Train/perplexity", perplexities[i], i)

    for epoch in tqdm(range(last_epoch, epochs + 1), desc="Epoch: ", leave=True):

        # train epoch
        aspect_model.train()

        tr_loss = 0
        steps = trange(len(training_batches), desc="Train Loss")
        for step in steps:
            context_input, node_input, edge_input, type_input, aspect_input, aspect_output = training_batches[step]

            loss = train(context_input, node_input, edge_input, type_input, aspect_input, aspect_output,
                         aspect_model, aspect_optimizer)

            tr_loss += loss
            global_step += 1

            losses.append(loss)
            perplexities.append(math.exp(loss))

            writer.add_scalar("Train/loss", loss, global_step)
            writer.add_scalar("Train/perplexity", math.exp(loss), global_step)

            steps.set_description("AspectModel (Loss=%g, PPL=%g)" % (round(loss, 4), round(math.exp(loss), 4)))

        cur_loss = tr_loss / len(training_batches)
        cur_ppl = math.exp(cur_loss)

        print('\nTrain | Epoch: {:3d} | Avg Loss={:4.4f} | Avg PPL={:4.4f}\n'.format(epoch, cur_loss, cur_ppl))

        # evaluate
        aspect_model.eval()
        with torch.no_grad():
            vl_loss = 0
            for val_batch in val_batches:
                context_input, node_input, edge_input, type_input, aspect_input, aspect_output = val_batch

                loss = evaluate(context_input, node_input, edge_input, type_input, aspect_input, aspect_output,
                                aspect_model)

                vl_loss += loss
            vl_loss /= len(val_batches)
            vl_ppl = math.exp(vl_loss)

        writer.add_scalar("Valid/loss", vl_loss, global_step)
        writer.add_scalar("Valid/perplexity", vl_ppl, global_step)

        print('Valid | Epoch: {:3d} | Avg Loss={:4.4f} | Avg PPL={:4.4f}\n'.format(epoch, vl_loss, vl_ppl))

        # Save the model if the validation loss is the best we've seen so far.
        model_path = os.path.join(save_dir, "model")
        if not best_val_loss or vl_loss < best_val_loss:
            directory = os.path.join(model_path, '{}_{}_{}'.format(batch_size, hidden_size, rnn_layers))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'global_step': global_step,
                'epoch': epoch,
                'aspect_model': aspect_model.state_dict(),
                'loss': losses,
                'perplexity': perplexities
            }, os.path.join(directory, '{}_{}_{}.tar'.format(epoch, round(vl_loss, 4), 'aspect_model')))
            best_val_loss = vl_loss

        if vl_loss > best_val_loss:
            print('validation loss is larger than best validation loss. Break!')
            break

        # learning rate adjust
        adjust_learning_rate(aspect_optimizer, epoch - last_epoch + 1, learning_rate, lr_decay_epoch, lr_decay_ratio)
