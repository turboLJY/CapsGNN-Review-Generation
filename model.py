# coding=utf-8
import torch
from torch import nn
from load import PAD_ID, UNK_ID, SOS_ID, EOS_ID
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv
import math
import numpy as np
import torch.backends.cudnn as cudnn


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """

    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class PrimaryCapsuleLayer(torch.nn.Module):
    def __init__(self, in_units, in_channels, num_units, capsule_size):
        super(PrimaryCapsuleLayer, self).__init__()
        self.num_units = num_units
        self.units = []
        for i in range(self.num_units):
            unit = torch.nn.Conv1d(in_channels=in_channels,  # gcn_filters
                                   out_channels=capsule_size,  # capsule_size
                                   kernel_size=(in_units, 1),  # (gcn_layers, 1)
                                   stride=1,
                                   bias=True)

            self.add_module("unit_" + str(i), unit)
            self.units.append(unit)

    @staticmethod
    def squash(s):
        """
        Squash activations.
        :param s: Signal.
        :return s: Activated signal.
        """
        mag_sq = torch.sum(s ** 2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        """
        Forward propagation pass.
        :param x: input features, shape [batch_size, num_node, gcn_layers, gcn_filters]
        :return : primary capsule features.
        """
        u = [self.units[i](x) for i in range(self.num_units)]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_units, -1)  # [batch_size, gcn_layers, num_node * capsule_size]
        return PrimaryCapsuleLayer.squash(u)


class SecondaryCapsuleLayer(torch.nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size):
        super(SecondaryCapsuleLayer, self).__init__()
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units

        # [1, capsule_size, capsule_num, gcn_layers]
        self.W = torch.nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))

    @staticmethod
    def squash(s):
        """
        Squash activations.
        :param s: Signal.
        :return s: Activated signal.
        """
        mag_sq = torch.sum(s ** 2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        """
        Forward propagation pass.
        :param x: input features, shape [num_node, gcn_layers, capsule_size]
        :return : Capsule output.
        """
        num_node = x.size(0)
        x = x.transpose(1, 2)  # [num_node, capsule_size, gcn_layers]
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)  # [N, capsule_size, capsule_num, gcn_layers, 1]
        W = torch.cat([self.W] * num_node, dim=0)  # [N, capsule_size, capsule_num, gcn_layers]
        u_hat = torch.matmul(W, x)  # [N, capsule_size, capsule_num, 1]
        b_ij = torch.zeros(1, self.in_channels, self.num_units, 1).to(u_hat.device)  # [1, capsule_size, capsule_num, 1]

        num_iterations = 3
        for _ in range(num_iterations):
            c_ij = torch.nn.functional.softmax(b_ij, dim=2)
            c_ij = torch.cat([c_ij] * num_node, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = SecondaryCapsuleLayer.squash(s_j)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)


class CapsuleNet(torch.nn.Module):
    def __init__(self, vocabs, capsule_size, hidden_size, gcn_layers, gcn_filters, capsule_num):
        super(CapsuleNet, self).__init__()
        self.vocabs = vocabs
        self.capsule_size = capsule_size
        self.hidden_size = hidden_size
        self.gcn_layers = gcn_layers
        self.gcn_filters = gcn_filters
        self.capsule_num = capsule_num

        self.attention_1 = nn.Linear(self.hidden_size + self.capsule_size, 1, bias=False)

        self.primary_capsule_layer = PrimaryCapsuleLayer(in_units=self.gcn_filters,
                                                         in_channels=self.gcn_layers,
                                                         num_units=self.gcn_layers,
                                                         capsule_size=self.capsule_size)
        self.graph_capsule_layer = SecondaryCapsuleLayer(self.gcn_layers,
                                                         self.capsule_size,
                                                         self.capsule_num,
                                                         self.capsule_size)
        self.aspect_capsule_layer = SecondaryCapsuleLayer(self.capsule_size,
                                                          self.capsule_num,
                                                          self.vocabs.n_aspects,
                                                          self.capsule_size)

    def adaptive_attention(self, x_in, hidden):
        """
        :param x_in: graph capsule output, shape [capsule_num, capsule_size]
        :param hidden: hidden representation from rnn, shape [seq_len, hidden_size]
        """
        seq_len = hidden.size(0)
        x_in = x_in.unsqueeze(0).expand(seq_len, -1, -1)
        hidden = hidden.unsqueeze(1).expand(-1, self.capsule_num, -1)

        # [seq_len, capsule_num, 1]
        attention_score_base = self.attention_1(torch.cat((x_in, hidden), dim=-1))
        attention_score = torch.nn.functional.softmax(attention_score_base, dim=1)
        condensed_x = x_in * attention_score
        return condensed_x

    def forward(self, graph_embed, hidden):
        """
        :param graph_embed: shape [batch_size, gcn_layers, gcn_filters, num_node] == primary capsules
        :param hidden: shape [batch_size, seq_len, hidden_size]
        :return:
        """
        aspect_capsules = []
        batch_size = graph_embed.size(0)
        for bi in range(batch_size):
            # [gcn_layers, num_node * capsule_size] -> [num_node, gcn_layers * capsule_size]
            primary_capsule_output = self.primary_capsule_layer(graph_embed[bi, :, :, :].unsqueeze(0))
            primary_capsule_output = primary_capsule_output.view(-1, self.gcn_layers, self.capsule_size)

            # [num_node, gcn_layers * capsule_size] -> [capsule_num, capsule_size]
            graph_capsule_output = self.graph_capsule_layer(primary_capsule_output)
            graph_capsule_output = graph_capsule_output.view(-1, self.capsule_size, self.capsule_num)
            graph_capsule_output = torch.mean(graph_capsule_output, dim=0).view(self.capsule_num, self.capsule_size)

            # [capsule_num, capsule_size] -> [seq_len, capsule_num, capsule_size]
            adaptive_capsule_output = self.adaptive_attention(graph_capsule_output, hidden[bi, :, :])
            adaptive_capsule_output = adaptive_capsule_output.transpose(1, 2).contiguous()

            # [seq_len, capsule_num, capsule_size] -> [seq_len, n_aspects, capsule_size]
            aspect_capsule_output = self.aspect_capsule_layer(adaptive_capsule_output)
            aspect_capsule_output = aspect_capsule_output.view(-1, self.vocabs.n_aspects, self.capsule_size)

            aspect_capsules.append(aspect_capsule_output)
        aspect_capsules = torch.stack(aspect_capsules, dim=0)

        return aspect_capsules


class ContextEncoder(nn.Module):
    def __init__(self, vocabs, embed_size, hidden_size, rnn_layers):
        super(ContextEncoder, self).__init__()
        self.vocabs = vocabs
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers

        self.context_embedding = nn.Embedding(self.vocabs.n_context, self.embed_size)

        self.trans_linear = nn.Linear(self.embed_size * 3, self.rnn_layers * self.hidden_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.context_embedding.weight.data.uniform_(-initrange, initrange)

        self.trans_linear.bias.data.fill_(0)
        self.trans_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, context):
        """
        :param context: (batch, 3)
        """
        # (batch_size, 3, embed_size) -> (batch_size, rnn_layers * hidden_size)
        context_embed = self.context_embedding(context)
        hidden = torch.tanh(self.trans_linear(context_embed.view(-1, self.embed_size * 3)))

        # (rnn_layers, batch_size, hidden_size)
        hidden = hidden.view(-1, self.rnn_layers, self.hidden_size).transpose(0, 1).contiguous()

        return context_embed, hidden


class GraphEncoder(nn.Module):
    def __init__(self, vocabs, node_size, gcn_layers, gcn_filters):
        super(GraphEncoder, self).__init__()
        self.vocabs = vocabs
        self.node_size = node_size
        self.gcn_layers = gcn_layers
        self.gcn_filters = gcn_filters

        self.node_embedding = nn.Embedding(self.vocabs.n_nodes, self.node_size)

        self.gcn = [GCNConv(self.node_size, self.gcn_filters)]
        for layer in range(self.gcn_layers - 1):
            self.gcn.append(GCNConv(self.gcn_filters, self.gcn_filters))
        self.gcn = ListModule(*self.gcn)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.node_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, graph_nodes, graph_edges, edge_types):
        """
        :param graph_nodes: user graph nodes, in shape (batch_size, node_num)
        :param graph_edges: user graph edges, in shape (batch_size, 2, edge_num)
        :param edge_types: user graph edge types, in shape (batch_size, edge_num)
        """
        batch_size = graph_nodes.size(0)

        # (batch_size, node_num, gcn_filters)
        graph_embed = []
        for bi in range(batch_size):
            graph_feature = []
            feature = self.node_embedding(graph_nodes[bi, :])  # [node_num, node_size]
            edge = graph_edges[bi, :, :]
            for layer in self.gcn:
                feature = torch.nn.functional.relu(layer(feature, edge))  # [num_node, gcn_filters]
                graph_feature.append(feature)
            graph_feature = torch.stack(graph_feature, dim=0).transpose(1, 2).contiguous()
            graph_embed.append(graph_feature)
        graph_embed = torch.stack(graph_embed, 0)  # [batch_size, gcn_layers, gcn_filters, num_node]

        return graph_embed


class AspectModel(nn.Module):
    def __init__(self, vocabs, embed_size, node_size, hidden_size, capsule_size, gcn_layers, gcn_filters,
                 rnn_layers, capsule_num, dropout=0.2):
        super(AspectModel, self).__init__()
        self.vocabs = vocabs
        self.embed_size = embed_size
        self.node_size = node_size
        self.hidden_size = hidden_size
        self.capsule_size = capsule_size
        self.gcn_layers = gcn_layers
        self.gcn_filters = gcn_filters
        self.rnn_layers = rnn_layers
        self.capsule_num = capsule_num
        self.dropout = dropout

        # aspect embedding
        self.aspect_embedding = nn.Embedding(self.vocabs.n_aspects, self.embed_size, padding_idx=PAD_ID)

        self.context_encoder = ContextEncoder(self.vocabs, self.embed_size, self.hidden_size, self.rnn_layers)
        self.graph_encoder = GraphEncoder(self.vocabs, self.node_size, self.gcn_layers, self.gcn_filters)
        self.capsulenet = CapsuleNet(self.vocabs, self.capsule_size, self.hidden_size, self.gcn_layers,
                                     self.gcn_filters, self.capsule_num)
        self.decoder = nn.GRU(self.embed_size, self.hidden_size, self.rnn_layers, batch_first=True,
                              dropout=self.dropout)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.context_linear = nn.Linear(self.embed_size + self.hidden_size, 1, bias=False)
        self.global_linear = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size)
        self.pre_vocab_linear = nn.Linear(self.embed_size + self.node_size + self.hidden_size, self.hidden_size)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocabs.n_aspects)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.aspect_embedding.weight.data.uniform_(-initrange, initrange)

        self.context_linear.weight.data.uniform_(-initrange, initrange)

        self.global_linear.bias.data.fill_(0)
        self.global_linear.weight.data.uniform_(-initrange, initrange)

        self.pre_vocab_linear.bias.data.fill_(0)
        self.pre_vocab_linear.weight.data.uniform_(-initrange, initrange)

        self.vocab_linear.bias.data.fill_(0)
        self.vocab_linear.weight.data.uniform_(-initrange, initrange)

    def context_attention(self, output, context_embed):
        """
        args:
            output: shape [batch_size, seq_len, hidden_size]
            context_embed: shape [batch_size, 3, embed_size]
        outputs:
            context_repre: shape [batch_size, seq_len, node_size]
        """
        n_context = context_embed.size(1)
        seq_len = output.size(1)

        output = output.unsqueeze(2).expand(-1, -1, n_context, -1)
        context = context_embed.unsqueeze(1).expand(-1, seq_len, -1, -1)

        # [batch_size, seq_len, n_context]
        energies = self.context_linear(torch.cat([output, context], dim=-1)).squeeze(-1)

        context_probs = F.softmax(energies, dim=-1)  # [batch_size, seq_len, n_context]

        context_repre = torch.matmul(context_probs, context_embed)  # [batch_size, seq_len, embed_size]

        return context_repre

    def forward_context(self, context):
        context_embed, hidden = self.context_encoder(context)
        return context_embed, hidden

    def forward_graph(self, graph_nodes, graph_edges, edge_types):
        graph_embed = self.graph_encoder(graph_nodes, graph_edges, edge_types)
        return graph_embed

    def forward(self, context_embed, last_hidden, input_seq, graph_embed):
        """
        :param context_embed:
            context outputs from encoder in shape (batch_size, 3, node_size)
        :param last_hidden:
            tuple, last hidden stat of the decoder, in shape (n_layers, batch_size, hidden_size)
        :param input_seq:
            aspect input for all time steps, in shape (batch_size, seq_len)
        :param graph_embed:
            graph outputs from encoder in shape (batch_size, num_node, node_size)
        """
        aspect_embed = self.dropout_layer(self.aspect_embedding(input_seq))

        # rnn_output: [batch_size, seq_len, hidden_size]
        # hidden: [n_layers, batch_size, hidden_size]
        rnn_output, hidden = self.decoder(aspect_embed, last_hidden)

        # [batch_size, seq_len, embed_size]
        context_repre = self.context_attention(rnn_output, context_embed)

        # [batch_size, seq_len, hidden_size]
        global_repre = self.global_linear(torch.cat((rnn_output, context_repre), 2))

        # [batch_size, seq_len, n_aspects, capsule_size]
        aspect_capsules = self.capsulenet(graph_embed, global_repre)

        return aspect_capsules, hidden


class ReviewModel(nn.Module):
    def __init__(self, vocabs, embed_size, node_size, hidden_size, rnn_layers, dropout=0.2):
        super(ReviewModel, self).__init__()
        self.vocabs = vocabs
        self.embed_size = embed_size
        self.node_size = node_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout

        self.aspect_embedding = nn.Embedding(self.vocabs.n_aspects, self.embed_size, padding_idx=PAD_ID)
        self.token_embedding = nn.Embedding(self.vocabs.n_tokens, self.embed_size, padding_idx=PAD_ID)

        self.context_encoder = ContextEncoder(self.vocabs, self.embed_size, self.hidden_size, self.rnn_layers)
        self.decoder = nn.GRU(self.embed_size, self.hidden_size, self.rnn_layers, batch_first=True,
                              dropout=dropout)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.context_linear = nn.Linear(self.embed_size + self.hidden_size, 1, bias=False)
        self.extend_linear = nn.Linear(2 * self.embed_size + self.hidden_size, 1, bias=False)
        self.gate_linear = nn.Linear(self.embed_size + self.hidden_size, 1, bias=False)
        self.pre_vocab_linear = nn.Linear(self.embed_size + self.hidden_size, self.hidden_size)
        self.vocab_linear = nn.Linear(self.hidden_size, self.vocabs.n_tokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.aspect_embedding.weight.data.uniform_(-initrange, initrange)
        self.token_embedding.weight.data.uniform_(-initrange, initrange)

        self.context_linear.weight.data.uniform_(-initrange, initrange)
        self.extend_linear.weight.data.uniform_(-initrange, initrange)
        self.gate_linear.weight.data.uniform_(-initrange, initrange)

        self.pre_vocab_linear.bias.data.fill_(0)
        self.pre_vocab_linear.weight.data.uniform_(-initrange, initrange)

        self.vocab_linear.bias.data.fill_(0)
        self.vocab_linear.weight.data.uniform_(-initrange, initrange)

    def context_attention(self, output, context_embed):
        """
        args:
            output: shape [batch_size, seq_len, hidden_size]
            context_embed: shape [batch_size, 3, embed_size]
        outputs:
            context_repre: shape [batch_size, seq_len, embed_size]
        """
        n_context = context_embed.size(1)
        seq_len = output.size(1)

        output = output.unsqueeze(2).expand(-1, -1, n_context, -1)
        context = context_embed.unsqueeze(1).expand(-1, seq_len, -1, -1)

        # [batch_size, seq_len, n_context]
        energies = self.context_linear(torch.cat([output, context], dim=-1)).squeeze(-1)

        context_probs = F.softmax(energies, dim=-1)  # [batch_size, seq_len, n_context]

        context_repre = torch.matmul(context_probs, context_embed)  # [batch_size, seq_len, embed_size]

        return context_repre

    def extend_attention(self, global_repre, extend_embed, extend_mask):
        """
        args:
            global_repre: shape [batch_size, seq_len, embed_size + hidden_size]
            extend_embed: shape [batch_size, seq_len, extend_size, embed_size]
            extend_mask: shape [batch_size, seq_len, extend_size] 0/1
        outputs:
            extend_probs: shape [batch_size, seq_len, extend_size]
        """
        n_extend = extend_embed.size(2)

        global_repre = global_repre.unsqueeze(2).expand(-1, -1, n_extend, -1)

        # [batch_size, seq_len, n_extend]
        energies = self.extend_linear(torch.cat([global_repre, extend_embed], dim=-1)).squeeze(-1)
        extend_probs = F.softmax(energies, dim=-1) * extend_mask

        normalization_factor = extend_probs.sum(-1, keepdim=True) + 1e-12
        extend_probs = extend_probs / normalization_factor

        return extend_probs

    def forward_context(self, context):
        context_output, hidden = self.context_encoder(context)
        return context_output, hidden

    def forward(self, context_embed, last_hidden, input_seq, input_aspect, extend_input):
        """
        :param context_embed:
            attribute outputs, in shape (batch_size, 3, embed_size)
        :param last_hidden:
            last hidden stat of the decoder, in shape (rnn_layers, batch_size, hidden_size)
        :param input_seq:
            word input for all time steps, in shape (batch_size, seq_len)
        :param input_aspect:
            current aspect, in shape (batch_size, seq_len)
        :param extend_input:
            extend entity and word for the decoder, in shape (batch_size, seq_len, extend_size)
        """
        # (seq_len, batch_size, embed_size)
        token_embed = self.token_embedding(input_seq)
        aspect_embed = self.aspect_embedding(input_aspect)
        input_embed = self.dropout_layer(torch.mul(token_embed, aspect_embed))

        # (batch_size, seq_len, extend_size, embed_size)
        extend_embed = self.token_embedding(extend_input)
        extend_mask = torch.ne(extend_input, PAD_ID)

        # rnn_output: (batch_size, seq_len, hidden_size)
        # rnn_hidden: (rnn_layers, batch_size, hidden_size)
        rnn_output, hidden = self.decoder(input_embed, last_hidden)

        # [batch_size, seq_len, embed_size]
        context_repre = self.context_attention(rnn_output, context_embed)

        # global representation [batch_size, seq_len, hidden_size + embed_size]
        global_repre = torch.cat((rnn_output, context_repre), 2)

        # [batch_size, seq_len, 1]
        p_gen = torch.sigmoid(self.gate_linear(global_repre))

        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, token_vocab_size]
        pre_vocab_out = torch.tanh(self.pre_vocab_linear(global_repre))
        gen_prob = p_gen * F.softmax(self.vocab_linear(pre_vocab_out), dim=-1)

        # [batch_size, seq_len, extend_size]
        copy_prob = (1 - p_gen) * self.extend_attention(global_repre, extend_embed, extend_mask)

        probs = gen_prob.scatter_add_(-1, extend_input, copy_prob)

        output = torch.log(probs + 1e-12)

        return output, hidden
