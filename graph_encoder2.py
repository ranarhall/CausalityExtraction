import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import scipy.stats as stats
import modeling_bert

from gate import HighwayGateLayer
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32):
    """Outputs random values from a truncated normal distribution.
    The generated values follow a normal distribution with specified mean
    and standard deviation, except that values whose magnitude is more
    than 2 standard deviations from the mean are dropped and re-picked.
    API from: https://www.tensorflow.org/api_docs/python/tf/truncated_normal
    """
    lower = -2 * stddev + mean
    upper = 2 * stddev + mean
    X = stats.truncnorm((lower - mean) / stddev,
                        (upper - mean) / stddev,
                        loc=mean,
                        scale=stddev)
    values = X.rvs(size=shape)
    return torch.from_numpy(values.astype(dtype))


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_size, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_size)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class ScaledEmbedding(nn.Embedding):

    def reset_parameters(self):
        if os.path.exists('cache.pt') == True:
            self.weight.data = torch.load('cache.pt')
            return
        self.weight.data = truncated_normal(shape=(self.num_embeddings,self.embedding_dim),stddev=1.0 / math.sqrt(self.embedding_dim))
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

        torch.save(self.weight.data, 'cache.pt')


class GELU(nn.Module):
    def __init__(self, inplace=False):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.gelu(input, inplace=self.inplace)


class GNNClassifier(nn.Module):
    """ A wrapper classifier for GNNRelationModel. """
    def __init__(self):
        super().__init__()
        self.syntax_encoder = GNNRelationModel()
        self.classifier = nn.Linear(768, 3)

    def resize_token_embeddings(self, new_num_tokens=None):
        self.syntax_encoder.resize_token_embeddings(new_num_tokens)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, dep_head=None, dep_rel=None, wp_rows=None, align_sizes=None,
                seq_len=None, subj_pos=None, obj_pos=None):
        pooled_output, sequence_output = self.syntax_encoder(input_ids,
                                                             attention_mask,
                                                             dep_head,
                                                             dep_rel,
                                                             seq_len)
        logits = self.classifier(pooled_output)
        return logits

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func: from_pretrained`` class method.
        """
        return


class GNNRelationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        n_dim = 768
        
        self.syntax_encoder = GATEncoder().to(device)

        self.gate = HighwayGateLayer(768).to(device)

        out_dim = 768

        layers = [nn.Linear(out_dim,768),nn.Tanh()]
        self.out_mlp = nn.Sequential(*layers).to(device)
        self.pool_mask, self.subj_mask, self.obj_mask = (None, None, None)

    def resize_token_embeddings(self, new_num_tokens=None):
        if new_num_tokens is None:
            return

        old_num_tokens, old_embedding_dim = self.emb.weight.size()
        if old_num_tokens == new_num_tokens:
            return

        # Build new embeddings
        new_embeddings = ScaledEmbedding(new_num_tokens,old_embedding_dim,padding_idx=0)
        #new_embeddings.to(self.emb.weight.device)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = self.emb.weight.data[:num_tokens_to_copy, :]
        self.emb = new_embeddings

    def encode_with_rnn(self, rnn_inputs, seq_lens):
        batch_size = rnn_inputs.size(0)
        h0, c0 = rnn_zero_state(batch_size,384,1)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs,
                                                       seq_lens,
                                                       batch_first=True,
                                                       enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs,
                                         (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs,
                                                          batch_first=True)
        return rnn_outputs

    def forward(self, input_ids_or_bert_hidden, adj=None, dep_rel_matrix=None, wp_seq_lengths=None):

        attention_mask = adj.clone().detach().unsqueeze(1)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0

        h = self.syntax_encoder(input_ids_or_bert_hidden,attention_mask.to(device),None)
        #h = self.gate(input_ids_or_bert_hidden,h)

        return h


class GATEncoder(nn.Module):
    def __init__(self):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList().to(device)
        layer = GATEncoderLayer().to(device)
        self.layers.append(layer)
        self.ln = nn.LayerNorm(768+30,
                               eps=1e-6)

    def forward(self, e, attention_mask, dep_rel_matrix=None):
        for layer in self.layers:
            e = layer(e,
                      attention_mask,
                      dep_rel_matrix)
        e = self.ln(e)
        return e


class GATEncoderLayer(nn.Module):
    def __init__(self):
        super(GATEncoderLayer, self).__init__()
        self.syntax_attention = RelationalBertSelfAttention().to(device)
        self.finishing_linear_layer = nn.Linear(768,30).to(device)
        self.dropout1 = nn.Dropout(0.1)
        self.ln_2 = nn.LayerNorm(768,eps=1e-6)
        #self.feed_forward = FeedForwardLayer(0.1).to(device)
        #self.dropout2 = nn.Dropout(0.1)
        #self.ln_3 = nn.LayerNorm(768,eps=1e-6)
  

    def forward(self, e, attention_mask, dep_rel_matrix=None):
        sub = self.finishing_linear_layer(self.syntax_attention(self.ln_2(e),attention_mask,dep_rel_matrix)[0])
        sub = self.dropout1(sub)
        e = torch.cat((e,sub), 2)

        #sub = self.feed_forward(self.ln_3(e))
        #sub = self.dropout2(sub)
        #e = e + sub
        return e


class FeedForwardLayer(nn.Module):
    def __init__(self, activation_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.W_1 = nn.Linear(768,3072).to(device)
        self.act = modeling_bert.ACT2FN['gelu']
        self.dropout = nn.Dropout(activation_dropout)
        self.W_2 = nn.Linear(3072,768).to(device)

    def forward(self, e):
        e = self.dropout(self.act(self.W_1(e)))
        e = self.W_2(e)
        return e


class RelationalBertSelfAttention(nn.Module):
    def __init__(self):
        super(RelationalBertSelfAttention, self).__init__()
        self.output_attentions = False

        self.num_attention_heads = 12
        self.attention_head_size = int(768 / 12)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(768,self.all_head_size).to(device)
        self.key = nn.Linear(768,self.all_head_size).to(device)
        self.value = nn.Linear(768,self.all_head_size).to(device)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).to(device)

    def forward(self, hidden_states, attention_mask, dep_rel_matrix=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1, -2)).to(device)

        rel_attention_scores = 0

        attention_scores = (attention_scores + rel_attention_scores) / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs,value_layer).to(device)


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer.to(device), attention_probs.to(device)) if self.output_attentions else (context_layer,)
        return outputs
