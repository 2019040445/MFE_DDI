from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import time
from torch.nn.parameter import Parameter
from AMDE_implementations import Graph_encoder

torch.manual_seed(1)
np.random.seed(1)

class BilinearDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super(BilinearDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.relation = Parameter(torch.FloatTensor(input_dim, input_dim))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.relation.data)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_row = inputs
        inputs_col = inputs.transpose(0, 1)
        inputs_row = self.dropout(inputs_row)
        inputs_col = self.dropout(inputs_col)
        intermediate_product = torch.mm(inputs_row, self.relation)
        rec = torch.mm(intermediate_product, inputs_col)
        rec = nn.ReLU(True)(rec)
        n = rec.size(0)
        # print(n)
        rec = nn.BatchNorm1d(n).cuda()(rec)
        outputs = nn.Linear(n, 1).cuda()(rec)
        print('outputs: ', outputs)
        return outputs

'''
    sum
'''
class MultiLevelDDI(nn.Module):
    def __init__(self,**config):
        super(MultiLevelDDI, self).__init__()
        self.molor=Molormer(**config)
        self.amde=Graph_encoder(node_features_1=75,edge_features_1=4, node_features_2=75, edge_features_2=4, out_features=1,**config) #parameters

        self.hidden_dim = config['hidden_dim']
        self.input_dropout = nn.Dropout(config['input_dropout_rate'])
        self.icnn = nn.Conv1d(self.hidden_dim, 16, 3, 3)
        self.decoder = nn.Sequential(
            nn.Linear(539, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
        self.w = nn.Parameter(torch.ones(3))
    def forward(self, d1_node, d1_attn_bias, d1_spatial_pos, d1_in_degree, d1_out_degree, d1_edge_input,
                      d2_node, d2_attn_bias, d2_spatial_pos, d2_in_degree, d2_out_degree, d2_edge_input,
                      adj_1, nd_1, ed_1,
                      adj_2, nd_2, ed_2,
                      d1, d2, mask_1, mask_2):
        drug1_n_graph = d1_node.size()[0]
        start1 = time.time()
        molor_d1_feature , molor_d2_feature = self.molor(d1_node, d1_attn_bias, d1_spatial_pos, d1_in_degree, d1_out_degree, d1_edge_input,
                                 d2_node, d2_attn_bias, d2_spatial_pos, d2_in_degree, d2_out_degree, d2_edge_input)
        # print('Running time of Molormer: %s Seconds'%(end1 - start1))
        # drug1_output, drug2_output [b, 65, 256] 
        output_1, d1_seq_fts_layer1, output_2, d2_seq_fts_layer1= self.amde(adj_1, nd_1, ed_1,
                               adj_2, nd_2, ed_2,
                               d1, d2, mask_1, mask_2)  #[b,203]
        
        molor_d1_feature= self.icnn(molor_d1_feature.permute(0, 2, 1)) #[b, 16, 21])
        # print(molor_d1_feature.shape)
        d1_feature = molor_d1_feature.view(drug1_n_graph, -1) #[b, 336]
        molor_d2_feature = self.icnn(molor_d2_feature.permute(0, 2, 1))
        d2_feature = molor_d2_feature.view(drug1_n_graph, -1)

        '''
        权重归一化
        '''
        # w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        # w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        # w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        
        
        # d1_feature = torch.cat((w1*d1_feature , w2*output_1, w3*d1_seq_fts_layer1),dim = 1)
        # d2_feature = torch.cat((w1*d2_feature , w2*output_2, w3*d2_seq_fts_layer1),dim = 1)#[b, 539]

        d1_feature = torch.cat((d1_feature , output_1, d1_seq_fts_layer1),dim = 1)
        d2_feature = torch.cat((d2_feature , output_2, d2_seq_fts_layer1),dim = 1)#[b, 539]
        final_fts_sum = d1_feature + d2_feature

        score = self.decoder(final_fts_sum)
        
        return score

class Molormer(nn.Sequential):
    '''
        Molormer Network with spatial graph encoder and lightweight attention block
    '''
    def __init__(self, **config):
        super(Molormer, self).__init__()

        self.gpus = torch.cuda.device_count()
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.hidden_dim = config['hidden_dim']
        self.inter_dim = config['inter_dim']
        self.flatten_dim = config['flatten_dim']
        self.multi_hop_max_dist = 20
        
        # dropout
        self.encoder_dropout = config['encoder_dropout_rate']
        self.attention_dropout = config['attention_dropout_rate']
        self.input_dropout = nn.Dropout(config['input_dropout_rate'])

        # Embeddings
        self.d_node_encoder = nn.Embedding(512*9+1, self.hidden_dim, padding_idx=0)
        self.d_edge_encoder = nn.Embedding(512*3+1, self.num_heads, padding_idx=0)
        self.d_edge_dis_encoder = nn.Embedding(128 * self.num_heads * self.num_heads, 1)
        self.d_spatial_pos_encoder = nn.Embedding(512, self.num_heads, padding_idx=0)
        self.d_in_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)
        self.d_out_degree_encoder = nn.Embedding(512, self.hidden_dim, padding_idx=0)
      
        self.d_encoders = Encoder(hidden_dim=self.hidden_dim, inter_dim=self.inter_dim,
                                  n_layers=self.num_layers, n_heads=self.num_heads)

        self.d_final_ln = nn.LayerNorm(self.hidden_dim)
        self.d_graph_token = nn.Embedding(1, self.hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, config['num_heads'])
        self.icnn = nn.Conv1d(self.hidden_dim, 16, 3)


    def forward(self, d1_node, d1_attn_bias, d1_spatial_pos, d1_in_degree, d1_out_degree, d1_edge_input,
                      d2_node, d2_attn_bias, d2_spatial_pos, d2_in_degree, d2_out_degree, d2_edge_input):

        drug1_n_graph, drug1_n_node = d1_node.size()[:2] #2,256
        drug2_n_graph, drug2_n_node = d2_node.size()[:2]


        drug1_graph_attn_bias = d1_attn_bias.clone()
        # print('drug1_graph_attn_bias: ',drug1_graph_attn_bias.shape) torch.Size([2, 257, 257])
        # assert False
        drug1_graph_attn_bias = drug1_graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # print('drug1_graph_attn_bias: ',drug1_graph_attn_bias.shape)  #torch.Size([2, 8, 257, 257])
        # assert False

        drug2_graph_attn_bias = d2_attn_bias.clone()
        drug2_graph_attn_bias = drug2_graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        drug1_spatial_pos_bias = self.d_spatial_pos_encoder(d1_spatial_pos).permute(0, 3, 1, 2)


        #取前两个维度的全部，第三四个维度从第二个开始取
        drug1_graph_attn_bias[:, :, 1:, 1:] = drug1_graph_attn_bias[:, :, 1:, 1:] + drug1_spatial_pos_bias  #[2, 8, 257, 257]

        drug2_spatial_pos_bias = self.d_spatial_pos_encoder(d2_spatial_pos).permute(0, 3, 1, 2)
        drug2_graph_attn_bias[:, :, 1:, 1:] = drug2_graph_attn_bias[:, :, 1:, 1:] + drug2_spatial_pos_bias


        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)

        drug1_graph_attn_bias[:, :, 1:, 0] = drug1_graph_attn_bias[:, :, 1:, 0] + t
        drug1_graph_attn_bias[:, :, 0, :] = drug1_graph_attn_bias[:, :, 0, :] + t
        drug2_graph_attn_bias[:, :, 1:, 0] = drug2_graph_attn_bias[:, :, 1:, 0] + t
        drug2_graph_attn_bias[:, :, 0, :] = drug2_graph_attn_bias[:, :, 0, :] + t

        
        drug1_spatial_pos = d1_spatial_pos.clone() #[2, 256, 256]

        drug1_spatial_pos[drug1_spatial_pos == 0] = 1  # set pad to 1


        drug1_spatial_pos = torch.where(drug1_spatial_pos > 1, drug1_spatial_pos - 1, drug1_spatial_pos)

        drug1_spatial_pos = drug1_spatial_pos.clamp(0, self.multi_hop_max_dist)

        drug1_edge_input = d1_edge_input[:, :, :, :self.multi_hop_max_dist, :]
        
        # [n_graph, n_node, n_node, max_dist, n_head]
        drug1_edge_input = self.d_edge_encoder(drug1_edge_input).mean(-2)

        max_dist = drug1_edge_input.size(-2)
        drug1_edge_input_flat = drug1_edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        drug1_edge_input_flat = torch.bmm(drug1_edge_input_flat, self.d_edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :])
        drug1_edge_input = drug1_edge_input_flat.reshape(max_dist, drug1_n_graph, drug1_n_node, drug1_n_node, self.num_heads).permute(1, 2, 3, 0, 4)        # 2,43,43,20,8
        drug1_edge_input = (drug1_edge_input.sum(-2) /(drug1_spatial_pos.float().unsqueeze(-1))).permute(0, 3, 1, 2)          # 2,8,43,43

#################################################################################################################
       
        # edge_input
        drug2_spatial_pos = d2_spatial_pos.clone()
        drug2_spatial_pos[drug2_spatial_pos == 0] = 1  # set pad to 1
        
        drug2_spatial_pos = torch.where(drug2_spatial_pos > 1, drug2_spatial_pos - 1, drug2_spatial_pos)
        drug2_spatial_pos = drug2_spatial_pos.clamp(0, self.multi_hop_max_dist)
        drug2_edge_input = d2_edge_input[:, :, :, :self.multi_hop_max_dist, :]
        
        # [n_graph, n_node, n_node, max_dist, n_head]
        drug2_edge_input = self.d_edge_encoder(drug2_edge_input).mean(-2)

        max_dist = drug2_edge_input.size(-2)
        drug2_edge_input_flat = drug2_edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        drug2_edge_input_flat = torch.bmm(drug2_edge_input_flat, self.d_edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :])
        drug2_edge_input = drug2_edge_input_flat.reshape(max_dist, drug2_n_graph, drug2_n_node, drug2_n_node, self.num_heads).permute(1, 2, 3, 0, 4)  # 2,43,43,20,8
        drug2_edge_input = (drug2_edge_input.sum(-2) / (drug2_spatial_pos.float().unsqueeze(-1))).permute(0, 3, 1, 2)  # 2,8,43,43

        
        drug1_graph_attn_bias[:, :, 1:, 1:] = drug1_graph_attn_bias[:,:, 1:, 1:] + drug1_edge_input
        drug1_graph_attn_bias = drug1_graph_attn_bias + d1_attn_bias.unsqueeze(1)

        drug2_graph_attn_bias[:, :, 1:, 1:] = drug2_graph_attn_bias[:,:, 1:, 1:] + drug2_edge_input
        drug2_graph_attn_bias = drug2_graph_attn_bias + d2_attn_bias.unsqueeze(1)

        drug1_node_feature = self.d_node_encoder(d1_node).sum(dim=-2)
        drug1_node_feature = drug1_node_feature + self.d_in_degree_encoder(d1_in_degree) + self.d_out_degree_encoder(d1_out_degree)

        drug1_graph_token_feature = self.d_graph_token.weight.unsqueeze(0).repeat(drug1_n_graph, 1, 1)#[2, 1, 256]
        drug1_graph_node_feature = torch.cat([drug1_graph_token_feature, drug1_node_feature], dim = 1) #[2, 257, 256]

        drug2_node_feature = self.d_node_encoder(d2_node).sum(dim=-2) 
        drug2_node_feature = drug2_node_feature + self.d_in_degree_encoder(d2_in_degree) + self.d_out_degree_encoder(d2_out_degree)
        drug2_graph_token_feature = self.d_graph_token.weight.unsqueeze(0).repeat(drug2_n_graph, 1, 1)
        drug2_graph_node_feature = torch.cat([drug2_graph_token_feature, drug2_node_feature], dim=1)

        # transfomrer encoder
        drug1_output = self.input_dropout(drug1_graph_node_feature) #[2, 257, 256]-->[2, 257, 256]
        drug1_output = self.d_encoders(drug1_output, drug1_graph_attn_bias) #([2, 257, 256],[2, 8, 257, 257])--->[2, 65, 256]


        drug2_output = self.input_dropout(drug2_graph_node_feature)
        drug2_output = self.d_encoders(drug2_output, drug2_graph_attn_bias)

        # print('M: ',drug1_output.shape,drug2_output.shape)
        return drug1_output , drug2_output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

                                   
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(AttentionLayer, self).__init__()

        key_dim = hidden_dim // n_heads
        value_dim = hidden_dim // n_heads

        self.inner_attention = ProbAttention(False, factor=5, attention_dropout=0.0, output_attention=False)
        self.query_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.key_projection = nn.Linear(hidden_dim, key_dim * n_heads)
        self.value_projection = nn.Linear(hidden_dim, value_dim * n_heads)
        self.out_projection = nn.Linear(value_dim * n_heads, hidden_dim)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
     
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
      
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
        
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
  
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
      
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class Encoder(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_layers, n_heads, dropout=0.0):#255，255，3
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(Encoder_layer(hidden_dim, inter_dim, n_heads, dropout) for l in range(n_layers))
        self.conv_layers = nn.ModuleList(Distilling_layer(hidden_dim) for _ in range(n_layers - 1)) #2
        self.norm =torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            x = conv_layer(x)
            attns.append(attn)
        x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
        attns.append(attn)
    
        x = self.norm(x)
        
        return x


class Encoder_layer(nn.Module):
    def __init__(self, hidden_dim, inter_dim, n_heads, dropout):
        super(Encoder_layer, self).__init__()
        self.attention = AttentionLayer(hidden_dim=hidden_dim, n_heads=n_heads)
        self.conv1 = nn.Conv1d(hidden_dim, inter_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(inter_dim, hidden_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = F.relu

    def forward(self, x, attn_mask=None):
        attn_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_x)
        y = x = self.norm1(x)
        y = self.dropout(self.relu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Distilling_layer(nn.Module):
    def __init__(self, channel):
        super(Distilling_layer, self).__init__()
    
        self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(channel)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        out = self.maxPool(self.activation(self.norm(x))).transpose(1, 2)

        return out
