import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import os
import json
import random
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, confusion_matrix, roc_curve, auc
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import traceback

device = None

SECTOR_TO_STOCKS = {
    "Basic Materials": ["XOM", "RDS-B", "PTR", "CVX", "TOT", "BP", "BHP", "SNP", "SLB", "BBL"],
    "Consumer Goods": ["AAPL", "PG", "BUD", "KO", "PM", "TM", "PEP", "UN", "UL", "MO"],
    "Healthcare": ["JNJ", "PFE", "NVS", "UNH", "MRK", "AMGN", "MDT", "ABBV", "SNY", "CELG"],
    "Services": ["AMZN", "BABA", "WMT", "CMCSA", "HD", "DIS", "MCD", "CHTR", "UPS", "PCLN"],
    "Utilities": ["NEE", "DUK", "D", "SO", "NGG", "AEP", "PCG", "EXC", "SRE", "PPL"],
    "Conglomerates": ["IEP", "HRG", "CODI", "REX", "SPLP", "PICO", "AGFS", "GMRE"],
    "Financial": ["BCH", "BSAC", "BRK-A", "JPM", "WFC", "BAC", "V", "C", "HSBC", "MA"],
    "Industrial Goods": ["GE", "MMM", "BA", "HON", "UTX", "LMT", "CAT", "GD", "DHR", "ABB"],
    "Technology": ["GOOG", "MSFT", "FB", "T", "CHL", "ORCL", "TSM", "VZ", "INTC", "CSCO"]
}

def build_incidence_matrix(selected_companies, sector_to_stocks=SECTOR_TO_STOCKS):
    sectors = list(sector_to_stocks.keys())
    num_nodes = len(selected_companies)
    num_edges = len(sectors)
    H = torch.zeros((num_nodes, num_edges))
    for j, sector in enumerate(sectors):
        for stock in sector_to_stocks[sector]:
            if stock in selected_companies:
                i = selected_companies.index(stock)
                H[i][j] = 1.0
    return H.to(device)

# Hypergraph Attention Layer
class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.6, alpha=0.2):
        super(HypergraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.constant_(self.a, 0.1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, X, H):
        XW = torch.matmul(X, self.W)
        Dv = torch.sum(H, dim=1, keepdim=True)
        De = torch.sum(H, dim=0, keepdim=True)
        H_T = H.t()
        XW_H = torch.matmul(H_T, XW) / (De.t() + 1e-6)
        HXW = torch.matmul(H, XW_H) / (Dv + 1e-6)
        e = self.leakyrelu(HXW * self.a)
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        out = attention * HXW
        return F.elu(out)

# Step 1: Tweet Embedding Model
class TweetEmbeddingModel(nn.Module):
    def __init__(self):
        super(TweetEmbeddingModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, tweets, max_tweets=10):
        if not tweets or not isinstance(tweets, list):
            return torch.zeros((max_tweets, 768), dtype=torch.float32, device=self.bert.device)

        cleaned_tweets = [str(t) for t in tweets if isinstance(t, str)]
        if not cleaned_tweets:
            return torch.zeros((max_tweets, 768), dtype=torch.float32, device=self.bert.device)

        try:
            batch_size = 8
            embeddings = []
            for i in range(0, len(cleaned_tweets), batch_size):
                batch_tweets = cleaned_tweets[i:i + batch_size]
                encoded_input = self.tokenizer(
                    batch_tweets,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    max_length=128,
                    clean_up_tokenization_spaces=True
                )
                encoded_input = {k: v.to(self.bert.device) for k, v in encoded_input.items()}
                with torch.no_grad():
                    output = self.bert(**encoded_input)
                embeddings.append(output.last_hidden_state[:, 0, :])
            embeddings = torch.cat(embeddings, dim=0)
            padded_embeddings = torch.zeros((max_tweets, 768), dtype=torch.float32, device=self.bert.device)
            num_tweets = min(len(embeddings), max_tweets)
            padded_embeddings[:num_tweets] = embeddings[:num_tweets]
            return padded_embeddings
        except Exception as e:
            st.error(f"Error in BERT embedding: {e}")
            return torch.zeros((max_tweets, 768), dtype=torch.float32, device=self.bert.device)

# Step 2: Normalize price data
def normalize_price_data(price_data, prev_close=None):
    price_data = price_data.float()
    if prev_close is not None:
        prev_close = torch.clamp(prev_close, min=1e-4, max=1e6)
        normalized = price_data / (prev_close.unsqueeze(-1) + 1e-8)
        normalized = torch.clamp(normalized, min=-10.0, max=10.0)
        return normalized
    mean = price_data.mean(dim=0)
    std = price_data.std(dim=0) + 1e-8
    normalized = (price_data - mean) / std
    return torch.clamp(normalized, min=-10.0, max=10.0)

# Step 3: Financial evaluation metrics
def calculate_profit(predictions, actuals, prices):
    predictions = torch.tensor(predictions, dtype=torch.float32, device=prices.device)
    actuals = torch.tensor(actuals, dtype=torch.float32, device=prices.device)
    returns = predictions - actuals
    profit = (returns * prices[:, 2]).sum()
    return profit.item()

def calculate_sharpe_ratio(predictions, actuals):
    predictions = torch.tensor(predictions, dtype=torch.float32)
    actuals = torch.tensor(actuals, dtype=torch.float32)
    returns = predictions - actuals
    mean_return = returns.mean()
    std_return = returns.std() + 1e-8
    sharpe_ratio = mean_return / std_return
    return sharpe_ratio.item()

# Step 4: Graph Attention Layers
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        N = input.size()[0]
        edge = adj.nonzero().t()
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum + 1e-8)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

# Step 5: GRU and Attention Modules
class GRUModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUModule, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, inputs):
        full, last = self.gru(inputs)
        return full, last

class AttentionModule(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(AttentionModule, self).__init__()
        self.W1 = nn.Linear(in_shape, out_shape)
        self.W2 = nn.Linear(in_shape, out_shape)
        self.V = nn.Linear(in_shape, 1)

    def forward(self, full, last):
        score = self.V(F.tanh(self.W1(last) + self.W2(full)))
        score = torch.clamp(score, min=-5.0, max=5.0)
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * full
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

# Step 6: Main GAT/HGAT Model
class GATStockPredictionModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, stock_num, sparse=False, attention_mode='gat'):
        super(GATStockPredictionModel, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.attention_mode = attention_mode
        self.tweet_embedding = TweetEmbeddingModel()
        self.price_gru = nn.ModuleList([GRUModule(3, 64) for _ in range(stock_num)])
        self.price_attn = nn.ModuleList([AttentionModule(64, 64) for _ in range(stock_num)])
        self.tweet_gru = nn.ModuleList([GRUModule(768, 64) for _ in range(stock_num)])
        self.tweet_attn = nn.ModuleList([AttentionModule(64, 64) for _ in range(stock_num)])
        self.combined_gru = nn.ModuleList([GRUModule(64, 64) for _ in range(stock_num)])
        self.combined_attn = nn.ModuleList([AttentionModule(64, 64) for _ in range(stock_num)])
        self.bilinear = nn.ModuleList([nn.Bilinear(64, 64, 64) for _ in range(stock_num)])
        self.layer_normt = nn.ModuleList([nn.LayerNorm(64) for _ in range(stock_num)])
        self.layer_normp = nn.ModuleList([nn.LayerNorm(64) for _ in range(stock_num)])
        self.classifier = nn.ModuleList([nn.Linear(64, 2) for _ in range(stock_num)])
        self.regressor = nn.ModuleList([nn.Linear(64 + 3, 1) for _ in range(stock_num)])

        # GAT Layers
        AttentionLayer = SpGraphAttentionLayer if sparse else GraphAttentionLayer
        self.gat_attentions = nn.ModuleList([AttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        self.gat_out_att = AttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        # HGAT Layers
        self.hgat_attentions = nn.ModuleList([HypergraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)])
        self.hgat_out_att = HypergraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

        # Combination Layer for Both GAT and HGAT
        if attention_mode == 'both':
            self.combine_layer = nn.Linear(nclass * 2, nclass)

    def forward(self, text_input, price_input, adj, H=None):
        num_stocks = price_input.size(0)
        num_days = price_input.size(1)
        max_tweets = 10
        feature_list = []
        price_attn_weights = []
        tweet_attn_weights = []
        combined_attn_weights = []

        for i in range(num_stocks):
            price_full, price_last = self.price_gru[i](price_input[i].unsqueeze(0))
            price_vector, price_attn_w = self.price_attn[i](price_full, price_last)
            price_vector = price_vector.squeeze(0)
            price_vector = self.layer_normp[i](price_vector)
            price_attn_weights.append(price_attn_w.squeeze())

            tweet_vectors = []
            tweet_day_attn_weights = []
            for j in range(num_days):
                tweets = text_input[i][j]
                tweet_emb = self.tweet_embedding(tweets, max_tweets)
                tweet_full, tweet_last = self.tweet_gru[i](tweet_emb.unsqueeze(0))
                tweet_vector, tweet_attn_w = self.tweet_attn[i](tweet_full, tweet_last)
                tweet_vectors.append(tweet_vector.squeeze(0))
                tweet_day_attn_weights.append(tweet_attn_w.squeeze())
            tweet_vectors = torch.stack(tweet_vectors, dim=0).unsqueeze(0)
            tweet_attn_weights.append(tweet_day_attn_weights)

            combined_full, combined_last = self.combined_gru[i](tweet_vectors)
            combined_vector, combined_attn_w = self.combined_attn[i](combined_full, combined_last)
            combined_vector = combined_vector.squeeze(0)
            combined_vector = self.layer_normt[i](combined_vector)
            combined_attn_weights.append(combined_attn_w.squeeze())

            combined = F.tanh(self.bilinear[i](combined_vector, price_vector))
            feature_list.append(combined)

        ft_vec = torch.stack(feature_list, dim=0)
        class_out = [self.classifier[i](ft_vec[i]).unsqueeze(0) for i in range(num_stocks)]
        class_out = torch.cat(class_out, dim=0)

        x = F.dropout(ft_vec, self.dropout, training=self.training)
        if self.attention_mode == 'gat':
            x_gat = torch.cat([att(x, adj) for att in self.gat_attentions], dim=1)
            x_gat = F.dropout(x_gat, self.dropout, training=self.training)
            att_out = F.elu(self.gat_out_att(x_gat, adj))
        elif self.attention_mode == 'hgat':
            x_hgat = torch.cat([att(x, H) for att in self.hgat_attentions], dim=1)
            x_hgat = F.dropout(x_hgat, self.dropout, training=self.training)
            att_out = F.elu(self.hgat_out_att(x_hgat, H))
        else:  # both
            x_gat = torch.cat([att(x, adj) for att in self.gat_attentions], dim=1)
            x_gat = F.dropout(x_gat, self.dropout, training=self.training)
            gat_out = F.elu(self.gat_out_att(x_gat, adj))
            x_hgat = torch.cat([att(x, H) for att in self.hgat_attentions], dim=1)
            x_hgat = F.dropout(x_hgat, self.dropout, training=self.training)
            hgat_out = F.elu(self.hgat_out_att(x_hgat, H))
            combined_out = torch.cat([gat_out, hgat_out], dim=1)
            att_out = self.combine_layer(combined_out)

        price_normalized = normalize_price_data(price_input[:, -1, :])
        reg_out = [self.regressor[i](torch.cat([ft_vec[i], price_normalized[i]], dim=0)).unsqueeze(0) for i in range(num_stocks)]
        reg_out = torch.cat(reg_out, dim=0)
        reg_out = torch.clamp(reg_out, min=-10.0, max=10.0)

        return (class_out + att_out, reg_out, torch.stack(price_attn_weights), tweet_attn_weights, torch.stack(combined_attn_weights))

# Step 7: Data Loading Functions
@st.cache_data
def load_price_data(base_path, companies):
    price_data = {}
    for company in companies:
        file_path = os.path.join(base_path, 'price', 'preprocessed', f'{company}.txt')
        if os.path.exists(file_path):
            data = np.loadtxt(file_path, delimiter='\t', dtype=str)
            dates = data[:, 0]
            features = data[:, 1:].astype(float)
            features = features[:, [0, 1, 4]]  # Open, high, adjusted close
            price_data[company] = {'dates': dates, 'features': features}
    return price_data

@st.cache_data
def load_tweet_data(base_path, company, dates):
    tweet_data = {}
    for date in dates:
        file_path = os.path.join(base_path, 'tweet', 'preprocessed', company, date)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                tweets = [json.loads(line) for line in f]
            tweet_data[date] = [t['text'] for t in tweets]
        else:
            tweet_data[date] = []
    return tweet_data

@st.cache_data
def prepare_data(base_path, companies, days, val_split=0.2):
    price_data = load_price_data(base_path, companies)
    all_dates = sorted(set().union(*[set(d['dates']) for d in price_data.values()]))
    
    X_price = []
    X_text = []
    y_class = []
    y_reg = []
    date_keys = []
    
    for i in range(days, len(all_dates)):
        window_dates = all_dates[i-days:i]
        target_date = all_dates[i]
        
        price_window = []
        text_window = []
        class_labels = []
        reg_labels = []
        
        for company in companies:
            if company not in price_data:
                break
            company_prices = price_data[company]['features']
            company_dates = price_data[company]['dates']
            idx = [np.where(company_dates == d)[0][0] for d in window_dates if d in company_dates]
            
            if len(idx) == days:
                price_seq = company_prices[idx]
                price_window.append(price_seq)
                
                tweet_data = load_tweet_data(base_path, company, window_dates)
                tweet_seq = [tweet_data.get(date, []) for date in window_dates]
                text_window.append(tweet_seq)
                
                if target_date in company_dates and window_dates[-1] in company_dates:
                    curr_idx = np.where(company_dates == target_date)[0][0]
                    prev_idx = np.where(company_dates == window_dates[-1])[0][0]
                    class_label = 1 if company_prices[curr_idx, 2] > company_prices[prev_idx, 2] else 0
                    curr_close = company_prices[curr_idx, 2]
                    prev_close = company_prices[prev_idx, 2]
                    if prev_close > 1e-4 and curr_close > 1e-4:
                        ratio = curr_close / prev_close
                        reg_label = np.clip(np.log1p(ratio - 1), -1.0, 1.0)
                    else:
                        reg_label = 0.0
                    class_labels.append(class_label)
                    reg_labels.append(reg_label)
                else:
                    break
            else:
                break
        
        if len(price_window) == len(companies):
            price_tensor = torch.tensor(np.stack(price_window), dtype=torch.float32)
            for d in range(1, days):
                prev_close = price_tensor[:, d-1, 2]
                price_tensor[:, d, :] = normalize_price_data(price_tensor[:, d, :], prev_close)
            X_price.append(price_tensor)
            X_text.append(text_window)
            y_class.append(class_labels)
            y_reg.append(reg_labels)
            date_keys.append(target_date)
    
    y_class = np.array(y_class, dtype=np.int64)
    y_reg = np.array(y_reg, dtype=np.float32)
    
    num_samples = len(X_price)
    num_val = int(val_split * num_samples)
    indices = np.random.permutation(num_samples)
    train_idx, val_idx = indices[num_val:], indices[:num_val]
    
    X_price_train = torch.tensor(np.stack([X_price[i] for i in train_idx]), dtype=torch.float32)
    X_text_train = [X_text[i] for i in train_idx]
    y_class_train = torch.tensor(y_class[train_idx], dtype=torch.long)
    y_reg_train = torch.tensor(y_reg[train_idx], dtype=torch.float32)
    date_keys_train = [date_keys[i] for i in train_idx]
    
    X_price_val = torch.tensor(np.stack([X_price[i] for i in val_idx]), dtype=torch.float32)
    X_text_val = [X_text[i] for i in val_idx]
    y_class_val = torch.tensor(y_class[val_idx], dtype=torch.long)
    y_reg_val = torch.tensor(y_reg[val_idx], dtype=torch.float32)
    date_keys_val = [date_keys[i] for i in val_idx]
    
    return (X_price_train, X_text_train, y_class_train, y_reg_train, date_keys_train,
            X_price_val, X_text_val, y_class_val, y_reg_val, date_keys_val)

def load_adj(companies):
    num_stocks = len(companies)
    adj = np.ones((num_stocks, num_stocks)) - np.eye(num_stocks)
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_tensor = torch.FloatTensor(np.array(adj.todense())).to(device)
    H = build_incidence_matrix(companies)
    return adj_tensor, H

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

# Step 8: Evaluation Metrics
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Step 9: Training and Testing Functions
def train(epoch, model, optimizer, scheduler, X_price, X_text, y_class, y_reg, adj, H, num_samples, device, progress_bar, metrics_container, batch_size, progress_freq):
    model.train()
    optimizer.zero_grad()
    indices = np.random.permutation(num_samples)[:batch_size]
    
    try:
        total_loss = 0
        total_acc = 0
        total_profit = 0
        total_sharpe = 0
        for i in indices:
            price_input = X_price[i].to(device)
            text_input = X_text[i]
            class_labels = y_class[i].to(device)
            reg_labels = y_reg[i].to(device)
            
            class_out, reg_out, _, _, _ = model(text_input, price_input, adj, H)
            
            class_loss = F.cross_entropy(class_out, class_labels)
            reg_loss = F.mse_loss(reg_out.squeeze(), reg_labels)
            loss = class_loss + 0.01 * reg_loss
            
            acc = accuracy(class_out, class_labels)
            profit = calculate_profit(reg_out.squeeze(), reg_labels, price_input[:, -1, :])
            sharpe = calculate_sharpe_ratio(reg_out.squeeze(), reg_labels)
            
            loss.backward()
            total_loss += loss.item()
            total_acc += acc.item()
            total_profit += profit
            total_sharpe += sharpe
        
        total_loss /= batch_size
        total_acc /= batch_size
        total_profit /= batch_size
        total_sharpe /= batch_size
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % progress_freq == 0:
            metrics_container.write(
                f"Epoch: {epoch}, Loss: {total_loss:.4f}, Acc: {total_acc:.4f}, "
                f"Profit: {total_profit:.4f}, Sharpe: {total_sharpe:.4f}"
            )
            progress_bar.progress((epoch + 1) / st.session_state.epochs)
        
        return total_loss, total_acc, total_profit, total_sharpe
    except Exception as e:
        st.error(f"Error in training epoch {epoch}: {e}")
        traceback.print_exc()
        return None, None, None, None

def validate(model, X_price, X_text, y_class, y_reg, adj, H, device):
    model.eval()
    val_loss = []
    val_acc = []
    val_pred = []
    val_true = []
    val_reg_pred = []
    val_reg_true = []
    val_prices = []
    
    with torch.no_grad():
        for i in range(len(X_price)):
            price_input = X_price[i].to(device)
            text_input = X_text[i]
            class_labels = y_class[i].to(device)
            reg_labels = y_reg[i].to(device)
            
            try:
                class_out, reg_out, _, _, _ = model(text_input, price_input, adj, H)
                class_loss = F.cross_entropy(class_out, class_labels)
                reg_loss = F.mse_loss(reg_out.squeeze(), reg_labels)
                total_loss = class_loss + 0.01 * reg_loss
                acc = accuracy(class_out, class_labels)
                
                val_loss.append(total_loss.item())
                val_acc.append(acc.item())
                val_pred.append(class_out.max(1)[1].cpu().numpy())
                val_true.append(class_labels.cpu().numpy())
                val_reg_pred.append(reg_out.squeeze().cpu().numpy())
                val_reg_true.append(reg_labels.cpu().numpy())
                val_prices.append(price_input[:, -1, :].cpu().numpy())
            except Exception as e:
                st.error(f"Error in validation sample {i}: {e}")
                continue
    
    val_pred = np.concatenate(val_pred)
    val_true = np.concatenate(val_true)
    val_reg_pred = np.concatenate(val_reg_pred)
    val_reg_true = np.concatenate(val_reg_true)
    val_prices = np.concatenate(val_prices, axis=0)
    
    f1 = f1_score(val_true, val_pred, average='micro')
    mcc = matthews_corrcoef(val_true, val_pred)
    profit = calculate_profit(torch.tensor(val_reg_pred), torch.tensor(val_reg_true), torch.tensor(val_prices))
    sharpe = calculate_sharpe_ratio(torch.tensor(val_reg_pred), torch.tensor(val_reg_true))
    
    return np.mean(val_loss), np.mean(val_acc), f1, mcc, profit, sharpe

def test_dict(model, X_price, X_text, y_class, y_reg, adj, H, date_keys, companies, device):
    model.eval()
    pred_dict = {}
    reg_pred_dict = {}
    reg_true_dict = {}
    prev_price_dict = {}
    attn_dict = {}
    test_acc = []
    test_loss = []
    li_pred = []
    li_true = []
    li_reg_pred = []
    li_reg_true = []
    li_prices = []
    li_probs = []
    
    with torch.no_grad():
        for i in range(len(X_price)):
            price_input = X_price[i].to(device)
            text_input = X_text[i]
            class_labels = y_class[i].to(device)
            reg_labels = y_reg[i].to(device)
            date = date_keys[i]
            
            try:
                class_out, reg_out, price_attn_w, tweet_attn_w, combined_attn_w = model(text_input, price_input, adj, H)
                class_out_softmax = F.softmax(class_out, dim=1)
                
                pred_dict[date] = class_out_softmax.cpu().numpy()
                reg_pred_dict[date] = reg_out.squeeze().cpu().numpy()
                reg_true_dict[date] = reg_labels.cpu().numpy()
                prev_price_dict[date] = price_input[:, -1, 2].cpu().numpy()  # Adjusted close of the last day
                attn_dict[date] = {
                    'price_attn': price_attn_w.cpu().numpy(),
                    'tweet_attn': [w.cpu().numpy() for tw in tweet_attn_w for w in tw],
                    'combined_attn': combined_attn_w.cpu().numpy(),
                    'companies': companies
                }
                loss_test = F.cross_entropy(class_out, class_labels) + 0.01 * F.mse_loss(reg_out.squeeze(), reg_labels)
                acc_test = accuracy(class_out, class_labels)
                
                li_pred.append(class_out.max(1)[1].cpu().numpy())
                li_true.append(class_labels.cpu().numpy())
                li_reg_pred.append(reg_out.squeeze().cpu().numpy())
                li_reg_true.append(reg_labels.cpu().numpy())
                li_prices.append(price_input[:, -1, :].cpu().numpy())
                li_probs.append(class_out_softmax.cpu().numpy())
                
                test_loss.append(loss_test.item())
                test_acc.append(acc_test.item())
            except Exception as e:
                st.error(f"Error in test sample {i}: {e}")
                continue
    
    li_pred = np.concatenate(li_pred)
    li_true = np.concatenate(li_true)
    li_reg_pred = np.concatenate(li_reg_pred)
    li_reg_true = np.concatenate(li_reg_true)
    li_prices = np.concatenate(li_prices, axis=0)
    li_probs = np.concatenate(li_probs, axis=0)
    
    f1 = f1_score(li_true, li_pred, average='micro')
    mcc = matthews_corrcoef(li_true, li_pred)
    profit = calculate_profit(torch.tensor(li_reg_pred), torch.tensor(li_reg_true), torch.tensor(li_prices))
    sharpe = calculate_sharpe_ratio(torch.tensor(li_reg_pred), torch.tensor(li_reg_true))
    
    return pred_dict, reg_pred_dict, reg_true_dict, prev_price_dict, f1, mcc, attn_dict, np.mean(test_loss), np.mean(test_acc), profit, sharpe, li_pred, li_true, li_probs

# Step 10: streamlit
def main():
    st.title("Stock Price Prediction with GAT and Tweets")
    st.write("Train and test a Graph Attention Network model using stock prices and tweet sentiment.")
    
    with st.sidebar:
        st.header("Dataset Configuration")
        base_path = st.text_input("Dataset Path", "stocknet-dataset-master", help="Path to stocknet-dataset-master")
        companies = sorted([
            f[:-4] for f in os.listdir(os.path.join(base_path, 'price', 'preprocessed'))
            if f.endswith('.txt')
        ])
        st.session_state.num_companies = st.number_input(
            "Number of Companies",
            min_value=1,
            max_value=len(companies),
            value=5,
            help="Number of companies to randomly select for training/testing"
        )
        st.session_state.days = st.number_input(
            "Historical Days",
            min_value=1,
            max_value=500,
            value=5,
            help="Days of price/tweet history per sample (1–10)"
        )
        
        st.header("Training Configuration")
        st.session_state.use_cuda = st.checkbox(
            "Use GPU (CUDA)",
            value=torch.cuda.is_available(),
            help="Use GPU if available; uncheck for CPU"
        )
        attention_mode = st.radio(
            "Attention Mechanism",
            ["GAT Only", "HGAT Only", "Both GAT and HGAT"],
            help="Choose GAT (pairwise), HGAT (sector-based), or both"
        )
        st.session_state.attention_mode = 'both' if attention_mode == "Both GAT and HGAT" else 'hgat' if attention_mode == "HGAT Only" else 'gat'
        st.session_state.epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=5000,
            value=1000,
            help="Number of training epochs"
        )
        st.session_state.validate_freq = st.number_input(
            "Validation Frequency (every N epochs)",
            min_value=1,
            max_value=5000,
            value=100,
            help="Validate every N epochs (higher = faster)"
        )
        st.session_state.batch_size = st.number_input(
            "Training Batch Size",
            min_value=1,
            max_value=10,
            value=1,
            help="Samples per epoch (higher = faster convergence, more memory)"
        )
        st.session_state.progress_freq = st.number_input(
            "Progress Update Frequency (epochs)",
            min_value=1,
            max_value=50,
            value=1,
            help="Update UI every N epochs (higher = faster UI)"
        )
        sparse = st.checkbox("Use Sparse GAT", value=False, help="Use sparse GAT for large/sparse graphs (only with GAT)")
        mode = st.radio("Mode", ["Train", "Test"], help="Train model or test on validation set")
        train_button = st.button("Start Training") if mode == "Train" else None
        test_button = st.button("Run Test") if mode == "Test" else None
    
    global device
    device = torch.device('cuda' if st.session_state.use_cuda and torch.cuda.is_available() else 'cpu')
    st.write(f"Using device: {device}")
    
    random.seed(14)
    np.random.seed(14)
    torch.manual_seed(14)
    if device.type == 'cuda':
        torch.cuda.manual_seed(14)
    
    selected_companies = random.sample(companies, min(st.session_state.num_companies, len(companies)))
    
    if not selected_companies:
        st.error("Please select at least one company.")
        return
    
    with st.spinner("Loading data..."):
        try:
            (X_price_train, X_text_train, y_class_train, y_reg_train, date_keys_train,
             X_price_val, X_text_val, y_class_val, y_reg_val, date_keys_val) = prepare_data(
                base_path, selected_companies, days=st.session_state.days
            )
            adj, H = load_adj(selected_companies)
            stock_num = len(selected_companies)
            num_samples_train = len(X_price_train)
            num_samples_val = len(X_price_val)
            price_data = load_price_data(base_path, selected_companies)
            st.success(f"Loaded {num_samples_train} training samples, {num_samples_val} validation samples.")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            traceback.print_exc()
            return
    
    X_price_train = X_price_train.to(device)
    y_class_train = y_class_train.to(device)
    y_reg_train = y_reg_train.to(device)
    X_price_val = X_price_val.to(device)
    y_class_val = y_class_val.to(device)
    y_reg_val = y_reg_val.to(device)
    
    # Initialize model
    model = GATStockPredictionModel(
        nfeat=64,
        nhid=64,
        nclass=2,
        dropout=0.38,
        alpha=0.2,
        nheads=8,
        stock_num=stock_num,
        sparse=sparse,
        attention_mode=st.session_state.attention_mode
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training
    if mode == "Train" and train_button:
        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        metrics_container = st.empty()
        loss_plot_container = st.empty()
        acc_plot_container = st.empty()
        profit_plot_container = st.empty()
        sharpe_plot_container = st.empty()
        f1_plot_container = st.empty()
        mcc_plot_container = st.empty()
        val_profit_plot_container = st.empty()
        val_sharpe_plot_container = st.empty()
        price_train_plot_container = st.empty()
        
        loss_history = []
        acc_history = []
        profit_history = []
        sharpe_history = []
        val_f1_history = []
        val_mcc_history = []
        val_profit_history = []
        val_sharpe_history = []
        
        # Plot historical prices for training period
        st.subheader("Historical Prices (Training Period)")
        fig_train_price = go.Figure()
        for company in selected_companies:
            company_dates = price_data[company]['dates']
            company_prices = price_data[company]['features'][:, 2]  # Adjusted close
            train_indices = [np.where(company_dates == date)[0][0] for date in date_keys_train if date in company_dates]
            if train_indices:
                train_dates = [company_dates[idx] for idx in train_indices]
                train_prices = [company_prices[idx] for idx in train_indices]
                fig_train_price.add_trace(go.Scatter(
                    x=train_dates,
                    y=train_prices,
                    mode='lines',
                    name=company
                ))
        fig_train_price.update_layout(
            title="Historical Adjusted Close Prices (Training Period)",
            xaxis_title="Date",
            yaxis_title="Adjusted Close Price",
            xaxis_tickangle=45
        )
        price_train_plot_container.plotly_chart(fig_train_price, key="price_train_plot")
        
        for epoch in range(st.session_state.epochs):
            try:
                loss, acc, profit, sharpe = train(
                    epoch, model, optimizer, scheduler, X_price_train, X_text_train,
                    y_class_train, y_reg_train, adj, H, num_samples_train, device, progress_bar,
                    metrics_container, st.session_state.batch_size, st.session_state.progress_freq
                )
                if loss is not None:
                    loss_history.append(loss)
                    acc_history.append(acc)
                    profit_history.append(profit)
                    sharpe_history.append(sharpe)
                
                if epoch % st.session_state.validate_freq == 0 or epoch == st.session_state.epochs - 1:
                    val_loss, val_acc, val_f1, val_mcc, val_profit, val_sharpe = validate(
                        model, X_price_val, X_text_val, y_class_val, y_reg_val, adj, H, device
                    )
                    val_f1_history.append(val_f1)
                    val_mcc_history.append(val_mcc)
                    val_profit_history.append(val_profit)
                    val_sharpe_history.append(val_sharpe)
                    st.write(
                        f"Validation at Epoch {epoch}: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                        f"F1: {val_f1:.4f}, MCC: {val_mcc:.4f}, Profit: {val_profit:.4f}, Sharpe: {val_sharpe:.4f}"
                    )
                
                if epoch % st.session_state.progress_freq == 0:
                    # Training Loss
                    if loss_history:
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode='lines', name='Train Loss'))
                        fig_loss.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
                        loss_plot_container.plotly_chart(fig_loss, key=f"loss_plot_{epoch}")
                    
                    # Training Accuracy
                    if acc_history:
                        fig_acc = go.Figure()
                        fig_acc.add_trace(go.Scatter(x=list(range(len(acc_history))), y=acc_history, mode='lines', name='Train Accuracy'))
                        fig_acc.update_layout(title="Training Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
                        acc_plot_container.plotly_chart(fig_acc, key=f"acc_plot_{epoch}")
                    
                    # Training Profit
                    if profit_history:
                        fig_profit = go.Figure()
                        fig_profit.add_trace(go.Scatter(x=list(range(len(profit_history))), y=profit_history, mode='lines', name='Train Profit'))
                        fig_profit.update_layout(title="Training Profit", xaxis_title="Epoch", yaxis_title="Profit")
                        profit_plot_container.plotly_chart(fig_profit, key=f"profit_plot_{epoch}")
                    
                    # Training Sharpe Ratio
                    if sharpe_history:
                        fig_sharpe = go.Figure()
                        fig_sharpe.add_trace(go.Scatter(x=list(range(len(sharpe_history))), y=sharpe_history, mode='lines', name='Train Sharpe'))
                        fig_sharpe.update_layout(title="Training Sharpe Ratio", xaxis_title="Epoch", yaxis_title="Sharpe Ratio")
                        sharpe_plot_container.plotly_chart(fig_sharpe, key=f"sharpe_plot_{epoch}")
                    
                    # Validation F1
                    if val_f1_history:
                        fig_f1 = go.Figure()
                        fig_f1.add_trace(go.Scatter(
                            x=list(range(0, len(val_f1_history)*st.session_state.validate_freq, st.session_state.validate_freq)),
                            y=val_f1_history,
                            mode='lines',
                            name='Validation F1'
                        ))
                        fig_f1.update_layout(title="Validation F1 Score", xaxis_title="Epoch", yaxis_title="F1")
                        f1_plot_container.plotly_chart(fig_f1, key=f"f1_plot_{epoch}")
                    
                    # Validation MCC
                    if val_mcc_history:
                        fig_mcc = go.Figure()
                        fig_mcc.add_trace(go.Scatter(
                            x=list(range(0, len(val_mcc_history)*st.session_state.validate_freq, st.session_state.validate_freq)),
                            y=val_mcc_history,
                            mode='lines',
                            name='Validation MCC'
                        ))
                        fig_mcc.update_layout(title="Validation MCC", xaxis_title="Epoch", yaxis_title="MCC")
                        mcc_plot_container.plotly_chart(fig_mcc, key=f"mcc_plot_{epoch}")
                    
                    # Validation Profit
                    if val_profit_history:
                        fig_val_profit = go.Figure()
                        fig_val_profit.add_trace(go.Scatter(
                            x=list(range(0, len(val_profit_history)*st.session_state.validate_freq, st.session_state.validate_freq)),
                            y=val_profit_history,
                            mode='lines',
                            name='Validation Profit'
                        ))
                        fig_val_profit.update_layout(title="Validation Profit", xaxis_title="Epoch", yaxis_title="Profit")
                        val_profit_plot_container.plotly_chart(fig_val_profit, key=f"val_profit_plot_{epoch}")
                    
                    # Validation Sharpe Ratio
                    if val_sharpe_history:
                        fig_val_sharpe = go.Figure()
                        fig_val_sharpe.add_trace(go.Scatter(
                            x=list(range(0, len(val_sharpe_history)*st.session_state.validate_freq, st.session_state.validate_freq)),
                            y=val_sharpe_history,
                            mode='lines',
                            name='Validation Sharpe'
                        ))
                        fig_val_sharpe.update_layout(title="Validation Sharpe Ratio", xaxis_title="Epoch", yaxis_title="Sharpe Ratio")
                        val_sharpe_plot_container.plotly_chart(fig_val_sharpe, key=f"val_sharpe_plot_{epoch}")
            except Exception as e:
                st.error(f"Training stopped at epoch {epoch}: {e}")
                traceback.print_exc()
                break
    
    # Testing
    if mode == "Test" and test_button:
        st.subheader("Test Results")
        with st.spinner("Running test..."):
            try:
                pred_dict, reg_pred_dict, reg_true_dict, prev_price_dict, f1, mcc, attn_dict, test_loss, test_acc, profit, sharpe, li_pred, li_true, li_probs = test_dict(
                    model, X_price_val, X_text_val, y_class_val, y_reg_val, adj, H, date_keys_val, selected_companies, device
                )
                st.metric("Test Loss", f"{test_loss:.4f}")
                st.metric("Test Accuracy", f"{test_acc:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC", f"{mcc:.4f}")
                st.metric("Profit", f"{profit:.4f}")
                st.metric("Sharpe Ratio", f"{sharpe:.4f}")
                
                st.subheader("Predictions")
                pred_df = []
                for date, preds in pred_dict.items():
                    for i, company in enumerate(selected_companies):
                        pred_df.append({
                            "Date": date,
                            "Company": company,
                            "Up_Probability": preds[i][1],
                            "Prediction": "Up" if preds[i][1] > 0.5 else "Down"
                        })
                st.dataframe(pd.DataFrame(pred_df))
                
                # Historical Prices (Testing Period)
                st.subheader("Historical Prices (Testing Period)")
                fig_test_price = go.Figure()
                for company in selected_companies:
                    company_dates = price_data[company]['dates']
                    company_prices = price_data[company]['features'][:, 2]
                    test_indices = [np.where(company_dates == date)[0][0] for date in date_keys_val if date in company_dates]
                    if test_indices:
                        test_dates = [company_dates[idx] for idx in test_indices]
                        test_prices = [company_prices[idx] for idx in test_indices]
                        fig_test_price.add_trace(go.Scatter(
                            x=test_dates,
                            y=test_prices,
                            mode='lines',
                            name=company
                        ))
                fig_test_price.update_layout(
                    title="Historical Adjusted Close Prices (Testing Period)",
                    xaxis_title="Date",
                    yaxis_title="Adjusted Close Price",
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig_test_price, key="price_test_plot")
                
                # Pie Chart: Prediction Distribution
                st.subheader("Prediction Distribution")
                predictions = [row["Prediction"] for row in pred_df]
                up_count = predictions.count("Up")
                down_count = predictions.count("Down")
                fig_pie = go.Figure(data=[
                    go.Pie(labels=["Up", "Down"], values=[up_count, down_count])
                ])
                fig_pie.update_layout(title="Distribution of Up vs. Down Predictions")
                st.plotly_chart(fig_pie, key="pie_chart")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(li_true, li_pred)
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Down", "Up"],
                    y=["Down", "Up"],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    showscale=True
                ))
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted",
                    yaxis_title="Actual"
                )
                st.plotly_chart(fig_cm, key="confusion_matrix")
                
                # ROC Curve
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(li_true, li_probs[:, 1])
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'ROC curve (area = {roc_auc:.2f})'
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='Chance',
                    showlegend=True
                ))
                fig_roc.update_layout(
                    title="Receiver Operating Characteristic (ROC) Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    showlegend=True
                )
                st.plotly_chart(fig_roc, key="roc_curve")
                
                # Price Attention Weights Heatmap
                st.subheader("Attention Weights")
                if attn_dict:
                    first_date = list(attn_dict.keys())[0]
                    price_attn = attn_dict[first_date]['price_attn']
                    fig_attn = go.Figure(data=go.Heatmap(
                        z=price_attn,
                        x=[f"Day {i}" for i in range(price_attn.shape[1])],
                        y=selected_companies,
                        colorscale='Viridis'
                    ))
                    fig_attn.update_layout(title=f"Price Attention Weights for {first_date}")
                    st.plotly_chart(fig_attn, key="attention_plot")
                
                # Predicted vs. Actual Returns (Scatter Plot)
                st.subheader("Predicted vs. Actual Returns")
                all_reg_preds = []
                all_reg_trues = []
                for date in reg_pred_dict.keys():
                    all_reg_preds.extend(reg_pred_dict[date])
                    all_reg_trues.extend(reg_true_dict[date])
                all_reg_preds = np.array(all_reg_preds)
                all_reg_trues = np.array(all_reg_trues)
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=all_reg_trues,
                    y=all_reg_preds,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=8)
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=[min(all_reg_trues.min(), all_reg_preds.min()), max(all_reg_trues.max(), all_reg_preds.max())],
                    y=[min(all_reg_trues.min(), all_reg_preds.min()), max(all_reg_trues.max(), all_reg_preds.max())],
                    mode='lines',
                    name='Ideal',
                    line=dict(color='red', dash='dash')
                ))
                fig_scatter.update_layout(
                    title="Predicted vs. Actual Returns",
                    xaxis_title="Actual Returns",
                    yaxis_title="Predicted Returns"
                )
                st.plotly_chart(fig_scatter, key="scatter_plot")
                
                # Bar Chart: Up Probabilities for a Selected Date
                st.subheader("Up Probabilities")
                selected_date = st.selectbox("Select Date for Up Probabilities", list(pred_dict.keys()))
                if selected_date:
                    up_probs = pred_dict[selected_date][:, 1]
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=selected_companies,
                        y=up_probs,
                        name='Up Probability'
                    ))
                    fig_bar.update_layout(
                        title=f"Up Probabilities on {selected_date}",
                        xaxis_title="Company",
                        yaxis_title="Probability of Up",
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig_bar, key="bar_chart")
                
                # Predicted vs. Actual Prices
                st.subheader("Predicted vs. Actual Prices (Testing Period)")
                for company_idx, company in enumerate(selected_companies):
                    dates = []
                    actual_prices = []
                    predicted_prices = []
                    for date in date_keys_val:
                        if date in reg_pred_dict:
                            prev_price = prev_price_dict[date][company_idx]
                            reg_pred = reg_pred_dict[date][company_idx]
                            reg_true = reg_true_dict[date][company_idx]
                            # Compute predicted price: prev_price * (exp(reg_out) + 1)
                            pred_price = prev_price * (np.exp(reg_pred) + 1)
                            actual_price = prev_price * (np.exp(reg_true) + 1)
                            dates.append(date)
                            actual_prices.append(actual_price)
                            predicted_prices.append(pred_price)
                    
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(
                        x=dates,
                        y=actual_prices,
                        mode='lines',
                        name='Actual Price'
                    ))
                    fig_price.add_trace(go.Scatter(
                        x=dates,
                        y=predicted_prices,
                        mode='lines',
                        name='Predicted Price',
                        line=dict(dash='dash')
                    ))
                    fig_price.update_layout(
                        title=f"Predicted vs. Actual Prices for {company}",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig_price, key=f"price_pred_actual_{company}")
            except Exception as e:
                st.error(f"Error in testing: {e}")
                traceback.print_exc()

if __name__ == '__main__':
    main()