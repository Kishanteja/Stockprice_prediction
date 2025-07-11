import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import os
import json
import time
import random
import argparse
from collections import defaultdict
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

# Step 1: Tweet Embedding Model (Dynamic Tweet Count)
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
                encoded_input = self.tokenizer(batch_tweets, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
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
            print(f"Error in BERT embedding: {e}")
            return torch.zeros((max_tweets, 768), dtype=torch.float32, device=self.bert.device)

# Step 2: Normalize price data (Robust Normalization)
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
    returns = predictions - actuals
    profit = (returns * prices[:, 2]).sum()
    return profit.item()

def calculate_sharpe_ratio(predictions, actuals):
    returns = predictions - actuals
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = mean_return / (std_return + 1e-8)
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

# Step 5: GRU and Attention Modules (Stabilized Attention)
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

# Step 6: Main GAT Model (Clipped Regression Outputs)
class GATStockPredictionModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, stock_num, sparse=False):
        super(GATStockPredictionModel, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
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

        AttentionLayer = SpGraphAttentionLayer if sparse else GraphAttentionLayer
        self.attentions = nn.ModuleList([AttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        self.out_att = AttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, text_input, price_input, adj):
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
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        gat_out = F.elu(self.out_att(x, adj))

        price_normalized = normalize_price_data(price_input[:, -1, :])
        reg_out = [self.regressor[i](torch.cat([ft_vec[i], price_normalized[i]], dim=0)).unsqueeze(0) for i in range(num_stocks)]
        reg_out = torch.cat(reg_out, dim=0)
        reg_out = torch.clamp(reg_out, min=-10.0, max=10.0)

        return (class_out + gat_out, reg_out, 
                torch.stack(price_attn_weights), 
                tweet_attn_weights, 
                torch.stack(combined_attn_weights))

# Step 7: Data Loading Functions (Fix 1: Safe Log-Return, Fix 2: Efficient Tensor)
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

def prepare_data(base_path, companies, days=5):
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
                    # Fix 1: Safe log-return calculation
                    curr_close = company_prices[curr_idx, 2]
                    prev_close = company_prices[prev_idx, 2]
                    if prev_close > 1e-4 and curr_close > 1e-4:
                        ratio = curr_close / prev_close
                        reg_label = np.clip(np.log1p(ratio - 1), -1.0, 1.0)  # Use log1p for stability
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
    
    # Fix 2: Convert lists to NumPy arrays before tensor creation
    y_class = np.array(y_class, dtype=np.int64)
    y_reg = np.array(y_reg, dtype=np.float32)
    return (torch.tensor(np.stack(X_price), dtype=torch.float32), X_text, 
            torch.tensor(y_class, dtype=torch.long), torch.tensor(y_reg, dtype=torch.float32), date_keys)

def load_adj(companies):
    num_stocks = len(companies)
    adj = np.ones((num_stocks, num_stocks)) - np.eye(num_stocks)
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    return torch.FloatTensor(np.array(adj.todense()))

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
def train(epoch, model, optimizer, scheduler, X_price, X_text, y_class, y_reg, adj, args, num_samples):
    model.train()
    optimizer.zero_grad()
    i = np.random.randint(num_samples)
    price_input = X_price[i].to(device)
    text_input = X_text[i]
    class_labels = y_class[i].to(device)
    reg_labels = y_reg[i].to(device)
    
    class_out, reg_out, _, _, _ = model(text_input, price_input, adj)
    
    class_loss = F.cross_entropy(class_out, class_labels)
    reg_loss = F.mse_loss(reg_out.squeeze(), reg_labels)
    total_loss = class_loss + 0.01 * reg_loss
    
    acc_train = accuracy(class_out, class_labels)
    profit = calculate_profit(reg_out.squeeze(), reg_labels, price_input[:, -1, :])
    sharpe = calculate_sharpe_ratio(reg_out.squeeze(), reg_labels)
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    print(f'Epoch: {epoch}, Loss: {total_loss.item():.4f}, Class Loss: {class_loss.item():.4f}, '
          f'Reg Loss: {reg_loss.item():.4f}, Acc: {acc_train.item():.4f}, Profit: {profit:.4f}, Sharpe: {sharpe:.4f}')

def test_dict(model, X_price, X_text, y_class, y_reg, adj, date_keys, companies):
    model.eval()
    pred_dict = {}
    attn_dict = {}
    test_acc = []
    test_loss = []
    li_pred = []
    li_true = []
    li_reg_pred = []
    li_reg_true = []
    li_prices = []
    
    with torch.no_grad():
        for i in range(len(X_price)):
            price_input = X_price[i].to(device)
            text_input = X_text[i]
            class_labels = y_class[i].to(device)
            reg_labels = y_reg[i].to(device)
            date = date_keys[i]
            
            class_out, reg_out, price_attn_w, tweet_attn_w, combined_attn_w = model(text_input, price_input, adj)
            class_out_softmax = F.softmax(class_out, dim=1)
            
            pred_dict[date] = class_out_softmax.cpu().numpy()
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
            
            test_loss.append(loss_test.item())
            test_acc.append(acc_test.item())
    
    li_pred = np.concatenate(li_pred)
    li_true = np.concatenate(li_true)
    li_reg_pred = np.concatenate(li_reg_pred)
    li_reg_true = np.concatenate(li_reg_true)
    li_prices = np.concatenate(li_prices, axis=0)
    
    f1 = f1_score(li_true, li_pred, average='micro')
    mcc = matthews_corrcoef(li_true, li_pred)
    profit = calculate_profit(torch.tensor(li_reg_pred), torch.tensor(li_reg_true), torch.tensor(li_prices))
    sharpe = calculate_sharpe_ratio(torch.tensor(li_reg_pred), torch.tensor(li_reg_true))
    
    print("Test set results:",
          f"loss= {np.array(test_loss).mean():.4f}",
          f"accuracy= {np.array(test_acc).mean():.4f}",
          f"F1 score= {f1:.4f}",
          f"MCC= {mcc:.4f}",
          f"Profit= {profit:.4f}",
          f"Sharpe= {sharpe:.4f}")
    
    return pred_dict, f1, mcc, attn_dict

# Step 10: Main Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--sparse', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--nb_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.38)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=100)
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    base_path = "stocknet-dataset-master"
    companies = sorted([
        f[:-4] for f in os.listdir(os.path.join(base_path, 'price', 'preprocessed')) 
        if f.endswith('.txt')
    ])
    X_price, X_text, y_class, y_reg, date_keys = prepare_data(base_path, companies, days=5)
    adj = load_adj(companies)
    stock_num = len(companies)
    num_samples = len(X_price)
    
    X_price = X_price.to(device)
    y_class = y_class.to(device)
    y_reg = y_reg.to(device)
    adj = adj.to(device)
    
    model = GATStockPredictionModel(
        nfeat=64,
        nhid=args.hidden,
        nclass=2,
        dropout=args.dropout,
        alpha=args.alpha,
        nheads=args.nb_heads,
        stock_num=stock_num,
        sparse=args.sparse
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(args.epochs):
        train(epoch, model, optimizer, scheduler, X_price, X_text, y_class, y_reg, adj, args, num_samples)
    
    print("Optimization Finished!")
    pred_dict, f1, mcc, attn_dict = test_dict(model, X_price, X_text, y_class, y_reg, adj, date_keys, companies)