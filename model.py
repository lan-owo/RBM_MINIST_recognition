import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_visible=784, n_hidden=200, k=1, learning_rate=0.01, momentum=0.9, weight_decay=0.0001):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        
        self.register_buffer('W_momentum', torch.zeros(n_visible, n_hidden))
        self.register_buffer('v_bias_momentum', torch.zeros(n_visible))
        self.register_buffer('h_bias_momentum', torch.zeros(n_hidden))
        
    def sample_from_p(self, p):
        return torch.bernoulli(torch.clamp(p, 0, 1))
    
    def visible_to_hidden(self, v):
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return p_h, self.sample_from_p(p_h)
    
    def hidden_to_visible(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return p_v, self.sample_from_p(p_v)
    
    def contrastive_divergence(self, v0):
        """对比散度算法"""
        # 正向传播
        ph0, h0 = self.visible_to_hidden(v0)
        
        # Gibbs采样
        v_k = v0
        for _ in range(self.k):
            _, h_k = self.visible_to_hidden(v_k)
            p_vk, v_k = self.hidden_to_visible(h_k)
        
        # 负相
        ph_k, _ = self.visible_to_hidden(v_k)
        
        return v0, h0, v_k, ph_k
    
    def reconstruction_error(self, v):
        """计算重构误差"""
        p_h, h = self.visible_to_hidden(v)
        p_v, _ = self.hidden_to_visible(h)
        return torch.mean((v - p_v) ** 2)
    
    def update_parameters(self, v0, h0, v_k, ph_k, batch_size):
        """更新参数"""
        with torch.no_grad():
            positive_grad = torch.matmul(v0.t(), h0)
            negative_grad = torch.matmul(v_k.t(), ph_k)
            
            # 计算梯度并应用动量
            dw = (positive_grad - negative_grad) / batch_size - self.weight_decay * self.W
            dv_bias = torch.mean(v0 - v_k, dim=0)
            dh_bias = torch.mean(h0 - ph_k, dim=0)
            
            self.W_momentum = self.momentum * self.W_momentum + self.learning_rate * dw
            self.v_bias_momentum = self.momentum * self.v_bias_momentum + self.learning_rate * dv_bias
            self.h_bias_momentum = self.momentum * self.h_bias_momentum + self.learning_rate * dh_bias
            
            # 更新参数
            self.W.data += self.W_momentum
            self.v_bias.data += self.v_bias_momentum
            self.h_bias.data += self.h_bias_momentum
            
    def update_parameters2(self, v0, h0, v_k, ph_k, batch_size):
        """更新参数（不使用 momentum 与 weight_decay）"""
        with torch.no_grad():
            positive_grad = torch.matmul(v0.t(), h0)
            negative_grad = torch.matmul(v_k.t(), ph_k)
            
            # 梯度估计（均值）
            dw = (positive_grad - negative_grad) / float(batch_size)
            dv_bias = torch.mean(v0 - v_k, dim=0)
            dh_bias = torch.mean(h0 - ph_k, dim=0)
            
            # 直接应用学习率更新（in-place）
            self.W.add_(self.learning_rate * dw)
            self.v_bias.add_(self.learning_rate * dv_bias)
            self.h_bias.add_(self.learning_rate * dh_bias)