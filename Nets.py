
from torch import nn
import torch
import torch.nn.functional as F
class Theta_MLP_of_skip(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(Theta_MLP_of_skip, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                # torch.relu()
                h = torch.tanh(self.linears[layer](h))
                # h = F.leaky_relu(self.linears[layer](h))
                # h = self.linears[layer](h)
            return self.linears[self.num_layers - 1](h)


class Theta_MLP_of_action(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(Theta_MLP_of_action, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                # torch.relu()
                h = torch.tanh(self.linears[layer](h))
                # h = F.leaky_relu(self.linears[layer](h))
                # h = self.linears[layer](h)
            return self.linears[self.num_layers - 1](h)



class test_Feat_MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(test_Feat_MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)













class Feat_MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(Feat_MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)
class Actor_net(nn.Module):
    def __init__(self,layers_num,input_dim,hidden_dim,output_dim):
        super(Actor_net, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = layers_num

        if layers_num < 1:
            raise ValueError("number of layers should be positive!")
        elif layers_num == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(layers_num - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''
        pass
    def forward(self,feats):
        if self.linear_or_not:
            # If linear model
            return self.linear(feats)
        else:
            # If MLP
            h = feats
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                # h = torch.tanh(self.linears[layer](h))
                h = F.leaky_relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)
class Critic_net(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(Critic_net, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                # h = torch.tanh(self.linears[layer](h))
                h = F.leaky_relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)



class SingelHeadSelfAttention(nn.Module):
    def __init__(self,dim_in,dim_q,dim_out):
        super(SingelHeadSelfAttention, self).__init__()
        self.W_q = nn.Linear(dim_in, dim_q)
        self.W_k = nn.Linear(dim_in, dim_q)
        self.W_v = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        #x dim = [m,dim_in]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        k_T = k.transpose(-2,-1)
        alpha = torch.softmax(torch.matmul(q,k_T),dim=-1)
        c = torch.matmul(alpha,v)
        return F.leaky_relu(c)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, MHSAN_input_dim,q_dim,MHSAN_head_output_dim,MHSAN_num_heads):
        '''
        :param input_dim: 输入特征维度
        :param head_output_dim: 单头输出维度
        :param num_heads: 头数
        '''
        super(MultiHeadSelfAttention, self).__init__()
        self.MHSAN_num_heads = MHSAN_num_heads
        self.MultiHeadSelfAttentionLayers = nn.ModuleList()
        for i in range(MHSAN_num_heads):
            self.MultiHeadSelfAttentionLayers.append(SingelHeadSelfAttention(MHSAN_input_dim,q_dim,MHSAN_head_output_dim))
    def forward(self, x):
        # x dim = [m,dim_in]
        c_list = []
        for i in range(self.MHSAN_num_heads):
            singel_c = self.MultiHeadSelfAttentionLayers[i](x)
            c_list.append(singel_c)
        return torch.cat(c_list,dim=-1)































