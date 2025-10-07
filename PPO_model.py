import torch

from Nets import Actor_net,Feat_MLP,Critic_net,Theta_MLP_of_skip,Theta_MLP_of_action,SingelHeadSelfAttention
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from copy import deepcopy
from Nets import MultiHeadSelfAttention

class the_policy(nn.Module):
    def __init__(self,configs):
        super(the_policy, self).__init__()
        self.Job_MHSAL = MultiHeadSelfAttention(configs.MHSAN_input_dim_Job,configs.MHSAN_q_dim,configs.MHSAN_head_output_dim_Job,configs.MHSAN_num_heads_Job).to(configs.device)
        self.Machine_MHSAL = MultiHeadSelfAttention(configs.MHSAN_input_dim_Machine, configs.MHSAN_q_dim,configs.MHSAN_head_output_dim_Machine,configs.MHSAN_num_heads_Machine).to(configs.device)
        self.Actor=Actor_net(configs.num_mlp_layers_actor,configs.input_dim_actor,configs.hidden_dim_actor,configs.out_dim_actor).to(configs.device)#用于更新预测
        self.Critic=Critic_net(configs.num_mlp_layers_critic,configs.input_dim_critic,configs.hidden_dim_critic,configs.out_dim_critic).to(configs.device)
        self.Machine_node_SHSA = SingelHeadSelfAttention(configs.input_dim_machine_node,configs.MHSAN_q_dim,configs.output_dim_machine_node).to(configs.device)
        self.device=configs.device
    def get_pi(self,o_m_feats,illegal_mask):
        action_set_scores=self.Actor(o_m_feats).squeeze(-1)
        shield_illegal_action=action_set_scores.masked_fill(~illegal_mask,float('-inf')).view(-1)
        shield_illegal_action = shield_illegal_action - torch.max(shield_illegal_action)
        pi=F.softmax(shield_illegal_action, dim=0)
        #返回池化后的特征，以便后续更新模型
        return pi.detach()
    def get_feat_and_pool(self,feats):
        O_M_Feat_Job = feats[0]
        O_M_Feat_else = feats[2]
        Machine_connect_node_feat = feats[3]
        temp_list=[]
        test_list = []
        for i in range(len(Machine_connect_node_feat)):
            machine_node_i = torch.mean(self.Machine_node_SHSA(Machine_connect_node_feat[i].to(self.device)),dim=0)
            temp_list.append(machine_node_i)
            test_list.append(torch.mean(Machine_connect_node_feat[i],dim=0))
        Machine_connect_node_feat_with_mlp = torch.stack(temp_list)
        O_M_Feat_Machine = torch.cat((feats[1],Machine_connect_node_feat_with_mlp),dim=-1)
        ht_normal_job = F.normalize(O_M_Feat_Job, p=2, dim=-1)
        ht_normal_machine = F.normalize(O_M_Feat_Machine, p=2, dim=-1)
        via_MHSAL_O_M_Feat_Job = self.Job_MHSAL(ht_normal_job)
        via_MHSAL_O_M_Feat_Machine = self.Machine_MHSAL(ht_normal_machine)
        #经过多头注意力后的工序节点和机器节点求平均在拼接，形成全局特征
        pool = torch.concat((torch.mean(via_MHSAL_O_M_Feat_Job.squeeze(0),dim=0),torch.mean(via_MHSAL_O_M_Feat_Machine.squeeze(0),dim=0)),)
        # ht = pool.unsqueeze(0).unsqueeze(0).repeat(*O_M_Feat_else.size()[:2], 1)
        all_feat = torch.cat([via_MHSAL_O_M_Feat_Job.squeeze(0).unsqueeze(1).repeat(1, O_M_Feat_else.size()[1], 1),
                              via_MHSAL_O_M_Feat_Machine.squeeze(0).unsqueeze(0).repeat(O_M_Feat_else.size()[0], 1, 1),
                              O_M_Feat_else,pool.unsqueeze(0).unsqueeze(0).repeat(*O_M_Feat_else.size()[:2], 1)], dim=2)
        test1 = torch.unique(O_M_Feat_Job, dim=0)
        test2 = torch.unique(O_M_Feat_Machine, dim=0)
        test3 = torch.unique(torch.reshape(all_feat,(-1,all_feat.size()[-1])),dim=0)
        test4 = torch.unique(via_MHSAL_O_M_Feat_Job.squeeze(0), dim=0)
        test5 = torch.unique(via_MHSAL_O_M_Feat_Machine.squeeze(0), dim=0)
        return all_feat,pool


    def up_get_pi(self,Action_state,illegal_mask):
        action_set_scores = self.Actor(Action_state).squeeze(-1)
        shield_illegal_action = action_set_scores.masked_fill(~illegal_mask, float('-inf'))
        shield_illegal_action_expand = torch.reshape(shield_illegal_action, ( shield_illegal_action.size()[0], shield_illegal_action.size()[1] * shield_illegal_action.size()[2]))
        pi = F.softmax(shield_illegal_action_expand, dim=1)
        return pi
    def get_action(self,pi,is_sampl=True):
        if is_sampl:
            dist = Categorical(pi)
            order = dist.sample()
            return order, dist.log_prob(order)
        else:
            order = torch.argmax(pi)
            return order,''

class skip_NowTime_policy(nn.Module):
    def __init__(self,configs):
        super(skip_NowTime_policy, self).__init__()
        self.skip_actor = Theta_MLP_of_skip(configs.num_mlp_layers_skip_actor, configs.input_dim_skip_actor, configs.hidden_dim_skip_actor,
                                 configs.out_dim_skip_actor).to(configs.device)
        self.skip_critic = Theta_MLP_of_skip(configs.num_mlp_layers_skip_critic, configs.input_dim_skip_critic,
                                    configs.hidden_dim_skip_critic,
                                    configs.out_dim_skip_critic).to(configs.device)
        self.device = configs.device
    def get_pi_ht(self,state,mask):
        #前一个概率指去往下一步，后一个概率表示不去往下一个时间步的概率
        # mask = [False,False] 或者 [True,False]
        action_set_scores = self.skip_actor(state)
        shield_illegal_action = action_set_scores.masked_fill(~mask, float('-inf'))
        pi = F.softmax(shield_illegal_action, dim=-1)
        return pi
    def up_get_pi(self,state,masks):
        action_sc = self.skip_actor(state)
        shield_illegal_action = action_sc.masked_fill(~masks, float('-inf'))
        pi = F.softmax(shield_illegal_action, dim=-1)
        return pi
    def get_action_ht(self,pi,is_sampl=True):
        if is_sampl:
            dist = Categorical(pi.squeeze())
            order = dist.sample()
            return order, dist.log_prob(order)
        else:
            order = torch.argmax(pi)
            return order,''
class PPO_schedule():
    def __init__(self,configs):
        self.the_policy = the_policy(configs)
        self.ActorLossCoef = configs.ploss_coef  # actor网络的损失折扣系数
        self.CriticLossCoef = configs.vloss_coef  # critic网络的损失折扣系数
        self.entloss_coef = configs.entloss_coef  # 损失熵系数
        self.skip_policy = skip_NowTime_policy(configs)
        self.gamma = configs.gamma  # 奖励折扣系数
        self.eps_clip = configs.eps_clip  # PPO2算法的裁剪系数
        self.k_epochs = configs.k_epochs  # 更新次数
        self.MSE = nn.MSELoss()
        self.policy_optimizer = torch.optim.Adam(self.the_policy.parameters(), lr=configs.lr)
        self.skip_policy_optimizer = torch.optim.Adam(self.skip_policy.parameters(), lr=configs.lr)
        self.device = configs.device
        self.the_policy_old = deepcopy(self.the_policy)
        self.the_policy_old.load_state_dict(self.the_policy.state_dict())
        self.skip_policy_old = deepcopy(self.skip_policy)
        self.skip_policy_old.load_state_dict(self.skip_policy.state_dict())
