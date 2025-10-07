#离散时钟调度环境


from read_instance import via_jsp_get_Instance_info
import numpy as np
import torch
from copy import deepcopy
from draw_gannt import draw_gantt_via_ListScheduleInfo

class FJSP_Environment():
    #只调度一个算例的环境
    def __init__(self,data):
        #处理算例，使得机器数等于每个工件的工序数
        self.now_clock=0#系统当前所处时间，时钟
        self.Job_num=data['Job_num']
        self.Mchine_num=data['Machine_num']
        self.Jobs_opretion_num = data['Jobs_operation_num']
        self.Job_Mchine_time_info=torch.tensor(data['J_M_time_info'])
        self.Job_Mchine_bool_info=torch.tensor(data['J_M_bool_info'])
        self.Machine_release_time = torch.zeros((self.Mchine_num,))
        self.Job_release_time=torch.zeros((self.Job_num,))
        self.Job_done_info=torch.full((self.Job_num,),False)#当前工件是否已经完工
        self.Schedule_done_info=False
        self.Job_current_undressed=torch.full((self.Job_num,),0)#当前哪个工序未完成
        self.Job_begain_index=torch.full((self.Job_num,),0)
        begain_index = 0
        for i in range(self.Job_num):
            self.Job_begain_index[i]=begain_index
            begain_index += self.Jobs_opretion_num[i]

        self.Job_can_p_num = torch.sum(self.Job_Mchine_bool_info,dim=-1)
        self.Job_done_index = self.Job_begain_index + (torch.tensor(self.Jobs_opretion_num)-1)

        sum_time_in_per_Machine = torch.sum(
            torch.where(self.Job_Mchine_time_info == -1, torch.tensor(0), self.Job_Mchine_time_info), dim=1)
        # non_neg_one_count = torch.sum(self.Job_Mchine_time_info != -1, dim=1)
        self.Machine_set_len=torch.sum(self.Job_Mchine_time_info != -1, dim=1)
        self.Machine_set_mean = sum_time_in_per_Machine / self.Machine_set_len  # 每个工序在其原加工集上的平均加工时间
        action_num = torch.nonzero(self.Machine_set_mean).size(0)
        self.all_mean_time = torch.sum(torch.flatten(self.Machine_set_mean))/action_num
        self.skip_mask = torch.tensor([True,False])
        self.current_unschedule_operation_index=torch.full((sum(self.Jobs_opretion_num),),True)
        self.makespan = 0#测试时需要他
        self.Mchine_proces_num = torch.full(self.Machine_release_time.size(), self.now_clock)#机器调度工序数量
        self.Schedule_info_in_Machine = [[] for _ in range(self.Mchine_num)]
        self.Machine_set_min_value, _ = torch.min(
            torch.where(self.Job_Mchine_time_info == -1, torch.tensor(float('inf')), self.Job_Mchine_time_info), dim=1)
        self.Machine_set_max_value, _ = torch.max(self.Job_Mchine_time_info, dim=1)
        self.Machine_set_max_value = self.Machine_set_max_value.to(torch.float32)
        def caculate(a,b,c):
            # 根据索引将c分段并计算每段的平均值
            segments = []
            for start, end in zip(a, b):
                segment = c[start:end + 1]
                segment_mean = segment.mean().item()
                segments.append(segment_mean)
            return segments
        mean_mean_job = torch.tensor(caculate(self.Job_begain_index,self.Job_done_index,self.Machine_set_mean),dtype=torch.float32)
        mean_min_job = torch.tensor(caculate(self.Job_begain_index, self.Job_done_index, self.Machine_set_min_value),dtype=torch.float32)
        mean_max_job = torch.tensor(caculate(self.Job_begain_index, self.Job_done_index, self.Machine_set_max_value),dtype=torch.float32)
        self.Job_done_time = torch.full((self.Job_num,),0.0)
        for i in range(self.Job_num):
            self.Job_done_time[i] = torch.sum(self.Machine_set_mean[self.Job_begain_index[i]:self.Job_done_index[i]+1])
        self.operation_min_select = 0
    def do_Action(self,select_order_number,is_save_gante=False):
        #执行动作
        #更改特征-->更改特征依赖变量 #特征依赖变量：工件待加工工序、工件施放时间、当前调度时刻、机器释放时间、已调度到机器上的工序数
        #更改当前调度信息
        select_Job=select_order_number//self.Mchine_num#选择的工件
        select_Mchine = select_order_number % self.Mchine_num  # 选择的机器
        select_Job_operation=deepcopy(self.Job_current_undressed[select_Job])#这个工件的第几个工序
        select_Job_operation_index=int(self.Job_begain_index[select_Job] + select_Job_operation)
        select_Job_operation_in_Mchine_ProcessTime=int(self.Job_Mchine_time_info[select_Job_operation_index][int(select_Mchine)])
        select_Job_operation_name = 'O' + str(int(select_Job) + 1) + ',' + str(int(select_Job_operation)+1)
        #更改特征依赖变量、调度信息更改
        if select_Job_operation_index==self.Job_done_index[select_Job]:
            #如果是最后一道工序，那么这个工件就算是调度完成了
            self.Job_done_info[select_Job]=True
        else:
            self.Job_current_undressed[select_Job]+=1#工序前进，待调度工序+1  工件待加工工序更改
        self.Job_release_time[select_Job]=self.now_clock+select_Job_operation_in_Mchine_ProcessTime  #工件施放时间更改
        self.Machine_release_time[select_Mchine] = self.now_clock + select_Job_operation_in_Mchine_ProcessTime #机器释放时间更改
        self.Mchine_proces_num[select_Mchine] += 1  # 已调度到机器上的工序数 更改
        #当前工序已经被调度
        self.current_unschedule_operation_index[select_Job_operation_index] = False
        if all(self.Job_done_info):#如此便是所有工件都调度完毕
            self.Schedule_done_info=True
        # 把调度信息记录下来
        data_index = [select_Job_operation_name, int(select_Job) + 1, self.now_clock,
                      select_Job_operation_in_Mchine_ProcessTime]
        self.Schedule_info_in_Machine[select_Mchine].append(data_index)
        if is_save_gante:#是否绘制当前调度信息的图片
            draw_gantt_via_ListScheduleInfo(self.Schedule_info_in_Machine, is_save=True, is_show=False)
        if not (torch.any(self.get_mask()).item()) or False:  # 动作空间为空，跳到下一个时间步  当前调度时刻更改
            if self.to_next_TimeStep()==False:
                # print('最后一步，或者有误！')
                pass
        self.makespan = max(self.Machine_release_time)#完成当前所有调度的时间
        now_v_make_span ,_= torch.max(self.Job_done_time,dim=-1)
        # 预估完成时间发生变化
        #当前选择工件的预估完工时间发生变化，这个工件的完工时间应为当前工序的结束时间+后续工序平均加工时间之和
        #工件的完工时间应为当前工序的结束时间+后续工序平均加工时间之和
        self.Job_done_time[select_Job] = self.Job_release_time[select_Job]+torch.sum(self.Machine_set_mean[select_Job_operation_index+1:self.Job_done_index[select_Job]+1],dim=-1)
        next_v_make_span ,_= torch.max(self.Job_done_time,dim=-1)
        reward_makespan = now_v_make_span - next_v_make_span
        if self.Machine_set_min_value[select_Job_operation_index] == select_Job_operation_in_Mchine_ProcessTime:
            if select_Job_operation_in_Mchine_ProcessTime != 0:
                self.operation_min_select+=1#是否选择了最佳选择
        return reward_makespan,self.Machine_set_mean[select_Job_operation_index] - select_Job_operation_in_Mchine_ProcessTime
    def return_O_M_Feat_Job(self):
        Job_current_GX = self.Job_begain_index + self.Job_current_undressed
        # 特征：当前工件施放时间
        O_M_feat_Job_1 = deepcopy(self.Job_release_time)
        # 特征：当前工件工序的平均加工时间
        O_M_feat_Job_2 = deepcopy(self.Machine_set_mean[Job_current_GX])
        # 特征：当前工件工序的最小加工时间
        O_M_feat_Job_3 = deepcopy(self.Machine_set_min_value[Job_current_GX])
        # 特征：工件当前多久释放
        gap = self.Job_release_time - torch.tensor(self.now_clock).repeat(self.Job_num, )
        O_M_feat_Job_4 = torch.where(gap < 0, torch.tensor(0), gap)
        # 特征：工件还剩多少个未加工
        O_M_feat_Job_5 = torch.tensor(self.Jobs_opretion_num) - self.Job_current_undressed
        # 后续工序的特征。目的在于区分开不同工件的当前加工工序
        # 可加工机器数，最大、平均、最小加工时间，
        is_end = Job_current_GX == self.Job_done_index
        next_index = Job_current_GX + 1
        next_index[is_end] = 0
        can_use_num_of_next = self.Job_can_p_num[next_index]
        can_use_num_of_next[is_end] = 0
        max_time_of_next = self.Machine_set_max_value[next_index]
        max_time_of_next[is_end] = 0
        min_time_of_next = self.Machine_set_min_value[next_index]
        min_time_of_next[is_end] = 0
        mean_time_of_next = self.Machine_set_mean[next_index]
        min_time_of_next[is_end] = 0
        O_M_feat_Job_6 = self.Job_can_p_num[Job_current_GX]
        O_M_feat_Job_7 = can_use_num_of_next
        O_M_feat_Job_8 = max_time_of_next
        O_M_feat_Job_9 = min_time_of_next
        O_M_feat_Job_10 = mean_time_of_next
        feats = torch.cat(
            [O_M_feat_Job_1.unsqueeze(1), O_M_feat_Job_2.unsqueeze(1), O_M_feat_Job_3.unsqueeze(1), O_M_feat_Job_4.unsqueeze(1), O_M_feat_Job_5.unsqueeze(1),
                     O_M_feat_Job_6.unsqueeze(1),O_M_feat_Job_7.unsqueeze(1),O_M_feat_Job_8.unsqueeze(1),O_M_feat_Job_9.unsqueeze(1),O_M_feat_Job_10.unsqueeze(1)], dim=1)
        return feats
    def return_O_M_Feat_Machine(self):
        # 特征：机器已经调度工序数量
        O_M_feat_Machine_1 = deepcopy(self.Mchine_proces_num)
        # 特征：机器施放时间
        O_M_feat_Machine_2 = deepcopy(self.Machine_release_time)
        # 特征：机器多久可用
        gap = self.Machine_release_time - torch.tensor(self.now_clock).repeat(self.Mchine_num, )
        O_M_feat_Machine_3 = torch.where(gap < 0, torch.tensor(0), gap)
        feats = torch.cat(
            [O_M_feat_Machine_1.unsqueeze(1), O_M_feat_Machine_2.unsqueeze(1), O_M_feat_Machine_3.unsqueeze(1)], dim=1)
        return feats
    def connect_with_Machine(self):
        Machine_connect_node_feat = []
        for i in range(self.Mchine_num):
            can_to_machine_job_index = torch.where(self.Job_Mchine_bool_info[:, i] == True)[0]
            if can_to_machine_job_index.numel() == 0:
                #没有工序可以加工于当前机器
                Machine_connect_node_feat.append(torch.tensor([[0,0,0,0]]).to(torch.float32))
                continue
            is_shchedule_i = self.current_unschedule_operation_index[can_to_machine_job_index].to(torch.float32)
            in_i_time = self.Job_Mchine_time_info[:, i][can_to_machine_job_index].to(torch.float32)
            min_time_i = self.Machine_set_min_value[can_to_machine_job_index]
            mean_time_i = self.Machine_set_mean[can_to_machine_job_index]
            feat_i = torch.cat((is_shchedule_i.unsqueeze(-1), in_i_time.unsqueeze(-1), min_time_i.unsqueeze(-1),
                                mean_time_i.unsqueeze(-1)), dim=-1)
            Machine_connect_node_feat.append(feat_i)
        return Machine_connect_node_feat
    def return_O_M_else_Feat(self):
        Job_current_undressed_index = self.Job_current_undressed + self.Job_begain_index
        # 特征：在O_i_j在M上的加工时间
        O_M_time = self.Job_Mchine_time_info[Job_current_undressed_index]
        mask = self.get_mask()
        O_M_time[~mask] = 999
        Job_release_time = self.Job_release_time.unsqueeze(-1).repeat(1,self.Mchine_num)
        Machine_release_time = self.Machine_release_time.unsqueeze(0).repeat(self.Job_num, 1)
        #特征：最早可用时间
        canuse_time = torch.max(Job_release_time,Machine_release_time)
        # 特征：最小加工时间
        min_time = self.Machine_set_min_value[Job_current_undressed_index].unsqueeze(-1).repeat(1,self.Mchine_num)
        # 特征：最小加工时间和当前加工时间的差值
        gap_time = O_M_time - min_time
        # 特征：工件还剩多少个未加工
        else_GX_num = (torch.tensor(self.Jobs_opretion_num) - self.Job_current_undressed).unsqueeze(-1).repeat(1,self.Mchine_num)
        end_time = canuse_time + O_M_time
        else_feat = torch.concat((O_M_time.unsqueeze(-1),canuse_time.unsqueeze(-1),gap_time.unsqueeze(-1),else_GX_num.unsqueeze(-1),end_time.unsqueeze(-1)),dim=-1)
        return else_feat


    def return_space_feat(self):
        #####针对是否去下一步的一些状态设置
        # 1，工件可用率
        space_feat_1 = torch.sum(torch.where(self.Job_release_time > self.now_clock, 0, 1), dim=0) / self.Job_num
        # 2.工件平均施放时间
        release_gap = self.Job_release_time - torch.full_like(self.Job_release_time, self.now_clock)
        space_feat_2 = torch.mean(torch.where((release_gap) <= 0, 0, release_gap), dim=0)
        # 3.机器可利用率
        space_feat_3 = torch.sum(torch.where(self.Machine_release_time > self.now_clock, 0, 1),
                                      dim=0) / self.Mchine_num
        # 4.机器平均施放时间
        release_gap = self.Machine_release_time - torch.full_like(self.Machine_release_time, self.now_clock)
        space_feat_4 = torch.mean(torch.where((release_gap) <= 0, 0, release_gap), dim=0)
        # 对于当前可加工工件
        mask = self.get_mask()
        can_use_job_index = torch.where(torch.sum(mask, dim=-1) > 0)[0]
        gx_index = self.Job_begain_index[can_use_job_index] + self.Job_current_undressed[can_use_job_index]
        gx_jm_info = self.Job_Mchine_time_info[gx_index]
        gx_jm_info_now = torch.where(mask[can_use_job_index] == False, float('inf'), gx_jm_info)
        can_use_job_now_min, _ = torch.min(gx_jm_info_now, dim=-1)
        if can_use_job_index.size()[0] == 0:
            can_use_job_num = 1
        else:
            can_use_job_num = can_use_job_index.size()[0]
        # 当前可得到最优解的工序比率
        space_feat_5 = torch.sum(torch.where(can_use_job_now_min == self.Machine_set_min_value[gx_index], 1, 0),
                                      dim=0) / can_use_job_num
        if self.is_time_step() == False:
            # 没有下一步
            next_mask = self.get_mask()  # 同当前mask一致
        else:
            next_mask = self.get_mask(False, self.is_time_step()[0])
        gx_jm_info_next = torch.where(next_mask[can_use_job_index] == False, float('inf'), gx_jm_info)
        can_use_job_next_min, _ = torch.min(gx_jm_info_next, dim=-1)
        # 下一时刻得到最优解的比率
        space_feat_6 = torch.sum(torch.where(can_use_job_next_min == self.Machine_set_min_value[gx_index], 1, 0),
                                      dim=0) / can_use_job_num
        # 下一时刻可以得到更好解的比率
        space_feat_7 = torch.sum(torch.where(can_use_job_next_min < can_use_job_now_min, 1, 0),
                                      dim=0) / can_use_job_num
        # 下一时刻这些工件的‘期望’
        space_feat_8 = torch.sum((can_use_job_now_min - can_use_job_next_min), dim=0)
        # 对于当前时刻不能被加工的工件
        not_be_use_job_index = torch.where(torch.sum(mask, dim=-1) <= 0)[0]
        not_be_use_gx_index = self.Job_begain_index[not_be_use_job_index] + self.Job_current_undressed[
            not_be_use_job_index]
        not_be_use_gx_jm_info_next = torch.where(next_mask[not_be_use_job_index] == False, float('inf'),
                                                 self.Job_Mchine_time_info[not_be_use_gx_index])
        not_be_use_job_next_min, _ = torch.min(not_be_use_gx_jm_info_next, dim=-1)
        if not_be_use_job_index.size()[0] == 0:
            not_be_use_job_num = 1
        else:
            not_be_use_job_num = not_be_use_job_index.size()[0]

        # 这些工件在下一时刻得到最优解的比率
        space_feat_9 = torch.sum(
            torch.where(not_be_use_job_next_min == self.Machine_set_min_value[not_be_use_gx_index], 1, 0),
            dim=0) / not_be_use_job_num
        # 下一时刻其可被加工的比率
        space_feat_10 = torch.sum(torch.where(torch.sum(next_mask[not_be_use_job_index], dim=-1) > 0, 1, 0),
                                       dim=0) / not_be_use_job_num
        feats = torch.concat((space_feat_1.unsqueeze(0),space_feat_2.unsqueeze(0),space_feat_3.unsqueeze(0),space_feat_4.unsqueeze(0),
                              space_feat_5.unsqueeze(0),space_feat_6.unsqueeze(0),space_feat_7.unsqueeze(0),space_feat_8.unsqueeze(0),
                              space_feat_9.unsqueeze(0),space_feat_10.unsqueeze(0)),dim=-1)
        return feats
    def to_next_TimeStep(self):
        #如果当前动作空间为0，找到下一个动作空间不为0的时间步
        #机器的释放时间作为时间步选项
        next_steps = self.is_time_step()
        if next_steps == False:
            return False
        for time_step in next_steps:
            self.now_clock=time_step
            if (torch.any(self.get_mask()).item()):
                break
    def is_time_step(self):
        #是否有下一时间步
        #机器的释放时刻就是需要抉择的时刻
        machine_release_time = deepcopy(self.Machine_release_time)
        machine_release_time = sorted(
            set([i for i in machine_release_time.tolist() if i > self.now_clock]))  # 找大于当前时间的时间步
        if len(machine_release_time)==0:
            return False
        else:
            return machine_release_time
    def get_mask(self,is_Now = True,set_time = 0):
        #根据当前时间，机器释放时间，工件释放时间，以及工序可加工机器，已经完工的工件，——>得到需要屏蔽的动作对
        if is_Now:
            current_time = deepcopy(self.now_clock)
        else:
            current_time = set_time

        Mchine_release=deepcopy(self.Machine_release_time)
        Job_release=deepcopy(self.Job_release_time)
        Job_done_info=deepcopy(self.Job_done_info)
        Job_Mchine_bool_info=deepcopy(self.Job_Mchine_bool_info)
        #mask大小同动作空间大小一致，对应动作空间每个动作的合法性
        mask1=(torch.full(Mchine_release.size(),current_time)>=Mchine_release).unsqueeze(0).repeat(self.Job_num, 1)
        mask2=(torch.full(Job_release.size(),current_time)>=Job_release).unsqueeze(-1).repeat(1,self.Mchine_num)
        mask3=~Job_done_info.unsqueeze(-1).repeat(1,self.Mchine_num)
        mask4 = Job_Mchine_bool_info[self.Job_begain_index + self.Job_current_undressed]
        mask=mask1&mask2&mask3&mask4
        return mask
    def get_skip_mask(self):
        if self.is_time_step() == False:
            return ~torch.tensor([True,False])
        else:
            return ~torch.tensor([False,False])




