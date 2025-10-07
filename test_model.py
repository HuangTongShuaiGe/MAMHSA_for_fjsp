from copy import deepcopy

from FJSP_Env import FJSP_Environment

from read_instance import via_jsp_get_Instance_info
import torch
from Params import configs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(configs.device)
from draw_gannt import draw_gantt_via_ListScheduleInfo


def validate_MODEL(v_data,PPO,device):
    FJSP_instance = FJSP_Environment(v_data)
    step_index = 0
    next_time_list = []
    while (1):
        step_index+=1
        O_M_illegal_mask = FJSP_instance.get_mask().to(device)
        # 原始特征
        O_M_Feat_Job = FJSP_instance.return_O_M_Feat_Job().to(device)
        Machine_connect_node_feat = FJSP_instance.connect_with_Machine()
        O_M_Feat_Machine = FJSP_instance.return_O_M_Feat_Machine().to(device)
        O_M_Feat_else = FJSP_instance.return_O_M_else_Feat().to(device)
        space_feat = FJSP_instance.return_space_feat().to(device)
        # 映射后的特征
        with torch.no_grad():
            O_M_feats, pool_state = PPO.the_policy_old.get_feat_and_pool(
                [O_M_Feat_Job, O_M_Feat_Machine, O_M_Feat_else,Machine_connect_node_feat])
            time_step = FJSP_instance.is_time_step()
            if time_step == False:
                gap_time = 0
            else:
                gap_time = time_step[0] - FJSP_instance.now_clock
            skip_state = torch.concat((space_feat, pool_state, torch.tensor(gap_time).unsqueeze(0).to(device)), dim=-1)
            skip_mask = FJSP_instance.get_skip_mask().to(device)
            skip_pi = PPO.skip_policy_old.get_pi_ht(skip_state, skip_mask)
            skip_action, log_skip_p = PPO.skip_policy_old.get_action_ht(skip_pi,False)
        if skip_action == 0:
            # 此处定义动作0为执行去下一时间步
            next_time_list.append(FJSP_instance.now_clock)
            FJSP_instance.to_next_TimeStep()
            next_time_list.append(FJSP_instance.now_clock)
            continue
        else:
            pass
        with torch.no_grad():
            pi = PPO.the_policy_old.get_pi(O_M_feats, O_M_illegal_mask)
            select_order, log_p = PPO.the_policy_old.get_action(pi, False)
            FJSP_instance.do_Action(select_order, False)
        if FJSP_instance.Schedule_done_info:
            break
    print("在以下时刻发生了决策时间转移：",next_time_list)
    draw_gantt_via_ListScheduleInfo(FJSP_instance.Schedule_info_in_Machine,is_save=True,next_times = next_time_list,save_dir=r'gantts')
    print("目前绘制的甘特图保存在：gantts目录下，您可在代码中修改！")