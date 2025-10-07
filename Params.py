import argparse

MHSA_q_dim = 4
O_M_feat_Job_num = 10
MHSAN_num_heads_Job = 4
O_M_feat_Job_signal_head_num = 4
Job_MH_out_dim = MHSAN_num_heads_Job*O_M_feat_Job_signal_head_num
machine_node_SHSA_input = 4
machine_node_SHSA_output = 8
O_M_feat_Machine_num = 3
MHSAN_num_heads_Machine = 4
O_M_feat_Machine_signal_head_num = 2
Machine_MH_out_dim = MHSAN_num_heads_Machine*O_M_feat_Machine_signal_head_num
pool_dim = Job_MH_out_dim+Machine_MH_out_dim
O_M_else_feat_num = 5
space_feat_dim = 10
job_num = 15
machine_num = 8


parser = argparse.ArgumentParser(description='Arguments for ppo_fjssp')
parser.add_argument('--device', type=str, default="cuda", help='Training device')
configs = parser.parse_args()
