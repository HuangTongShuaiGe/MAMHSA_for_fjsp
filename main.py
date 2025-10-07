
import torch
from Params import configs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(configs.device)

if __name__ == '__main__':
    from test_model import validate_MODEL
    from read_instance import via_jsp_get_Instance_info
    Policy = torch.load(r'models/Brandimarte/Mk03.pt')
    validate_MODEL(via_jsp_get_Instance_info(r"FJSP/Brandimarte/mk03.fjs"), Policy,device)
