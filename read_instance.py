
import os

def via_jsp_get_Instance_info(jsp_path):
    #此处读取的数据已经处理，机器数等于每个工件的工序数
    import numpy as np
    with open(jsp_path,'r') as f:
        lines=f.readlines()
        first_line=lines[0].split()
        Job_num,Mchine_num =int(first_line[0]),int(first_line[1])
        J_M_info=[]
        Jobs_opeation_num = []
        for i in lines[1:Job_num+1]:
            i_1=[int(x) for x in i.split()]
            J_M_info.append(i_1)
            Jobs_opeation_num.append(int(i.split()[0]))

    J_M_time_info=np.full((sum(Jobs_opeation_num),Mchine_num),-1)



    J_M_index=0
    for i in range(Job_num):
        Jon_i_info=J_M_info[i][1:]
        Jon_i_info_index=0
        while(Jon_i_info_index<len(Jon_i_info)):
            opreation_i_can_chooseNum = Jon_i_info[Jon_i_info_index]#当前工序可选择的机器数量
            the_opreation_list = Jon_i_info[Jon_i_info_index + 1:Jon_i_info_index + opreation_i_can_chooseNum * 2 + 1]
            can_choose_Mchine=[the_opreation_list[i] for i in range(0, len(the_opreation_list), 2)]
            in_Machine_time=[the_opreation_list[i] for i in range(1, len(the_opreation_list), 2)]
            can_choose_Mchine=np.array(can_choose_Mchine)
            in_Machine_time = np.array(in_Machine_time)
            can_choose_Mchine-=1#索引
            J_M_time_info[J_M_index][can_choose_Mchine]=in_Machine_time

            Jon_i_info_index += (opreation_i_can_chooseNum * 2 + 1)
            J_M_index+=1

    J_M_bool_info=np.where(J_M_time_info >= 0, True, False)

    data_set = {}
    data_set['Job_num'] = Job_num
    data_set['Machine_num'] = Mchine_num
    data_set['Jobs_operation_num'] = Jobs_opeation_num
    data_set['J_M_time_info'] = J_M_time_info
    data_set['J_M_bool_info'] = J_M_bool_info


    return data_set



# via_jsp_get_Instance_info(r'D:\HT_project\action_dim_is_gx\FJSP\Brandimarte_Data\Mk01.fjs')