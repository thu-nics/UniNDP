from tools import *
from backend.base import BaseCodegen
import numpy as np

class aim16(BaseCodegen):
    def __init__(self, require_power_of_2):
        super(aim16, self).__init__(require_power_of_2)
        # TODO: predictor should be defined
        self.predictor = np.array([
            0, # 'pu' 
            SimConfig.pu_lat, # 'pu_col'1
            SimConfig.read_row_change_apox, # 'pu_row_change'2
            0, # 'device_reg2buf'3
            0, # 'device_buf2reg'
            0, # 'device_buf2bk'
            0, # 'device_buf2bk_col'
            0, # 'device_bk2buf'
            0, # 'device_bk2buf_col'
            0, # 'device_bk2gb'
            0, # 'device_bk2gb_col'
            0, # 'device_gb2bk'
            0, # 'device_gb2bk_col'
            0, # 'host_read'
            0, # 'host_read_col'
            0, # 'host_write'
            0, # 'host_write_col'
            0, # 'host_write_device_buffer'
            SimConfig.BL/2, # 'host_write_device_buffer_col'
            0, # 'host_write_pu_inbuf'
            0, # 'host_write_pu_inbuf_col'
            SimConfig.BL/2, # 'host_read_mac_reg'
            SimConfig.BL/2, # 'host_write_mac_reg'
        ])

    # TODO: code micro for mm operator, aim global buffer
    def mm_micro(self, mm_schedule, base_group_id,
                    channel_list, rank_list, device_list, pu_num, simd_l,
                    input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset,
                    m_block, k_block, l_block, b_block,
                    m_row, k_row, l_row, b_row, 
                    m_block_corner, k_block_corner, l_block_corner, b_block_corner,
                    om_block, ol_block, ob_block,
                    om_row, ol_row, ob_row,
                    om_block_corner, ol_block_corner, ob_block_corner,
                    pu_m, pu_k, pu_l, pu_b,
                    pu_list, performance_threshold, profile_level = 0):
        tmp_inst_groups = []
        group_id = base_group_id
        cmd_left = performance_threshold
        if not self.gen_code:
            row_change_num = m_row * k_row * l_row
            pu = max(((m_row-1) * m_block + m_block_corner) * k_row * ((l_row-1) * l_block + l_block_corner), 0)
            pu_col = ((m_row-1) * m_block + m_block_corner) * ((k_row-1) * k_block + k_block_corner) * ((l_row-1) * l_block + l_block_corner)
            read_mac = max(((m_row-1) * m_block + m_block_corner) * k_row * ((l_row-1) * l_block + l_block_corner), 0)
            write_mac = max(((m_row-1) * m_block + m_block_corner) * k_row * ((l_row-1) * l_block + l_block_corner), 0)
            host_write_col = ((m_row-1) * m_block + m_block_corner) * ((k_row-1) * k_block + k_block_corner)
            return {}, ((row_change_num) * SimConfig.read_row_change_apox +\
                pu_col * SimConfig.pu_lat +\
                      host_write_col * SimConfig.BL/2 +\
                          read_mac * SimConfig.BL/2 + write_mac * SimConfig.BL/2) * pu_m * pu_k

        #inst_len = 0
        for channel_id in channel_list:
            for rank_id in rank_list:
                device_mask = [i in device_list for i in range(SimConfig.de)]
                if mm_schedule == 'mkl': # m>k>l
                    tmp_inst_list = []
                    # 记录输出点是否被访问过，若曾被访问，则
                    outpoint_log = np.zeros((om_block*om_row+om_block_corner-om_block,
                                                (ol_block*ol_row+ol_block_corner-ol_block)*simd_l), dtype=np.bool_)
                    
                    # row loop m-k-l, col loop m-l-k (fixed, best for input reuse)
                    for m_row_id in range(m_row):
                        # print(f"p_id = {os.getpid()}, 进度 = {m_row_id}/{m_row}")
                        m_block_real = m_block if m_row_id < m_row - 1 else m_block_corner
                        for k_row_id in range(k_row):
                            k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
                            
                            for input_group in range(pu_m*pu_k):
                                # only the pu that shares input can work together
                                limited_pu_list = pu_list[input_group*pu_l:(input_group+1)*pu_l]
                                pu_mask = [(i in limited_pu_list) for i in range(pu_num)]
                                # write new input to global buffer
                                input_row_id = m_row_id * k_row + k_row_id # input row id
                                # 此处是一个从host写入global buffer的指令
                                if profile_level == 0:
                                    tmp_inst_list.append(
                                        self.create_host_write_device_buffer(
                                            channel_id, rank_id, device_mask, 0, k_block_real * m_block_real
                                        )
                                    )
                                if not self.gen_code:
                                    predict_ = np.dot(self.inst_count, self.predictor)
                                    outer_loop = m_row * k_row * pu_m * pu_k
                                    partial = predict_*outer_loop
                                    self.reset_inst_count()
                                    # return {}, predict_*outer_loop, 0
                                for l_row_id in range(l_row):
                                    # consider corner case
                                    l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                                    # get the row id in input & weight
                                    input_row_id = k_row_id + m_row_id * k_row
                                    weight_row_id = l_row_id + k_row_id * l_row
                                    # compute length
                                    col_len = k_block_real
                                    # loop over the block with the row fixed
                                    for m_block_id in range(m_block_real):
                                        for l_block_id in range(l_block_real):
                                            input_col_offset = m_block_id * k_block_real
                                            weight_col_offset = l_block_id * k_block_real
                                            # determine precharge
                                            input_rowchange = (m_block_id == m_block_real - 1)
                                            weight_rowchange = (l_block_id == l_block_real - 1)
                                            # get output id
                                            om_id = m_block_id + m_row_id * m_block
                                            ol_id = l_block_id + l_row_id * l_block
                                            # get l position in output
                                            ol_col_flat_id = ol_id // simd_l
                                            ol_col_id = ol_col_flat_id % ol_block
                                            ol_row_id = ol_col_flat_id // ol_block
                                            # get l position in output
                                            om_col_flat_id = om_id
                                            om_col_id = om_col_flat_id % om_block
                                            om_row_id = om_col_flat_id // om_block
                                            # get the col & row of output point
                                            o_col_id = om_col_id * ol_block + ol_col_id
                                            o_row_id = om_row_id * ol_row + ol_row_id
                                            # 用 host 读出并写入所有device中各16个计算单元的输出Reg
                                            if outpoint_log[om_id, ol_id]:
                                                if profile_level == 0:
                                                    tmp_inst_list.append(
                                                        # 2. Host -> 输出Reg
                                                        self.create_host_write_mac_reg(
                                                            channel_id, rank_id, device_mask, pu_mask
                                                        )
                                                    )
                                            outpoint_log[om_id, ol_id] = True
                                            for device_id in device_list:
                                                # compute
                                                if profile_level <= 1:
                                                    tmp_inst_list.append(self.create_device_pu(
                                                        channel_id, rank_id, device_id, pu_num, pu_mask, 
                                                        (weight_bank, weight_row_offset + weight_row_id, weight_col_offset), 
                                                        (weight_bank, 1, input_col_offset), # 此处>1，因此没有问题
                                                        col_len, input_rowchange and weight_rowchange
                                                    ))
                                                else:
                                                    tmp_inst_list.append(self.create_device_pu(
                                                        channel_id, rank_id, device_id, pu_num, pu_mask, 
                                                        (weight_bank, weight_row_offset, weight_col_offset), 
                                                        (weight_bank, 1, input_col_offset), # 此处>1，因此没有问题
                                                        col_len, False
                                                    ))
                                            if profile_level == 0:
                                                tmp_inst_list.append(
                                                    # 1. 输出Reg -> Host
                                                    self.create_host_read_mac_reg(
                                                        channel_id, rank_id, device_mask, pu_mask
                                                    )
                                                )
                                            # check the command threshold'
                                    if not self.gen_code:
                                        predict_ = np.dot(self.inst_count, self.predictor)
                                        outer_loop = m_row * k_row * l_row * pu_m*pu_k
                                        return {}, predict_*outer_loop+partial

                    tmp_inst_groups.append((group_id, [], tmp_inst_list))
                    cmd_left -= len(tmp_inst_list)
                    group_id += 1

                elif mm_schedule == 'mlk': # 此时输出切换的次数比较少
                    pass

            break # FIXME: 考虑到仿真的问题，暂时只生成一个channel的指令
        
        return tmp_inst_groups, performance_threshold - cmd_left
    
    def elewise_micro(self, mm_schedule, base_group_id,
                      channel_list, rank_list, device_list, pu_num, simd_l,
                    input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset,
                    m_block, k_block, l_block, b_block,
                    m_row, k_row, l_row, b_row, 
                    m_block_corner, k_block_corner, l_block_corner, b_block_corner,
                    om_block, ol_block, ob_block,
                    om_row, ol_row, ob_row,
                    om_block_corner, ol_block_corner, ob_block_corner,
                    pu_m, pu_k, pu_l, pu_b,
                    pu_list, performance_threshold):
        tmp_inst_groups = []
        group_id = base_group_id
        cmd_left = performance_threshold
        # only k counts
        for channel_id in channel_list:
            for rank_id in rank_list:
                device_mask = [i in device_list for i in range(SimConfig.de)]
                pu_mask = [True for i in range(len(pu_list))] 
                if mm_schedule == 'mkl': # m>k>l
                    tmp_inst_list = []
                    # 记录输出点是否被访问过，若曾被访问，则
                    for k_row_id in range(k_row):
                        k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
                        col_len = k_block_real
                        for device_id in device_list:
                            # compute
                            tmp_inst_list.append(self.create_device_pu(
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (input_bank, weight_row_offset + k_row_id, 0), 
                                (weight_bank, weight_row_offset + k_row_id, 0), # 此处>1，因此没有问题
                                col_len, True
                            ))
                        # check the command threshold'
                        if not self.gen_code:
                            predict_ = np.dot(self.inst_count, self.predictor)
                            outer_loop =  k_row
                            return {}, predict_ * outer_loop
                        
                    tmp_inst_groups.append((group_id, [], tmp_inst_list))
                    cmd_left -= len(tmp_inst_list)
                    group_id += 1

                elif mm_schedule == 'mlk': # 此时输出切换的次数比较少
                    pass

            break # FIXME: 考虑到仿真的问题，暂时只生成一个channel的指令
        
        return tmp_inst_groups, performance_threshold - cmd_left
    
    def softmax_micro(self, mm_schedule, base_group_id,
                      channel_list, rank_list, device_list, pu_num, simd_l,
                    input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset,
                    m_block, k_block, l_block, b_block,
                    m_row, k_row, l_row, b_row, 
                    m_block_corner, k_block_corner, l_block_corner, b_block_corner,
                    om_block, ol_block, ob_block,
                    om_row, ol_row, ob_row,
                    om_block_corner, ol_block_corner, ob_block_corner,
                    pu_m, pu_k, pu_l, pu_b,
                    pu_list, performance_threshold):
        tmp_inst_groups = []
        group_id = base_group_id
        cmd_left = performance_threshold
        # k, l counts
        # only k counts
        for channel_id in channel_list:
            for rank_id in rank_list:
                device_mask = [i in device_list for i in range(SimConfig.de)]
                pu_mask = [True for i in range(len(pu_list))] 
                if mm_schedule == 'mkl': # m>k>l
                    tmp_inst_list = []
                    for l_row_id in range(l_row):
                        l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                        for k_row_id in range(k_row):
                            k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
                            col_len = k_block_real * l_block_real
                            # device num = 1
                            device_id = 0
                            # compute e^x
                            tmp_inst_list.append(self.create_device_pu(
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (input_bank, input_row_offset + k_row_id, 0), 
                                (weight_bank, weight_row_offset + k_row_id, 0), # 此处>1，因此没有问题
                                col_len, False
                            ))
                            for l_block_id in range(l_block_real):
                                tmp_inst_list.append(self.create_host_write_mac_reg(
                                    channel_id, rank_id, device_mask, pu_mask
                                ))
                                # compute e^x sum
                                tmp_inst_list.append(self.create_device_pu(
                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                    (input_bank, input_row_offset + k_row_id, 0), 
                                    (weight_bank, weight_row_offset + k_row_id, 0), # 此处>1，因此没有问题
                                    k_block_real, l_block_id == l_block_real - 1
                                ))
                                # create host read, collect the sum
                                tmp_inst_list.append(self.create_host_read_mac_reg(
                                    channel_id, rank_id, device_mask, pu_mask
                                ))
                            # check the command threshold'
                            if not self.gen_code:
                                predict_ = np.dot(self.inst_count, self.predictor)
                                outer_loop =  k_row * l_row
                                return {}, predict_ * outer_loop
                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                cmd_left -= len(tmp_inst_list)
                group_id += 1
            break # FIXME: 考虑到仿真的问题，暂时只生成一个channel的指令
        return tmp_inst_groups, performance_threshold - cmd_left