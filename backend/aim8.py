from tools import *
from backend.base import BaseCodegen
import numpy as np

class aim8(BaseCodegen):
    def __init__(self, require_power_of_2):
        super(aim8, self).__init__(require_power_of_2)
        # TODO: predictor should be defined
        self.predictor = np.array([
            SimConfig.read_row_change_apox/2, # 'pu'
            SimConfig.pu_lat, # 'pu_col'
            SimConfig.read_row_change_apox/2, # 'pu_row_change'
            0, # 'device_reg2buf'
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
            #(SimConfig.write_row_change_apox-SimConfig.BL), # 'host_write'
            0, # 'host_write'
            max(SimConfig.tCCDL, SimConfig.BL/2), # 'host_write_col'
            0, # 'host_write_device_buffer'
            0, # 'host_write_device_buffer_col'
            0, # 'host_write_pu_inbuf'
            0, # 'host_write_pu_inbuf_col'
            SimConfig.BL/2, # 'host_read_mac_reg'
            SimConfig.BL/2, # 'host_write_mac_reg'
        ])

    # TODO: code micro for mm operator, compute using 2 banks
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
                    pu_list, performance_threshold, profile_level=0):
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
            return {}, (row_change_num+pu)/2 * SimConfig.read_row_change_apox +\
                pu_col * SimConfig.pu_lat +\
                      host_write_col * max(SimConfig.tCCDL, SimConfig.BL/2) * pu_m * pu_k +\
                          read_mac * SimConfig.BL/2 + write_mac * SimConfig.BL/2

        #inst_len = 0
        for channel_id in channel_list:
            for rank_id in rank_list:
                device_mask = [(i in device_list) for i in range(SimConfig.de)]
                pu_mask = [(i in pu_list) for i in range(pu_num)]
                tmp_inst_list = []
                # 1. 从Host端读入数据
                # 每个bank，写input_row_num_after_div行，并且
                
                for m_row_id in range(m_row):
                    # print(f"p_id = {os.getpid()}, 进度 = {m_row_id}/{m_row}")
                    m_block_real = m_block if m_row_id < m_row - 1 else m_block_corner
                    for k_row_id in range(k_row):
                        k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
                        # NOTE: 每个input group写的时候是广播写入的，因此大概率来说还是CCDL的
                        for input_group in range(pu_m*pu_k):
                            # only the pu that shares input can work together
                            row_id = k_row_id + m_row_id * k_row
                            bank_per_pu = self.bank_num // pu_num
                            limited_pu_list = pu_list[input_group*pu_l:(input_group+1)*pu_l]
                            bank_list = [i*bank_per_pu+input_bank for i in limited_pu_list]
                            bank_mask = [(i in bank_list) for i in range(self.bank_num)]
                            # 交替bank group写入，这样最大的影响只在于总线，但实际上误差的并不多
                            if profile_level == 0:
                                tmp_inst_list.append(self.create_host_write(
                                    channel_id, rank_id, device_mask, bank_mask,
                                    input_row_offset + row_id, 0, m_block_real * k_block_real, True
                                ))
                        
                        if not self.gen_code:
                            predict_ = np.dot(self.inst_count, self.predictor)
                            outer_loop = m_row * k_row
                            partial = predict_*outer_loop
                            break
                        
                    if not self.gen_code:
                        self.reset_inst_count()
                        break
                # 2. 计算
                outpoint_log = np.zeros((om_block*om_row+om_block_corner-om_block,
                                            (ol_block*ol_row+ol_block_corner-ol_block)*simd_l), dtype=np.bool_)
                # row loop m-k-l, col loop m-l-k (fixed, best for output change)
                if mm_schedule == 'mkl':
                    # row loop m-k-l, col loop m-l-k (fixed, best for input reuse)
                    for m_row_id in range(m_row):
                        # print(f"p_id = {os.getpid()}, 进度 = {m_row_id}/{m_row}")
                        m_block_real = m_block if m_row_id < m_row - 1 else m_block_corner
                        for k_row_id in range(k_row):
                            k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
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
                                            if profile_level <= 1:
                                            # compute
                                                tmp_inst_list.append(self.create_device_pu(
                                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                                    (weight_bank, weight_row_offset + weight_row_id, weight_col_offset), 
                                                    (input_bank, input_row_offset + input_row_id, input_col_offset), # 此处>1，因此没有问题
                                                    col_len, input_rowchange and weight_rowchange
                                                ))
                                            else:
                                                tmp_inst_list.append(self.create_device_pu(
                                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                                    (weight_bank, weight_row_offset, weight_col_offset), 
                                                    (input_bank, input_row_offset + input_row_id, input_col_offset), # 此处>1，因此没有问题
                                                    col_len, False
                                                ))
                                        if profile_level == 0:
                                            tmp_inst_list.append(
                                                # 1. 输出Reg -> Host
                                                self.create_host_read_mac_reg(
                                                    channel_id, rank_id, device_mask, pu_mask
                                                )
                                            )
                                        # # check the command threshold
                                        # if len(tmp_inst_list) > cmd_left:
                                        #     return None, 0, 0
                                if not self.gen_code:
                                    predict_ = np.dot(self.inst_count, self.predictor)
                                    outer_loop = m_row * k_row * l_row
                                    return {}, predict_*outer_loop+partial

                elif mm_schedule == 'kml':
                    pass
                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                group_id += 1
                cmd_left -= len(tmp_inst_list)
            break
        return tmp_inst_groups, performance_threshold - cmd_left