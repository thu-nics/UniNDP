from tools import *
from backend.base import BaseCodegen
import numpy as np
from math import ceil

class upmem(BaseCodegen):
    def __init__(self, require_power_of_2):
        super(upmem, self).__init__(require_power_of_2)
        device_num = SimConfig.de * SimConfig.ra
        rank_num = SimConfig.ra
        self.predictor = np.array([
            0, # 'pu'
            SimConfig.pu_lat/device_num, # 'pu_col'
            (SimConfig.read_row_change_apox-SimConfig.pu_lat)/device_num, # 'pu_row_change'
            1/device_num+1, # 'device_reg2buf'
            1/device_num, # 'device_buf2reg'
            SimConfig.write_row_change_apox/device_num, # 'device_buf2bk' # will auto-precharge
            SimConfig.col_change_apox/device_num, # 'device_buf2bk_col'
            SimConfig.read_row_change_apox/device_num, # 'device_bk2buf' # will auto-precharge
            SimConfig.col_change_apox/device_num, # 'device_bk2buf_col'
            0, # 'device_bk2gb'
            0, # 'device_bk2gb_col'
            0, # 'device_gb2bk'
            0, # 'device_gb2bk_col'
            SimConfig.read_row_change_apox/rank_num, # 'host_read'
            max(SimConfig.tCCDL, SimConfig.BL/2)/rank_num, # 'host_read_col'
            0, # 'host_write'
            0, # 'host_write_col'
            0, # 'host_write_device_buffer'
            0, # 'host_write_device_buffer_col'
            0, # 'host_write_pu_inbuf'
            SimConfig.BL/2/rank_num, # 'host_write_pu_inbuf_col'
            0, # 'host_read_mac_reg'
            0, # 'host_write_mac_reg'
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
                    pu_list, performance_threshold, profile_level=0):
        tmp_inst_groups = []
        cmd_left = performance_threshold
        group_id = base_group_id
        # pu_row_change_lat = self.device_num * SimConfig.read_row_change_apox / SimConfig.col_change_apox
        for channel_id in channel_list:
            for rank_id in rank_list:
                device_mask = [i in device_list for i in range(SimConfig.de)]
                tmp_inst_list = []
                pu_mask = [(i in pu_list) for i in range(pu_num)]
                # output reg / buffer
                self.reset_output_buffer()
                outpoint_log = np.zeros((om_block*om_row+om_block_corner-om_block,
                                            (ol_block*ol_row+ol_block_corner-ol_block)*simd_l), dtype=np.bool_)
                # row loop m-k-l, col loop m-l-k (fixed, best for output change)
                if mm_schedule == 'mkl':
                    for m_row_id in range(m_row):
                        m_block_real = m_block if m_row_id < m_row - 1 else m_block_corner
                        for k_row_id in range(k_row):
                            # consider corner case
                            k_block_real = k_block if k_row_id < k_row - 1 else k_block_corner
                            # compute length
                            col_len = k_block_real
                            # reload inputs
                            # NOTE: lat approx
                            # if first_rank:
                            #     cmd_left -= input_buf_write_lat * k_block_real * m_block_real
                            for input_group in range(pu_m*pu_k):
                                limited_pu_list = pu_list[input_group*pu_l:(input_group+1)*pu_l]
                                # bank_list = [ i * bank_per_pu + input_bank for i in limited_pu_list]
                                # bank_mask = [( i in bank_list ) for i in range(self.bank_num)]
                                limited_pu_mask = [(i in limited_pu_list) for i in range(pu_num)]
                                # device并行的写入pu的输入buffer
                                if profile_level == 0:
                                    tmp_inst_list.append(
                                        self.create_host_write_pu_inbuf(
                                            channel_id, rank_id, device_mask, limited_pu_mask,
                                            0, k_block_real * m_block_real
                                        )
                                    )
                            if not self.gen_code:
                                predict_ = np.dot(self.inst_count, self.predictor)
                                outer_loop = m_row * k_row * len(rank_list)
                                partial = predict_ * outer_loop
                                self.reset_inst_count()
                            for l_row_id in range(l_row):
                                l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                                weight_row_id = l_row_id + k_row_id * l_row
                                for m_block_id in range(m_block_real):
                                    for l_block_id in range(l_block_real):
                                        input_col_offset = m_block_id * k_block_real
                                        weight_col_offset = l_block_id * k_block_real
                                        # determine precharge
                                        input_rowchange = (m_block_id == m_block_real - 1)
                                        weight_rowchange = (l_block_id == l_block_real - 1)
                                        need_rowchange = input_rowchange and weight_rowchange
                                        # if need_rowchange and first_rank:
                                        #     cmd_left -= pu_row_change_lat
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
                                        # simulate the PU buffer, to see if it need change
                                        # need_change, last_col, last_row = self.output_buffer(o_col_id, o_row_id)
                                        # load in result from buffer
                                        for device_id in device_list:
                                            if profile_level == 0:
                                                tmp_inst_list.append(self.create_device_buf2reg(
                                                    channel_id, rank_id, device_id, pu_num, pu_mask, 0
                                                ))
                                        # compute using pu in buf
                                        for device_id in device_list:
                                            if profile_level <= 1:
                                                tmp_inst_list.append(self.create_device_pu(
                                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                                    (weight_bank, weight_row_offset + weight_row_id, weight_col_offset), 
                                                    (weight_bank, 0, input_col_offset), 
                                                    col_len, need_rowchange
                                                ))
                                            else:
                                                tmp_inst_list.append(self.create_device_pu(
                                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                                    (weight_bank, weight_row_offset, weight_col_offset), 
                                                    (weight_bank, 0, input_col_offset), 
                                                    col_len, False
                                                ))
                                        # write back to buffer
                                        for device_id in device_list:
                                            if profile_level == 0:
                                                tmp_inst_list.append(self.create_device_reg2buf(
                                                    channel_id, rank_id, device_id, pu_num, pu_mask, 0
                                                ))
                                        # check the command threshold
                                        # if first_rank_threshold and self.inst_count[1] > cmd_left:
                                        #     return None, 0, 0
                                        if not self.gen_code:
                                            predict_ = np.dot(self.inst_count, self.predictor)
                                            outer_loop = k_row * \
                                                ((m_row - 1)*m_block + m_block_corner) *\
                                                ((l_row - 1)*l_block + l_block_corner) *\
                                                len(rank_list)
                                            # as rank A can change row when rank B occupies the bus, no row change occurs when reading out
                                            read_out_latency =  len(pu_list) * len(rank_list) * \
                                                                ((om_row - 1) * om_block + om_block_corner) * \
                                                                ((ol_row - 1) * ol_block + ol_block_corner) * \
                                                                SimConfig.col_change_apox
                                            return {}, predict_ * outer_loop + partial + read_out_latency
                                            # partial = predict_*outer_loop
                                            

                elif mm_schedule == 'kml':
                    pass
                
                """
                3. reduce the output buffer
                """
                # for each pu, read out its outputs
                output_bank_list = [pu_id * self.bank_num // pu_num + output_bank for pu_id in pu_list]
                for output_bank_id in output_bank_list:
                    for om_row_id in range(om_row):
                        for ol_row_id in range(ol_row):
                            o_row_id = om_row_id * ol_row + ol_row_id
                            col_len = (ol_block if ol_row_id < ol_row - 1 else ol_block_corner) * (om_block if om_row_id < om_row - 1 else om_block_corner)
                            if profile_level == 0:
                                tmp_inst_list.append(
                                    self.create_host_read(
                                        channel_id, rank_id, device_mask, output_bank_id, output_row_offset + o_row_id, 0, col_len, True
                                    )
                                )
                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                group_id += 1
                # cmd_left -= len(tmp_inst_list)
            break
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
        cmd_left = performance_threshold
        group_id = base_group_id
        # pu_row_change_lat = self.device_num * SimConfig.read_row_change_apox / SimConfig.col_change_apox
        for channel_id in channel_list:
            for rank_id in rank_list:
                device_mask = [i in device_list for i in range(SimConfig.de)]
                tmp_inst_list = []
                pu_mask = [(i in pu_list) for i in range(pu_num)]
                # row loop m-k-l, col loop m-l-k (fixed, best for output change)
                if mm_schedule == 'mkl':
                    rw_delay = 0
                    for k_row_id in range(k_row):
                        col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                        # 先load一行进入PU的buffer, DRAM read
                        for device_id in device_list:
                            tmp_inst_list.append(
                                self.create_device_bk2buf(
                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                    (weight_bank, weight_row_offset + k_row_id, 0), (True, 0, col_len), 
                                    True
                                )
                            )
                        # 再用这个计算
                        # compute using pu in buf, DRAM read
                        for device_id in device_list:
                            tmp_inst_list.append(self.create_device_pu(
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (weight_bank, weight_row_offset + k_row_id + k_row, 0), 
                                (weight_bank, 0, 0), 
                                col_len, False
                            ))
                        rw_delay += SimConfig.read_to_write_apox
                        # 结果写回Bank, DRAM write
                        for device_id in device_list:
                            tmp_inst_list.append(
                                self.create_device_buf2bk(
                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                    (weight_bank, weight_row_offset + k_row_id + k_row, 0), 
                                    (False, 0, 0), True
                                )
                            )
                    if not self.gen_code:
                        predict_ = np.dot(self.inst_count, self.predictor)
                        outer_loop = len(rank_list)
                        return {}, predict_ * outer_loop + rw_delay
            
                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                group_id += 1
                cmd_left -= len(tmp_inst_list)
            break
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
        # k = reduce dimension, l = parrallel dimension
        tmp_inst_groups = []
        cmd_left = performance_threshold
        group_id = base_group_id
        # pu_row_change_lat = self.device_num * SimConfig.read_row_change_apox / SimConfig.col_change_apox
        for channel_id in channel_list:
            for rank_id in rank_list:
                device_mask = [i in device_list for i in range(SimConfig.de)]
                tmp_inst_list = []
                pu_mask = [(i in pu_list) for i in range(pu_num)]
                # output reg / buffer
                # NOTE: assume that values to be softmax already in DRAM
                rw_delay = 0
                # 1. compute exp , exp sum
                for l_row_id in range(l_row):
                    for k_row_id in range(k_row):
                        col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                        l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                        # Element-wise exp, RD
                        for device_id in device_list:
                            tmp_inst_list.append(self.create_device_pu( # PU: compute exp, elewise
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                (weight_bank, 0, 0), 
                                col_len * l_block_real, False
                            ))
                        # Reduce to l_block_real, RD
                        for device_id in device_list:
                            tmp_inst_list.append(self.create_device_pu( # PU: compute sum, reduce
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                (weight_bank, 0, 0), 
                                col_len * l_block_real, False
                            ))
                        rw_delay += SimConfig.read_to_write_apox
                        # Exp back to DRAM, WR
                        for device_id in device_list:
                            tmp_inst_list.append(
                                self.create_device_buf2bk(
                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                    (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                    (False, 0, 0), True
                                )
                            )
                # Host read out output buffer
                output_len = ceil((l_block * (l_row-1) + l_block_corner) / self.simd)
                # Host collect sum of exp, reuse write_pu_buf cmd
                for bank_id in range(pu_num):
                    pu_mask = [(i == bank_id) for i in range(pu_num)]
                    tmp_inst_list.append(
                        self.create_host_write_pu_inbuf(
                            channel_id, rank_id, device_mask, pu_mask, 
                            0, output_len
                        )
                    )
                # 2. element-wise
                # Host broadcast the sum of exp = fp16 data
                sum_len = 1
                pu_mask = [True for _ in range(pu_num)]
                self.create_host_write_pu_inbuf(
                    channel_id, rank_id, device_mask, pu_mask, 
                    0, sum_len
                )
                # Element-wise division
                for l_row_id in range(l_row):
                    for k_row_id in range(k_row):
                        col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                        l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                        # Element-wise div, RD
                        for device_id in device_list:
                            tmp_inst_list.append(self.create_device_pu( # PU: compute div, elewise
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                (weight_bank, 0, 0), 
                                col_len * l_block_real, False
                            ))
                        rw_delay += SimConfig.read_to_write_apox
                        # dib result back to DRAM, WR
                        for device_id in device_list:
                            tmp_inst_list.append(
                                self.create_device_buf2bk(
                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                    (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                    (False, 0, 0), True
                                )
                            )
                
                # NOTE: and final result is in the same place as the input

                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                group_id += 1
                cmd_left -= len(tmp_inst_list)

                if not self.gen_code:
                    outer_loop = len(rank_list)
                    predict_ = np.dot(self.inst_count, self.predictor)
                    return {}, (predict_+rw_delay)*outer_loop
            break
        return tmp_inst_groups, performance_threshold - cmd_left

    def layernorm_micro(self, mm_schedule, base_group_id,
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
        # k = reduce dimension, l = parrallel dimension
        tmp_inst_groups = []
        cmd_left = performance_threshold
        group_id = base_group_id
        # pu_row_change_lat = self.device_num * SimConfig.read_row_change_apox / SimConfig.col_change_apox
        for channel_id in channel_list:
            for rank_id in rank_list:
                device_mask = [i in device_list for i in range(SimConfig.de)]
                tmp_inst_list = []
                pu_mask = [(i in pu_list) for i in range(pu_num)]
                # output reg / buffer
                
                # NOTE: assume that values to be softmax already in DRAM
                rw_delay = 0
                # 1. compute sum(x^2), sum(x)
                for l_row_id in range(l_row):
                    for k_row_id in range(k_row):
                        col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                        l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                        # read x to PU inbuf, [optional]
                        # compute x * x element-wise, store in output buffer
                        for device_id in device_list:
                            tmp_inst_list.append(self.create_device_pu( # PU: mul, elewise
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                (weight_bank, 0, 0), 
                                col_len * l_block_real, False
                            ))
                        # compute sum of x * x, reuse data in output buffer
                        for device_id in device_list:
                            tmp_inst_list.append(self.create_device_pu( # PU: sum, reduce
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                (weight_bank, 0, 0), 
                                col_len * l_block_real, False
                            ))
                        # compute sum of x, reuse data in input buffer
                        for device_id in device_list:
                            tmp_inst_list.append(self.create_device_pu( # PU: sum, reduce
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                (weight_bank, 0, 0), 
                                col_len * l_block_real, True
                            ))
                # Host read out output buffer
                output_len = ceil((l_block * (l_row-1) + l_block_corner) * 2 / self.simd)
                # Host collect sum of exp, reuse write_pu_buf cmd
                for bank_id in range(pu_num):
                    pu_mask = [(i == bank_id) for i in range(pu_num)]
                    tmp_inst_list.append(
                        self.create_host_write_pu_inbuf(
                            channel_id, rank_id, device_mask, pu_mask, 
                            0, output_len
                        )
                    )
                # 2. element-wise
                # Host broadcast sum(x^2), sum(x) = fp16 data
                sum_len = ceil (2 / self.simd)
                pu_mask = [True for _ in range(pu_num)]
                self.create_host_write_pu_inbuf(
                    channel_id, rank_id, device_mask, pu_mask, 
                    0, sum_len
                )
                # Element-wise computation
                for l_row_id in range(l_row):
                    for k_row_id in range(k_row):
                        col_len = k_block if k_row_id < k_row - 1 else k_block_corner
                        l_block_real = l_block if l_row_id < l_row - 1 else l_block_corner
                        # Element-wise minus
                        for device_id in device_list:
                            tmp_inst_list.append(self.create_device_pu( # PU: compute minus, elewise
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                (weight_bank, 0, 0), 
                                col_len * l_block_real, False
                            ))
                        # Element-wise div
                        for device_id in device_list:
                            tmp_inst_list.append(self.create_device_pu( # PU: compute div, elewise
                                channel_id, rank_id, device_id, pu_num, pu_mask, 
                                (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                (weight_bank, 0, 0), 
                                col_len * l_block_real, False
                            ))
                        rw_delay += SimConfig.read_to_write_apox
                        # div result back to DRAM
                        for device_id in device_list:
                            tmp_inst_list.append(
                                self.create_device_buf2bk(
                                    channel_id, rank_id, device_id, pu_num, pu_mask, 
                                    (weight_bank, weight_row_offset + l_row_id * k_row + k_row_id, 0), 
                                    (False, 0, 0), True
                                )
                            )
                
                # NOTE: and final result is in the same place as the input

                tmp_inst_groups.append((group_id, [], tmp_inst_list))
                group_id += 1
                cmd_left -= len(tmp_inst_list)

                if not self.gen_code:
                    outer_loop = len(rank_list)
                    predict_ = np.dot(self.inst_count, self.predictor)
                    return {}, predict_ * outer_loop + rw_delay
            break
        return tmp_inst_groups, performance_threshold - cmd_left
    