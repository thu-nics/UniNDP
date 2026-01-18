import abc
from tools import *
import numpy as np

class BaseCodegen(HW_info):
    def __init__(self, require_power_of_2):
        super(BaseCodegen, self).__init__(require_power_of_2)
        self.de_pu_outbuf_col = SimConfig.de_pu_bf / SimConfig.co_w
        # print('de_pu_outbuf_col:', self.de_pu_outbuf_col)
        self.gen_code = False
        self.last_buffer_col = 0
        self.last_buffer_row = 0
        self.buffered = []
        self.total_inst = 0
        self.inst_info = [
            'pu', 'pu_col', 'pu_row_change',
            'device_reg2buf', 'device_buf2reg',
            'device_buf2bk', 'device_buf2bk_col',
            'device_bk2buf', 'device_bk2buf_col',
            'device_bk2gb', 'device_bk2gb_col',
            'device_gb2bk', 'device_gb2bk_col',
            'host_read', 'host_read_col',
            'host_write', 'host_write_col',
            'host_write_device_buffer', 'host_write_device_buffer_col',
            'host_write_pu_inbuf', 'host_write_pu_inbuf_col',
            'host_read_mac_reg', 'host_write_mac_reg',
        ]
        self.inst_count = np.zeros(len(self.inst_info), dtype=np.int64)
        self.predictor = None # predictor should be defined in the child class

    def set_gen(self):
        self.gen_code = True

    def reset_inst_count(self):
        self.inst_count = np.zeros(len(self.inst_info), dtype=np.int64)

    # instruction generater
    def create_device_pu(self, ch_id, ra_id, de_id, pu_num, pu_mask, op1, op2, col_num, auto_precharge):
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 op1:(bank, row_id, col_offset), op2:,       col_num,auto_precharge    
        #return (LEVEL.DE,  OPTYPE.pu,      0,      0,      0,     (8, [True for _ in range(8)]),   (0, 0, 0),                      (1, 0, 0),  64,     False)
        # self.inst_count['pu'] += 1
        self.inst_count[0] += 1
        self.inst_count[1] += col_num
        if auto_precharge:
            # self.inst_count['pu_row_change'] += 1
            self.inst_count[2] += 1
        return (LEVEL.DE,  OPTYPE.pu, ch_id, ra_id, de_id, (pu_num, pu_mask), op1, op2, col_num, auto_precharge)

    def create_device_reg2buf(self, ch_id, ra_id, de_id, pu_num, pu_mask, buffer_addr):
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 buffer_addr
        #return (LEVEL.DE,  OPTYPE.reg2buf, 0,      0,      0,     (16, [True for _ in range(16)]), 0)
        # self.inst_count['device_reg2buf'] += 1
        self.inst_count[3] += 1
        return (LEVEL.DE,  OPTYPE.reg2buf, ch_id, ra_id, de_id, (pu_num, pu_mask), buffer_addr)

    def create_device_buf2reg(self, ch_id, ra_id, de_id, pu_num, pu_mask, buffer_addr):
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 buffer_addr
        #return (LEVEL.DE,  OPTYPE.buf2reg, 0,      0,      0,     (16, [True for _ in range(16)]), 1)
        # self.inst_count['device_buf2reg'] += 1
        self.inst_count[4] += 1
        return (LEVEL.DE,  OPTYPE.buf2reg, ch_id, ra_id, de_id, (pu_num, pu_mask), buffer_addr)

    def create_device_buf2bk(self, ch_id, ra_id, de_id, pu_num, pu_mask, op1, buf, auto_precharge):
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 op:(bank, row_id, col_offset), auto_precharge
        #return (LEVEL.DE,  OPTYPE.buf2bk,  0,      0,      0,     (8, [True for _ in range(8)]),   (0, 1, 0),                      False)
        # self.inst_count['device_buf2bk'] += 1
        self.inst_count[5] += 1
        # self.inst_count['device_buf2bk_col'] += self.de_pu_outbuf_col
        self.inst_count[6] += self.de_pu_outbuf_col
        return (LEVEL.DE,  OPTYPE.buf2bk, ch_id, ra_id, de_id, (pu_num, pu_mask), op1, buf, auto_precharge)

    def create_device_bk2buf(self, ch_id, ra_id, de_id, pu_num, pu_mask, op1, buf, auto_precharge):
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 op:(bank, row_id, col_offset), auto_precharge
        #return (LEVEL.DE,  OPTYPE.bk2buf,  0,      0,      0,     (16, [True for _ in range(16)]), (0, 1, 32),                     True)
        # self.inst_count['device_bk2buf'] += 1
        self.inst_count[7] += 1
        # self.inst_count['device_bk2buf_col'] += op1[2] if op1[0] else self.de_pu_outbuf_col
        self.inst_count[8] += buf[2] if buf[0] else self.de_pu_outbuf_col
        return (LEVEL.DE,  OPTYPE.bk2buf, ch_id, ra_id, de_id, (pu_num, pu_mask), op1, buf, auto_precharge)

    def create_device_bk2gb(self, ch_id, ra_id, de_id, bank_id, op1, gb_col_offset, col_num, auto_precharge):
        #        op-level   op-type         ch_id,  ra_id,  de_id,  bank_id,    op:(row_id, col_offset),    gb_col_offset   col_num, auto_precharge
        #return (LEVEL.DE,  OPTYPE.bk2gb,   0,      0,      0,      0,          (1, 0),                     0,              32,      True)
        # self.inst_count['device_bk2gb'] += 1
        self.inst_count[9] += 1
        # self.inst_count['device_bk2gb_col'] += col_num
        self.inst_count[10] += col_num
        return (LEVEL.DE,  OPTYPE.bk2gb, ch_id, ra_id, de_id, bank_id, op1, gb_col_offset, col_num, auto_precharge)

    def create_device_gb2bk(self, ch_id, ra_id, de_id, bank_mask, op1, gb_col_offset, col_num, auto_precharge):
        #        op-level   op-type         ch_id,  ra_id,  de_id,  bank mask,        op:(row_id, col_offset),    gb_col_offset   col_num, auto_precharge
        #return (LEVEL.DE,  OPTYPE.gb2bk,   0,      0,      0,      0,                          (1, 0),                     0,              32,      True)
        # self.inst_count['device_gb2bk'] += 1
        self.inst_count[11] += 1
        # self.inst_count['device_gb2bk_col'] += col_num
        self.inst_count[12] += col_num
        return (LEVEL.DE,  OPTYPE.gb2bk, ch_id, ra_id, de_id, bank_mask, op1, gb_col_offset, col_num, auto_precharge)

    def create_rank_pu(self, ch_id, ra_id, pu_num, pu_mask, op1, op2, col_num, auto_precharge):
        # self.inst_count['pu'] += 1
        self.inst_count[0] += 1
        # self.inst_count['pu_col'] += col_num
        self.inst_count[1] += col_num
        if auto_precharge:
            # self.inst_count['pu_row_change'] += 1
            self.inst_count[2] += 1
        return (LEVEL.RA,  OPTYPE.pu, ch_id, ra_id, (pu_num, pu_mask), op1, op2, col_num, auto_precharge)

    def create_host_read(self, ch_id, ra_id, de_mask, bank_id, row_id, col_offset, col_num, auto_precharge):
        #        op-level   op-type         ch_id,  ra_id,  de_list,  bank_id,    row_id, col_offset, col_num, auto_precharge
        #return (LEVEL.HOST,OPTYPE.read,    0,      0,      0,      0,          0,      0,          32)
        self.inst_count[13] += 1
        self.inst_count[14] += col_num
        return (LEVEL.SYS, OPTYPE.host_read, ch_id, ra_id, de_mask, bank_id, row_id, col_offset, col_num, auto_precharge)

    def create_host_write(self, ch_id, ra_id, de_mask, bank_mask, row_id, col_offset, col_num, auto_precharge):
        #        op-level   op-type         ch_id,  ra_id,  de_id,  bank_mask,    row_id, col_offset, col_num, auto_precharge
        #return (LEVEL.HOST,OPTYPE.write,   0,      0,      0,      0,          0,      0,          32)
        self.inst_count[15] += 1
        self.inst_count[16] += col_num
        return (LEVEL.SYS, OPTYPE.host_write, ch_id, ra_id, de_mask, bank_mask, row_id, col_offset, col_num, auto_precharge)

    def create_host_write_device_buffer(self, ch_id, ra_id, device_mask, buffer_addr, col_num):
    #   op-level    op-type                 ch_id,      ra_id,      device_mask-先有一个，实在不行给他全置为True即可
        self.inst_count[17] += 1
        self.inst_count[18] += col_num
        return (LEVEL.SYS, OPTYPE.host_write_device_buffer, ch_id, ra_id, device_mask, buffer_addr, col_num)

    def create_host_read_device_buffer(self, ch_id, ra_id, device_mask, buffer_addr, col_num):
        raise NotImplementedError

    def create_host_write_pu_inbuf(self, ch_id, ra_id, device_mask, pu_mask, col_offset, col_num):
        #        op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 buffer_addr
        #return (LEVEL.DE,  OPTYPE.reg2buf, 0,      0,      0,     (16, [True for _ in range(16)]), 0)
        self.inst_count[19] += 1
        self.inst_count[20] += col_num
        return (LEVEL.SYS, OPTYPE.host_write_pu_inbuf, ch_id, ra_id, device_mask, pu_mask, col_offset, col_num)

    def create_host_read_mac_reg(self, ch_id, ra_id, device_mask, pu_mask):
        self.inst_count[21] += 1
        return (LEVEL.SYS, OPTYPE.host_read_mac_reg, ch_id, ra_id, device_mask, pu_mask)

    def create_host_write_mac_reg(self, ch_id, ra_id, device_mask, pu_mask):
        self.inst_count[22] += 1
        return (LEVEL.SYS, OPTYPE.host_write_mac_reg, ch_id, ra_id, device_mask, pu_mask)
    
    def create_host_read_rank_pu_reg(self, ch_id, ra_id, rank_pu_mask):
        self.inst_count[21] += 1
        return (LEVEL.SYS, OPTYPE.host_read_rank_pu_reg, ch_id, ra_id, rank_pu_mask)

    def create_host_write_rank_pu_reg(self, ch_id, ra_id, rank_pu_mask):
        self.inst_count[22] += 1
        return (LEVEL.SYS, OPTYPE.host_write_rank_pu_reg, ch_id, ra_id, rank_pu_mask)

    def output_buffer(self, result_col, result_row):
        last_row = self.last_buffer_row
        last_col = self.last_buffer_col
        if result_row != self.last_buffer_row:
            # exit(3)
            self.last_buffer_row = result_row
            self.buffered = [result_col]
            # print('change row')
            return True, last_col, last_row
        else:
            if result_col not in self.buffered:
                self.buffered.append(result_col)
            if len(self.buffered) > self.de_pu_outbuf_col:
                # exit(3)
                self.buffered = [result_col]
                # print('change row')
                return True, last_col, last_row
        return False, last_col, last_row

    def reset_output_buffer(self):
        self.last_buffer_col = 0
        self.last_buffer_row = 0
        self.buffered = []

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    def codegen(self, kernel_name, compute_level, pu_num, partition, 
                simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                hw_id_list, mem_mapping, mm_schedule='mkl', cmd_threshold=0, profile_level=0):
        
        # get partition & mapping
        if SimConfig.pu_level == LEVEL.RA:
            pu_m, pu_k, pu_l, pu_b = partition[2]
        else:
            pu_m, pu_k, pu_l, pu_b = partition[3]

        input_bank, input_row_offset,\
            weight_bank, weight_row_offset,\
                output_bank, output_row_offset = mem_mapping

        # input decode
        in_block, in_row, in_corner = mkl_Input_to_row
        m_block, k_block, l_block, b_block = in_block
        m_row, k_row, l_row, b_row = in_row
        m_block_corner, k_block_corner, l_block_corner, b_block_corner = in_corner
        # output decode
        out_block, out_row, out_corner = ml_Out_to_row
        om_block, ol_block, ob_block = out_block
        om_row, ol_row, ob_row = out_row
        om_block_corner, ol_block_corner, ob_block_corner = out_corner
        # get assigned hw
        channel_list, rank_list, device_list, pu_list = hw_id_list

        # select the kernel for codegen
        if kernel_name == 'mm':
            kernel = self.mm_micro
        elif kernel_name == 'elewise':
            kernel = self.elewise_micro
        elif kernel_name == 'softmax':
            kernel = self.softmax_micro
        elif kernel_name in ['layernorm','batchnorm']:
            kernel = self.layernorm_micro
        else: raise NotImplementedError
        
        # generate code
        if isinstance(self, __import__('backend.aim16', fromlist=['aim16']).aim16) \
            or isinstance(self, __import__('backend.aim8', fromlist=['aim8']).aim8) \
                or isinstance(self, __import__('backend.upmem', fromlist=['upmem']).upmem) \
                    or isinstance(self, __import__('backend.hbm_pim', fromlist=['hbmpim']).hbmpim):
            # 这里可以加入针对aim16子类的特殊处理
            # pass
            inst_groups, predict_result = \
                kernel(mm_schedule, 0, channel_list, rank_list, device_list, pu_num, simd_l,
                    input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset,
                    m_block, k_block, l_block, b_block,
                    m_row, k_row, l_row, b_row,
                    m_block_corner, k_block_corner, l_block_corner, b_block_corner,
                    om_block, ol_block, ob_block,
                    om_row, ol_row, ob_row,
                    om_block_corner, ol_block_corner, ob_block_corner,
                    pu_m, pu_k, pu_l, pu_b, pu_list, cmd_threshold, profile_level)
        else:
            inst_groups, predict_result = \
                kernel(mm_schedule, 0, channel_list, rank_list, device_list, pu_num, simd_l,
                    input_bank, input_row_offset, weight_bank, weight_row_offset, output_bank, output_row_offset,
                    m_block, k_block, l_block, b_block,
                    m_row, k_row, l_row, b_row,
                    m_block_corner, k_block_corner, l_block_corner, b_block_corner,
                    om_block, ol_block, ob_block,
                    om_row, ol_row, ob_row,
                    om_block_corner, ol_block_corner, ob_block_corner,
                    pu_m, pu_k, pu_l, pu_b, pu_list, cmd_threshold)
        return inst_groups, self.inst_count.tolist(), predict_result

    def get_matrix(self):
        assert self.gen_code, "Please set_gen() before get_matrix()"
        # inst num
        # sum_list = [0, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19] micro inst
        sum_list = [1, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22] # dram inst, 解释：Host端指令本身就是不区分Device的，所以只是在统计读写次数的时候用了
        inst_num = sum(self.inst_count[sum_list])
        inst_num += self.inst_count[2] * 2 # 增加换行次数的估计
        # DRAM access num
        sum_list = [1, 6, 8, 10, 12]
        pu_dram_num = sum(self.inst_count[sum_list])
        sum_list = [14, 16, 18, 20, 21, 22]
        host_dram_num = sum(self.inst_count[sum_list]) * self.device_num # 本质上是DRAM对总线的占用
        # row change num
        row_change_num = self.inst_count[2]
        return inst_num, pu_dram_num, host_dram_num, row_change_num