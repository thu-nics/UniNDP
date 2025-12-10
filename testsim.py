from sim import sim
from tools import *

"""
test sim code
"""

# test commands
test_command_0 = [
#   0. MAC, Global buffer + DRAM, op1.bank = op2.bank, op2 row > 0
#    op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 op1:(bank, row_id, col_offset), op2:,       col_num,auto_precharge
    (LEVEL.DE,  OPTYPE.pu,      0,      0,      0,     (8, [True for _ in range(8)]),   (0, 0, 0),                      (1, 0, 0),  64,     False),
    (LEVEL.DE,  OPTYPE.pu,      0,      0,      0,     (16, [True for _ in range(16)]), (0, 1, 0),                      (0, 1, 0),  64,     True), # NOTE: op2 row > 0 -> global buffer
#   1. PU reg <-> local buffer
#    op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 buffer_addr
    (LEVEL.DE,  OPTYPE.reg2buf, 0,      0,      0,     (16, [True for _ in range(16)]), 0),
    (LEVEL.DE,  OPTYPE.buf2reg, 0,      0,      0,     (16, [True for _ in range(16)]), 1),
#   2. local buffer (output) <-> bank
#    op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 op:(bank, row_id, col_offset), (is_input, buf_addr, col_len), auto_precharge
    (LEVEL.DE,  OPTYPE.buf2bk,  0,      0,      0,     (8, [True for _ in range(8)]),   (0, 1, 0),                     (False, 0, 0),                 False         ),
    (LEVEL.DE,  OPTYPE.bk2buf,  0,      0,      0,     (16, [True for _ in range(16)]), (0, 1, 32),                    (False, 0, 0),                 True          ),
#   3. bank -> local buffer (input), require is_input = True
#   op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                  op:(bank, row_id, col_offset), (is_input, buf_addr, col_len), auto_precharge
    (LEVEL.DE,  OPTYPE.bk2buf,  0,      0,      0,     (16, [True for _ in range(16)]), (0, 1, 0),                     (True, 0, 64),                 False         ),
#   4. MAC, Local buffer + DRAM, op1.bank = op2.bank
#    op-level   op-type         ch_id,  ra_id,  de_id,  pu:(num, mask),                 op1:(bank, row_id, col_offset), op2:,       col_num,    auto_precharge
    (LEVEL.DE,  OPTYPE.pu,      0,      0,      0,     (16, [True for _ in range(16)]), (0, 1, 0),                      (0, 0, 0),  64,         True), # NOTE: op2 row == 0 -> local input buffer
#   5. bank <-> global buffer
#    op-level   op-type         ch_id,  ra_id,  de_id,  bank_id / bank mask,        op:(row_id, col_offset),    gb_col_offset   col_num, auto_precharge
    (LEVEL.DE,  OPTYPE.bk2gb,   0,      0,      0,      0,                          (1, 0),                     0,              32,      True),
    (LEVEL.DE,  OPTYPE.gb2bk,   0,      0,      0,     [True for _ in range(16)],   (2, 0),                     0,              32,      True),

]

# test rank level instruction
test_command_2 = [
#   some RA-level pu command, what about PIM, but PIM need to keep all the data in SRAM
#   op-level    op-type         ch_id,  ra_id,  pu:(num, mask),                 op1:(device, bank, row_id, col_offset), op2:,           col_num,    auto_precharge
    (LEVEL.RA,  OPTYPE.pu,      0,      0,      (4, [True for _ in range(4)]),  (0, 0, 0, 0),                           (1, 0, 0, 0),   15,         False),
    (LEVEL.RA,  OPTYPE.pu,      0,      0,      (4, [True for _ in range(4)]),  (0, 0, 0, 0),                           (1, 0, 0, 0),   15,         True),
]

# test Host Read / Write Command
test_command_3 = [
#   1. Host read bank
#   op-level    op-type                 ch_id,      ra_id,      device_mask
    (LEVEL.SYS, OPTYPE.host_read,       0,          0,          [True for _ in range(8)],
#   bank_id,                            row_id,     col_offset, col_num,    auto_precharge
    1,                                  0,          0,          64,         False),

#   2. Host write bank
#   op-level    op-type                 ch_id,      ra_id,      device_mask
    (LEVEL.SYS, OPTYPE.host_write,      0,          0,          [True for _ in range(8)],
#   bank_mask,                          row_id,     col_offset, col_num,    auto_precharge
    [True for _ in range(16)],          0,          0,          64,         True),

#   3. host write device buffer
#   op-level    op-type                 ch_id,      ra_id,      device_mask
    (LEVEL.SYS, OPTYPE.host_write_device_buffer, 0, 0,          [True for _ in range(8)],
#   buffer_addr,col_num
    0,          64    ),

# #   4. host read device buffer
# #   op-level   op-type                 ch_id,      ra_id,      device_mask
#     (LEVEL.SYS, OPTYPE.host_read_device_buffer, 0, 0,          [True for _ in range(8)],
# #   buffer_addr,col_num
#     0,          64    ),

#  5. host write pu in buffer, allow broadcast
#   op-level   op-type                 ch_id,      ra_id,      device_mask
    (LEVEL.SYS, OPTYPE.host_write_pu_inbuf, 0, 0,          [True for _ in range(8)],
#   pu_mask,                     col_offset, col_num
    [True for _ in range(16)],   0,          64),

#  6. host read pu reg
#   op-level   op-type                 ch_id,      ra_id,      device_mask
    (LEVEL.SYS, OPTYPE.host_read_mac_reg,   0,     0,          [True for _ in range(8)],
#   pu_mask,                      
    [True for _ in range(16)]),

#  7. host write pu reg
#   op-level   op-type                 ch_id,      ra_id,      device_mask
    (LEVEL.SYS, OPTYPE.host_write_mac_reg, 0,      0,          [True for _ in range(8)],
#   (pu_num, pu_mask)
    [True for _ in range(16)]),

#8. op-level   op-type          ch_id,  ra_id,  rank_pu_mask,                
    (LEVEL.SYS, OPTYPE.host_read_rank_pu_reg, 0, 0, [True for _ in range(4)]),
#9. op-level   op-type          ch_id,  ra_id,  rank_pu_mask,
    (LEVEL.SYS, OPTYPE.host_write_rank_pu_reg, 0, 0, [True for _ in range(4)]),
]

SimConfig.read_from_yaml('./config/testsim.yaml')

lat = sim([ # dependency: 0 <- 1 <- 2
    (0, [], test_command_2),
    (1, [], test_command_3),
    (2, [], test_command_0),
], silent=False)

print(f"latency: {lat}")