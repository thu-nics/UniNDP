from frontend import *
from midend import *
from backend import *
from sim import sim
from tools import *
import argparse
import time
import tqdm
import csv
import os

# NOTE: if you want to disable tqdm in the framework, you can use the following code
# def tqdm_replacement(iterable_object,*args,**kwargs):
#     return iterable_object
# tqdm_copy = tqdm.tqdm # store it if you want to use it later
# tqdm.tqdm = tqdm_replacement

tCK = 1
tREFI = 3900
tRFC = 350

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--workloadsize', '-S', nargs='+', type=int, default=[5000,5000])
    argparser.add_argument('--po2', '-P', action='store_true')
    argparser.add_argument('--allow_under_ultize', '-UU', action='store_true')
    args = argparser.parse_args()

    # set up log file
    output_dir_name = "verify_result" # NOTE: you can change the output dir name
    os.makedirs(f"{output_dir_name}/csv", exist_ok=True)
    os.makedirs(f"{output_dir_name}/log", exist_ok=True)
    outcsv_name = f"./{output_dir_name}/csv/{args.workloadsize}.csv"
    log_name = f"./{output_dir_name}/log/{args.workloadsize}.log"
    log_file = open(log_name, 'w+')

    # workload size: [M, K, N, B]
    assert len(args.workloadsize) == 2 , f"Invalid workload size: {args.workloadsize}"
    mm_size = tuple([1,args.workloadsize[0], args.workloadsize[1], 1])

    # hw config
    SimConfig.read_from_yaml('./config/hbm-pim.yaml')
    Codegen = hbmpim_verify

    # set pu level
    SimConfig.pu_level = LEVEL.DE
    # check sim config
    print("MM Size: ", mm_size, file=log_file)

    """
    NOTE: 1. Get design space
    """
    # A. get hw partition space
    design_space = []
    partition_tool = Partition(require_power_of_2 = args.po2)
    partition_space = partition_tool.get_partition_space_mm(mm_size)
    # filter hw partition space
    filtered_partition_space = partition_tool.choose_from_partition_space_mm(partition_space)
    if not args.allow_under_ultize:
        partition_space = filtered_partition_space
    # print(len(partition_space), file=log_file)
    # partition space + mapping space = design space
    for index in tqdm.tqdm(range(len(partition_space))):
        compute_level, pu_num, partition = partition_space[index]
        # B. get mem partition space
        simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row \
            = partition_tool.mem_partition_mm(mm_size, partition)
        # report design space
        for input_choice in reversed(mkl_Input_to_row):
            for output_choice in reversed(ml_Out_to_row):
                # print(f"simd_k: {simd_k}, mkl_Input_to_row: {mkl_Input_to_row}, simd_l: {simd_l}, ml_Out_to_row: {ml_Out_to_row}")
                design_space.append((compute_level, pu_num, partition, simd_k, input_choice, simd_l, output_choice))
    # create ./dump/csv/ if not exist
    os.makedirs(os.path.dirname(outcsv_name), exist_ok=True)
    csvfile = open(outcsv_name, 'w', newline='')
    writer = csv.writer(csvfile)
    
    """
    NOTE: 2. get the baseline result
    """
    baseline = None

    for compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row in design_space:
        # m,k,l,b: only partition on l, and k = 8
        if partition[3][0] * partition[3][1] * partition[3][3] * \
            partition[2][0] * partition[2][1] * partition[2][3] * \
            partition[1][0] * partition[1][1] * partition[1][3] * \
            partition[0][0] * partition[0][1] * partition[0][3] == 1 and \
            mkl_Input_to_row[0][1] == 8:
            baseline = compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row
            print(f"strategy: {baseline}", file=log_file)
            break

    if baseline == None: # corner case
        baseline = design_space[0]
        print(f"strategy: {baseline}", file=log_file)

    compute_level, pu_num, partition, simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row = baseline
    
    # A. hw mapping
    mapping_tool = Mapping(require_power_of_2 = args.po2)
    hw_id_list = mapping_tool.assign_hw(partition)
    # B. dram mapping
    input_bank, input_row_offset, \
    weight_bank, weight_row_offset, \
    output_bank, output_row_offset \
        = mapping_tool.assign_dram(pu_num, mkl_Input_to_row, ml_Out_to_row, partition) 
    # C. scheduling: TODO
    # D. Codegen
    codegen_tool = Codegen(require_power_of_2 = args.po2)
    codegen_tool.set_gen()
    start_time = time.time()
    gen_code, inst_count, predict_result = \
    codegen_tool.codegen('mm', compute_level, pu_num, partition,
                simd_k, mkl_Input_to_row, simd_l, ml_Out_to_row,
                hw_id_list, (input_bank, input_row_offset,
                            weight_bank, weight_row_offset,
                            output_bank, output_row_offset),
                            cmd_threshold=0)
    # E. simulation
    baseline_sim_result = sim(gen_code, silent=True, sim_verify=1)
    baseline_sim_result = tCK * baseline_sim_result * (tRFC + tREFI) / tREFI

    end_time = time.time()

    # input_size, output_size, result
    writer.writerow([mm_size[1], mm_size[2], baseline_sim_result])
    csvfile.flush()
    
    print(f"result: {baseline_sim_result}", file=log_file)
    print(f"simulation time: {end_time - start_time}", file=log_file)
    csvfile.close()

if __name__ == '__main__':
    main()
