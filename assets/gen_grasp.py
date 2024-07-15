import os
import subprocess

import numpy as np


HORA_HOME = os.path.join(os.path.dirname(__file__), "../")
SCRIPT_NAME = "./scripts/gen_grasp.sh"

WHITELIST, BLACKLIST = {}, {}
# if given, will only include the objects
WHITELIST["miscnet"] = {
    "44447", "15cfi6q3", "34gw8w1t", "teapot",
    "3ebm2mp0", "xqmslxc8", "30admsmv", "206038",
    "igea", "3q783q28", "270o9y3w", "2626okj4",
    "1b0kr9wf", "37384", "208741", "spot",
    "75443"
}
# if given, will exclude the objects
BLACKLIST["ycb"] = {
    "019_pitcher_base", "025_mug", "063-a_marbles",
    "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups",
    "065-f_cups", "065-g_cups", "065-h_cups", "065-i_cups", "065-j_cups",
    "070-a_colored_wood_blocks", "073-e_lego_duplo"
}

def gen_grasp_for_single_object_scale(dataset, object, scale, cache_size):
    # 定义要执行的 shell 脚本和参数
    shell_script = SCRIPT_NAME
    script_args = [
        "0",
        f"{scale}",
        f"assets/{dataset}/urdf/{object}.urdf",
        f"{cache_size}"
    ]  # 脚本参数

    # 定义工作目录
    working_directory = HORA_HOME

    # 创建完整的命令
    command = [shell_script] + script_args

    # 使用 subprocess.run 执行 shell 脚本
    gen_grasp_success = False
    try:
        result = subprocess.run(command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print("Shell script output:")
        # print(result.stdout)
        if result.returncode == 0 and os.path.isfile(f"{dataset}/cache/{object}/grasp_50k_s{str(round(scale, 2)).replace('.', '')}.npy"):
            gen_grasp_success = True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    return gen_grasp_success

def gen_grasp_for_dataset(dataset, cache_size=50):
    folder = f"./{dataset}/urdf"
    object_names = os.listdir(folder)
    object_names = [name.split(".")[0] for name in object_names]
    object_names.sort()

    if dataset in WHITELIST:
        object_names = [name for name in object_names if name in WHITELIST[dataset]]
    if dataset in BLACKLIST:
        object_names = [name for name in object_names if name not in BLACKLIST[dataset]]

    print("Will generate grasp cache for the following objects:")
    print(object_names)

    # for scale in np.arange(0.46, 0.68+0.02, 0.02):
    # for scale in np.arange(0.70, 0.86, 0.02)[::-1]:
    for scale in [0.78, 0.76, 0.74, 0.72]:
        # for name in WHITELIST:
        for name in object_names:
            # if "_" in name:
            #     continue
            if os.path.isfile(f"{dataset}/cache/{name}/grasp_50k_s{str(round(scale, 2)).replace('.', '')}.npy"):
                print(f"skipping as grasp cache for {name} with scale {scale} already exists!")
                continue
            print(f"generating grasp cache for {name} with scale {scale}...")
            is_success = gen_grasp_for_single_object_scale(dataset, name, scale, cache_size)
            if is_success:
                print("generation succeeded!")
            else:
                print("skipping as generation failed!")

        
if __name__ == "__main__":
    gen_grasp_for_dataset("ycb", cache_size=400)
