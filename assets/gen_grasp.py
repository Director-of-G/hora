import os
import subprocess

import numpy as np


HORA_HOME = os.path.join(os.path.dirname(__file__), "../")
SCRIPT_NAME = "./scripts/gen_grasp.sh"

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
    # for scale in np.arange(0.46, 0.68+0.02, 0.02):
    for scale in np.arange(0.70, 0.86, 0.02)[::-1]:
        for name in object_names:
            if "_" in name:
                continue
            if os.path.isfile(f"{dataset}/cache/{name}/grasp_50k_s{str(round(scale, 2)).replace('.', '')}.npy"):
                print(f"skipping as grasp cache for {name} with scale {scale} already exists!")
                continue
            print(f"generating grasp cache for {name} with scale {scale}...")
            is_success = gen_grasp_for_single_object_scale(dataset, name, scale, cache_size)
            if not is_success: print("skipping as generation failed!")

        
if __name__ == "__main__":
    gen_grasp_for_dataset("miscnet")
