import os
import shutil
import re

def organize_files_by_region_improved(source_folder):
    """
    根据文件名中的区域信息（位于下划线分隔的第3,4,5,6部分）创建对应文件夹并移动/复制文件
    文件名格式：prefix_suffix_Y1_Y2_X1_X2.ext
    """
    
    # 编译一个更精确的正则表达式来匹配 Y1_Y2_X1_X2 部分
    # 这个模式寻找至少三个由下划线分隔的数字组，并捕获最后四个作为坐标
    pattern = re.compile(r'.*?_\d+_(\d+)_(\d+)_(\d+)_(\d+)(?:\.[^.]+)?$')
    
    processed_count = 0
    skipped_count = 0
    
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            match = pattern.match(filename)
            
            if match:
                y1, y2, x1, x2 = match.groups()
                region_folder_name = f"{y1}_{y2}_{x1}_{x2}"
                
                target_folder = os.path.join(source_folder, region_folder_name)
                
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                
                target_path = os.path.join(target_folder, filename)
                
                # 可以选择 move 或 copy
                # shutil.move(file_path, target_path) # 移动
                shutil.copy2(file_path, target_path) # 复制
                
                print(f"已处理: {filename} -> {region_folder_name}/")
                processed_count += 1
            else:
                # 再尝试一个更通用的模式，查找任意位置的连续4个数字下划线组
                # 例如：任何 a_b_c_d_e_f_g_h 形式，其中 c_d_e_f 是数字
                generic_pattern = re.compile(r'(_|^)(\d+)_(\d+)_(\d+)_(\d+)(_|$)')
                matches_generic = list(generic_pattern.finditer(filename))
                
                if matches_generic:
                    # 取最后一个匹配项，通常是区域坐标
                    last_match = matches_generic[-1]
                    y1, y2, x1, x2 = last_match.group(2), last_match.group(3), last_match.group(4), last_match.group(5)
                    region_folder_name = f"{y1}_{y2}_{x1}_{x2}"
                    
                    target_folder = os.path.join(source_folder, region_folder_name)
                    
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    
                    target_path = os.path.join(target_folder, filename)
                    
                    # 可以选择 move 或 copy
                    # shutil.move(file_path, target_path) # 移动
                    shutil.copy2(file_path, target_path) # 复制
                    
                    print(f"已处理 (通用模式): {filename} -> {region_folder_name}/")
                    processed_count += 1
                else:
                    print(f"跳过非标准格式文件: {filename}")
                    skipped_count += 1

    print(f"\n--- 整理完成 ---")
    print(f"成功处理文件数: {processed_count}")
    print(f"跳过文件数: {skipped_count}")


def organize_files_by_region_move_mode(source_folder):
    """
    移动模式
    """
    pattern = re.compile(r'.*?_\d+_(\d+)_(\d+)_(\d+)_(\d+)(?:\.[^.]+)?$')
    
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            match = pattern.match(filename)
            
            if match:
                y1, y2, x1, x2 = match.groups()
                region_folder_name = f"{y1}_{y2}_{x1}_{x2}"
                
                target_folder = os.path.join(source_folder, region_folder_name)
                
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                
                target_path = os.path.join(target_folder, filename)
                
                shutil.move(file_path, target_path) # 移动
                
                print(f"已移动: {filename} -> {region_folder_name}/")
            else:
                generic_pattern = re.compile(r'(_|^)(\d+)_(\d+)_(\d+)_(\d+)(_|$)')
                matches_generic = list(generic_pattern.finditer(filename))
                
                if matches_generic:
                    last_match = matches_generic[-1]
                    y1, y2, x1, x2 = last_match.group(2), last_match.group(3), last_match.group(4), last_match.group(5)
                    region_folder_name = f"{y1}_{y2}_{x1}_{x2}"
                    
                    target_folder = os.path.join(source_folder, region_folder_name)
                    
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    
                    target_path = os.path.join(target_folder, filename)
                    
                    shutil.move(file_path, target_path) # 移动
                    
                    print(f"已移动 (通用模式): {filename} -> {region_folder_name}/")
                else:
                    print(f"跳过非标准格式文件: {filename}")

if __name__ == "__main__":
    source_directory = input("请输入要整理的文件夹路径: ").strip()
    
    if not os.path.exists(source_directory):
        print("指定的文件夹不存在！")
    else:
        print("\n请注意：")
        print("- 推荐先使用 'copy' 模式进行测试，确认无误后再使用 'move' 模式。")
        print("- 例如，文件 seq27_000600_0_1080_2560_3840.png 将被放入名为 '0_1080_2560_3840' 的文件夹。")
        mode = input("选择操作模式 - 输入 'move' 移动文件 或 'copy' 复制文件: ").strip().lower()
        
        if mode == 'move':
            print("\n开始移动文件...")
            organize_files_by_region_move_mode(source_directory)
        elif mode == 'copy':
            print("\n开始复制文件...")
            organize_files_by_region_improved(source_directory)
        else:
            print("无效选项，请输入 'move' 或 'copy'")
        
        print("\n任务完成！")
