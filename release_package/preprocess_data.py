import pandas as pd
import os
import numpy as np

# 1. 配置路径
BASE_DIR = "/public/home/ziyang/code/dlp_affinity/data/7KMG/"
TRAIN_FILE = "LY-CoV555_DMS_train_model_input.csv"
VAL_FILE = "LY-CoV555_DMS_val_model_input.csv"

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_and_save(filename, tag):
    input_path = os.path.join(BASE_DIR, filename)
    print(f"正在处理: {input_path} ...")
    
    # 读取原始数据
    df = pd.read_csv(input_path)
    
    # 2. 核心处理：重命名列以匹配模型默认配置
    # 原始列名 -> 模型所需列名
    rename_map = {
        "antibody_seq": "seq_ab",
        "antigen_seq":  "seq_ag",
        "escape_fraction": "kd"  # 暂时将 Escape Fraction 映射为 'kd' 标签列
    }
    
    # 检查列是否存在
    for col in rename_map.keys():
        if col not in df.columns:
            raise ValueError(f"错误：文件中找不到列 '{col}'")
            
    # 重命名并只保留必要列
    df_processed = df.rename(columns=rename_map)
    df_final = df_processed[["seq_ab", "seq_ag", "kd"]]
    
    # 3. 数据检查
    # 检查是否有空值
    if df_final.isnull().any().any():
        print(f"警告：{tag} 数据中存在空值，已自动丢弃。")
        df_final = df_final.dropna()
        
    # 检查 Escape Fraction (kd) 的范围
    min_val = df_final['kd'].min()
    max_val = df_final['kd'].max()
    print(f"  - {tag}集数据量: {len(df_final)}")
    print(f"  - 标签值(kd/escape)范围: {min_val:.6f} ~ {max_val:.6f}")
    
    # 特别提示：如果有 0 值，训练时必须关闭 log_transform
    if min_val <= 0:
        print(f"  ! 注意: 存在 <= 0 的标签值。后续训练时必须在配置中设置 'log_transform_kd = False'")

    # 4. 保存结果
    output_path = os.path.join(OUTPUT_DIR, f"{tag}.csv")
    df_final.to_csv(output_path, index=False)
    print(f"  -> 已保存至: {output_path}\n")

if __name__ == "__main__":
    print("=== 开始数据预处理 ===\n")
    process_and_save(TRAIN_FILE, "train")
    process_and_save(VAL_FILE, "val")
    print("=== 处理完成 ===")
    print(f"处理后的数据位于: {OUTPUT_DIR}")