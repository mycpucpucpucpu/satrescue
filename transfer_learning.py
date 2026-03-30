#迁移学习代码
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import time
import psutil

# =========================
# 参数设置
# =========================
old_data_path = 'old_data.csv'  # 原始旧数据路径
new_data_path = 'new_data.csv'  # 新数据路径
output_dir = './outputs/'
model_dir = './models/transfer_model/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
output_file = os.path.join(output_dir, '迁移学习_最优角度预测.csv')

label = '电流'  # 目标变量
TARGET_OLD_SAMPLES = 50000  # 旧数据目标采样量
OLD_WEIGHT = 0.8  # 降低旧数据权重
NEW_WEIGHT = 2.5  # 提高新数据权重

# =========================
# 1. 读取数据
# =========================
print("读取旧数据...")
old_df = pd.read_csv(old_data_path)
print(f"原始旧数据量: {len(old_df):,} 行")

print("读取新数据...")
new_df = pd.read_csv(new_data_path)
print(f"新数据量: {len(new_df):,} 行")

# =========================
# 2. 旧数据智能抽样 (75万 → 5万)
# =========================
def stratified_sample(df, target_samples, key_cols=['电压', '温度']):
    """基于关键特征的分层抽样"""
    # 如果关键列不在数据中，使用随机抽样
    if not all(col in df.columns for col in key_cols):
        print("⚠️ 关键列缺失，使用随机抽样")
        return df.sample(n=target_samples, random_state=42)
    
    sample_df = df.groupby(key_cols).apply(
        lambda x: x.sample(min(max(1, int(len(x) * target_samples/len(df))), len(x)))
    ).reset_index(drop=True)
    
    # 如果抽样后仍超过目标量，进行二次抽样
    if len(sample_df) > target_samples:
        sample_df = sample_df.sample(n=target_samples, random_state=42)
    return sample_df

if len(old_df) > TARGET_OLD_SAMPLES:
    print(f"\n🔄 开始旧数据抽样 ({len(old_df):,} → {TARGET_OLD_SAMPLES:,})...")
    
    # 方法1：快速随机抽样
    old_df = old_df.sample(n=TARGET_OLD_SAMPLES, random_state=42)
    
    # 方法2：分层抽样（取消注释使用）
    # old_df = stratified_sample(old_df, TARGET_OLD_SAMPLES)
    
    print(f"✅ 抽样完成，当前旧数据量: {len(old_df):,} 行")

# =========================
# 3. 列检查与对齐
# =========================
if set(old_df.columns) != set(new_df.columns):
    print("⚠️ 新旧数据列不一致，尝试对齐...")
    common_cols = list(set(old_df.columns) & set(new_df.columns))
    if label not in common_cols:
        raise ValueError(f"❌ 目标列'{label}'缺失")
    
    old_df = old_df[common_cols]
    new_df = new_df[common_cols]
    
print("\n✅ 列已对齐")
print(f"共有列: {len(old_df.columns)}个")
print("列名:", ', '.join(old_df.columns.tolist()))

# =========================
# 4. 添加权重并合并
# =========================
print("\n📊 添加样本权重...")
old_df['sample_weight'] = OLD_WEIGHT
new_df['sample_weight'] = NEW_WEIGHT

# 创建验证集（仅使用新数据）
val_df = new_df.sample(frac=0.2, random_state=42)
train_df = pd.concat([old_df, new_df.drop(val_df.index)], ignore_index=True)

print(f"训练数据: {len(train_df):,} 行 (旧:{len(old_df):,}, 新:{len(new_df)-len(val_df):,})")
print(f"验证数据: {len(val_df):,} 行")

# =========================
# 5. 训练迁移学习模型（修复参数错误）
# =========================
print("\n🚀 开始训练迁移学习模型...")
start_train = time.time()

# 确保模型目录存在
os.makedirs(model_dir, exist_ok=True)

# 检查并清理旧模型
if os.path.exists(os.path.join(model_dir, "predictor.pkl")):
    print("⚠️ 已存在旧模型，将被覆盖")
    # 可以选择删除旧模型目录
    # import shutil
    # shutil.rmtree(model_dir)
    # os.makedirs(model_dir, exist_ok=True)

# 创建预测器
predictor = TabularPredictor(
    label=label,
    path=model_dir,
    sample_weight='sample_weight',
    eval_metric='root_mean_squared_error'
)

# 修复：移除不支持的fit_weighted_metrics参数
try:
    predictor.fit(
        train_data=TabularDataset(train_df),
        tuning_data=TabularDataset(val_df),  # 使用新数据验证
        presets='medium_quality',
        time_limit=600,  # 10分钟
        # fit_weighted_metrics=True  # 此参数在新版本中已被移除
    )
except Exception as e:
    print(f"❌ 训练出错: {e}")
    print("尝试不使用tuning_data参数...")
    predictor.fit(
        train_data=TabularDataset(train_df),
        presets='medium_quality',
        time_limit=600
    )

print(f"✅ 模型训练完成，耗时: {time.time()-start_train:.2f}秒")
print("模型性能评估:")
try:
    print(predictor.leaderboard(silent=False))
except:
    print("无法获取leaderboard")

# =========================
# 6. 最优角度预测（增强稳定性）
# =========================
def predict_optimal_angles(df, predictor, label, batch_size=100):
    """批量预测最优角度"""
    print("\n🚀 开始最优角度预测...")
    start_time = time.time()
    
    # 删除不需要的列
    df_infer = df.copy()
    drop_cols = ['sample_weight', label]
    for col in drop_cols:
        if col in df_infer.columns:
            df_infer = df_infer.drop(columns=[col])
    
    # 添加必要的特征（如果缺失）
    required_features = predictor.feature_metadata_in.get_features()
    for feat in required_features:
        if feat not in df_infer.columns:
            print(f"⚠️ 添加缺失特征: {feat} (使用默认值0)")
            df_infer[feat] = 0
    
    N = len(df_infer)
    
    # 动态调整批次大小
    available_mem = psutil.virtual_memory().available / 1e9  # 可用内存(GB)
    if available_mem > 32:
        batch_size = 500
    elif available_mem > 16:
        batch_size = 200
    else:
        batch_size = 50
    print(f"可用内存: {available_mem:.1f}GB, 使用批次大小: {batch_size}")
    
    b = []
    all_best_angles = []
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_df = df_infer.iloc[start:end]
        print(f"\n🟢 处理批次: {start}-{end} ({len(batch_df)}行)")
        
        # 扩展数据 (每行生成360个角度)
        df_expanded = batch_df.loc[batch_df.index.repeat(360)].reset_index(drop=True)
        print(f"  扩展后: {len(df_expanded):,}行")
        
        # 添加角度特征
        angles = np.tile(np.arange(1, 361), len(batch_df))
        angle_rad = np.radians(angles)
        df_expanded['angle_sin'] = np.sin(angle_rad)
        df_expanded['angle_cos'] = np.cos(angle_rad)
        
        # 确保所有特征都存在
        for feat in required_features:
            if feat not in df_expanded.columns:
                print(f"⚠️ 添加扩展后缺失特征: {feat} (使用默认值0)")
                df_expanded[feat] = 0
        
        # 预测电流值
        try:
            preds = predictor.predict(df_expanded)
            print(f"  预测完成: {len(preds):,}个结果")
            
            # 找出每行最佳角度
            pred_matrix = preds.values.reshape(-1, 360)
            max_idx = np.argmax(pred_matrix, axis=1)
            max_currents = np.max(pred_matrix, axis=1)
            best_angles = max_idx + 1
            
            b.extend(max_currents)
            all_best_angles.extend(best_angles)
        
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            # 使用默认值填充失败的批次
            print("⚠️ 使用默认值0填充当前批次")
            b.extend([0]*len(batch_df))
            all_best_angles.extend([180]*len(batch_df))
    
    # 添加结果列
    df_infer['max_current'] = b
    df_infer['best_angle'] = all_best_angles
    
    print(f"\n✅ 角度预测完成，总耗时: {time.time()-start_time:.2f}秒")
    return df_infer

# 对合并数据预测
full_df = pd.concat([train_df, val_df], ignore_index=True)
result_df = predict_optimal_angles(full_df, predictor, label)

# =========================
# 7. 保存结果
# =========================
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n📁 结果已保存至: {output_file}")

# 性能报告
print("\n===== 性能报告 =====")
print(f"原始旧数据: {old_data_path} ({len(pd.read_csv(old_data_path)):,}行)")
print(f"处理后旧数据: {len(old_df):,}行")
print(f"新数据: {len(new_df):,}行")
print(f"总处理数据: {len(result_df):,}行")
print(f"最佳角度范围: {result_df['best_angle'].min()}° - {result_df['best_angle'].max()}°")
print(f"最大电流范围: {result_df['max_current'].min():.2f}A - {result_df['max_current'].max():.2f}A")
print("预测完成!")