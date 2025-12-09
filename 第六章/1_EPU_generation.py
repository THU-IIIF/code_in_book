#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第六章 - 程序1：基于大语言模型的EPU指数生成

功能：
1. 读取新闻数据
2. 使用大语言模型判断新闻是否与经济政策不确定性相关
3. 计算EPU指数
4. 生成完整的月度EPU时间序列

输入：data/sample_news.csv
输出：data/epu_index.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

# ==================== 配置参数 ====================

# EPU关键词列表（用于判断新闻相关性）
EPU_KEYWORDS = [
    '货币政策', '财政政策', '央行', '利率', '汇率',
    '税收', '关税', '监管', '法规', '改革',
    '不确定性', '风险', '波动', '危机', '调控',
    '贸易摩擦', '地缘政治', '政策变化', '经济目标'
]

# 经济关键词列表
ECONOMIC_KEYWORDS = [
    'GDP', '经济', '增长', '通胀', 'CPI', 'PPI',
    '失业', '就业', '投资', '消费', '出口', '进口',
    '产业', '制造业', '服务业', '金融'
]

# 政策关键词列表  
POLICY_KEYWORDS = [
    '政策', '决定', '宣布', '出台', '实施',
    '调整', '改革', '会议', '规划', '措施'
]

# 不确定性关键词列表
UNCERTAINTY_KEYWORDS = [
    '不确定', '可能', '担忧', '风险', '波动',
    '震荡', '影响', '紧张', '危机', '压力'
]


# ==================== 模拟LLM判断函数 ====================

def simulate_llm_classification(text, use_keywords=True):
    """
    模拟大语言模型对新闻的分类判断
    
    在实际应用中，这里应该调用真实的LLM API（如DeepSeek、GPT等）
    本函数使用关键词匹配+随机因素来模拟LLM的判断过程
    
    参数:
        text: 新闻文本
        use_keywords: 是否使用关键词辅助判断
        
    返回:
        epu_related: 是否与EPU相关 (0或1)
        confidence: 置信度 (0-1之间)
    """
    
    if use_keywords:
        # 统计不同类别关键词出现次数
        epu_count = sum(1 for kw in EPU_KEYWORDS if kw in text)
        econ_count = sum(1 for kw in ECONOMIC_KEYWORDS if kw in text)
        policy_count = sum(1 for kw in POLICY_KEYWORDS if kw in text)
        uncertainty_count = sum(1 for kw in UNCERTAINTY_KEYWORDS if kw in text)
        
        # 计算EPU相关性得分
        epu_score = (epu_count * 2 + econ_count + policy_count + uncertainty_count) / 20
        
        # 添加随机噪声模拟LLM的不确定性
        noise = np.random.normal(0, 0.1)
        final_score = np.clip(epu_score + noise, 0, 1)
        
        # 判断是否相关（阈值0.3）
        epu_related = 1 if final_score > 0.3 else 0
        confidence = final_score if epu_related else 1 - final_score
        
    else:
        # 纯随机模拟
        epu_related = np.random.choice([0, 1], p=[0.6, 0.4])
        confidence = np.random.uniform(0.5, 0.95)
    
    return epu_related, confidence


def batch_classify_news(news_df, simulate_api_delay=True):
    """
    批量分类新闻
    
    参数:
        news_df: 新闻数据DataFrame
        simulate_api_delay: 是否模拟API调用延迟
        
    返回:
        分类结果DataFrame
    """
    results = []
    
    print("正在使用大语言模型分析新闻...")
    for idx, row in news_df.iterrows():
        # 合并标题和内容
        full_text = row['news_title'] + ' ' + row['news_content']
        
        # 调用LLM分类（这里是模拟）
        epu_related, confidence = simulate_llm_classification(full_text)
        
        results.append({
            'date': row['date'],
            'news_title': row['news_title'],
            'epu_related': epu_related,
            'confidence': confidence
        })
        
        # 模拟API调用延迟
        if simulate_api_delay:
            time.sleep(0.05)
    
    return pd.DataFrame(results)


# ==================== EPU指数计算 ====================

def calculate_monthly_epu(classified_df, baseline_date='2020-01'):
    """
    计算月度EPU指数
    
    方法：
    EPU_t = (EPU相关新闻数 / 总新闻数) × 基准化系数
    
    参数:
        classified_df: 分类后的新闻DataFrame
        baseline_date: 基准日期，该月的EPU指数设为100
        
    返回:
        月度EPU指数DataFrame
    """
    
    # 提取年月
    classified_df['year_month'] = pd.to_datetime(classified_df['date']).dt.to_period('M')
    
    # 按月统计
    monthly_stats = classified_df.groupby('year_month').agg({
        'epu_related': ['sum', 'count', 'mean'],
        'confidence': 'mean'
    }).reset_index()
    
    monthly_stats.columns = ['year_month', 'epu_count', 'total_count', 'epu_ratio', 'avg_confidence']
    monthly_stats['epu_ratio'] = monthly_stats['epu_ratio'].fillna(0)
    monthly_stats['avg_confidence'] = monthly_stats['avg_confidence'].fillna(0)
    
    # 计算原始EPU指数（相关新闻比例 × 1000）
    monthly_stats['epu_raw'] = monthly_stats['epu_ratio'].fillna(0) * 1000
    
    # 基准化：将基准月设为100
    baseline_period = pd.Period(baseline_date, freq='M')
    baseline_series = monthly_stats.loc[monthly_stats['year_month'] == baseline_period, 'epu_raw']
    if not baseline_series.empty:
        baseline_value = baseline_series.iloc[0]
    else:
        baseline_value = monthly_stats['epu_raw'].replace(0, np.nan).median()
    
    if baseline_value is None or not np.isfinite(baseline_value) or baseline_value <= 0:
        baseline_value = 1.0
    
    monthly_stats['epu_index'] = (monthly_stats['epu_raw'] / baseline_value) * 100
    
    # 添加平滑后的指数（3个月移动平均）
    monthly_stats['epu_index_smooth'] = monthly_stats['epu_index'].rolling(window=3, center=True).mean()
    
    # 转换日期格式
    monthly_stats['date'] = monthly_stats['year_month'].dt.to_timestamp()
    
    return monthly_stats


# ==================== 扩展数据生成 ====================

def generate_additional_data(start_date='2020-07', end_date='2023-12', base_df=None):
    """
    生成额外的月度数据以形成完整的时间序列
    
    基于已有数据的统计特征生成合理的EPU指数值
    """
    
    if base_df is None or len(base_df) == 0:
        # 如果没有基础数据，使用默认参数生成
        mean_epu = 100
        std_epu = 30
    else:
        # 基于已有数据计算统计量
        mean_epu = base_df['epu_index'].mean()
        std_epu = base_df['epu_index'].std()
    
    # 生成日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # 生成EPU指数（带趋势和周期性）
    np.random.seed(42)
    n = len(date_range)
    
    # 基础趋势
    trend = np.linspace(0, 10, n)
    
    # 周期性成分（年度周期）
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 12)
    
    # 随机波动
    noise = np.random.normal(0, std_epu * 0.5, n)
    
    # 组合生成EPU指数
    epu_values = mean_epu + trend + seasonal + noise
    
    # 确保为正值
    epu_values = np.maximum(epu_values, 20)
    
    # 创建DataFrame
    extended_df = pd.DataFrame({
        'date': date_range,
        'epu_index': epu_values,
        'epu_count': np.random.randint(5, 20, n),
        'total_count': np.random.randint(20, 50, n),
    })
    
    # 计算相关统计量
    extended_df['epu_ratio'] = extended_df['epu_count'] / extended_df['total_count']
    extended_df['avg_confidence'] = np.random.uniform(0.7, 0.9, n)
    extended_df['epu_raw'] = extended_df['epu_ratio'] * 1000
    extended_df['epu_index_smooth'] = extended_df['epu_index'].rolling(window=3, center=True).mean()
    
    return extended_df


# ==================== 市场数据生成 ====================

def generate_market_data(epu_df):
    """
    生成与EPU指数对应的市场数据
    
    包括：
    - 市场波动率 (Volatility)
    - 换手率 (TurnoverRate)
    - 利率 (Interest)
    - Baker EPU指数（作为对比基准）
    """
    
    np.random.seed(42)
    n = len(epu_df)
    
    # 从EPU指数提取归一化信号
    epu_normalized = (epu_df['epu_index'] - epu_df['epu_index'].mean()) / epu_df['epu_index'].std()
    
    # 生成市场波动率（与EPU正相关，相关系数约0.7）
    volatility_base = 0.024  # 基础波动率
    volatility = volatility_base + 0.004 * epu_normalized + np.random.normal(0, 0.001, n)
    volatility = np.clip(volatility, 0.01, 0.05)
    
    # 生成换手率（与EPU弱正相关，相关系数约0.12）
    turnover_base = 0.593
    turnover = turnover_base + 0.05 * epu_normalized + np.random.normal(0, 0.1, n)
    turnover = np.clip(turnover, 0.2, 1.2)
    
    # 生成利率（与EPU弱负相关，相关系数约-0.22）
    interest_base = 0.004
    interest = interest_base - 0.0005 * epu_normalized + np.random.normal(0, 0.0005, n)
    interest = np.clip(interest, 0.001, 0.008)
    
    # 生成Baker EPU指数（作为对比基准，与DeepSeek EPU弱相关，相关系数约0.2）
    baker_base = 117.283
    baker_epu = baker_base + 10 * epu_normalized * 0.2 + np.random.normal(0, 30, n)
    baker_epu = np.maximum(baker_epu, 20)
    
    # 添加到DataFrame
    result_df = epu_df.copy()
    result_df['Volatility'] = volatility
    result_df['TurnoverRate'] = turnover
    result_df['Interest'] = interest
    result_df['EPU_Baker'] = baker_epu
    result_df['EPU_Deepseek'] = result_df['epu_index'] / 100 * 5.595  # 标准化到书中的尺度
    
    return result_df


# ==================== 主程序 ====================

def main():
    """主执行流程"""
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / 'output'
    cell_dir = output_root / 'cell04_生成EPU指数'
    cell_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("程序1：基于大语言模型的EPU指数生成")
    print("=" * 60)
    print()
    
    # 1. 读取新闻数据
    print("[步骤 1/5] 读取新闻数据...")
    news_file = 'data/sample_news.csv'
    
    if not os.path.exists(news_file):
        print(f"错误：找不到文件 {news_file}")
        print("请确保在 chapter6_code 目录下运行程序")
        return
    
    news_df = pd.read_csv(news_file)
    print(f"✓ 成功读取 {len(news_df)} 条新闻数据")
    print(f"  时间范围：{news_df['date'].min()} 至 {news_df['date'].max()}")
    print()
    
    # 2. 使用LLM分类新闻
    print("[步骤 2/5] 使用大语言模型分类新闻（判断是否与EPU相关）...")
    print("注：实际应用中应调用真实的LLM API，此处为模拟")
    classified_df = batch_classify_news(news_df, simulate_api_delay=False)
    
    epu_related_count = classified_df['epu_related'].sum()
    epu_ratio = epu_related_count / len(classified_df) * 100
    print(f"✓ 分类完成")
    print(f"  EPU相关新闻：{epu_related_count}/{len(classified_df)} ({epu_ratio:.1f}%)")
    print(f"  平均置信度：{classified_df['confidence'].mean():.3f}")
    print()
    
    # 3. 计算月度EPU指数
    print("[步骤 3/5] 计算月度EPU指数...")
    monthly_epu = calculate_monthly_epu(classified_df, baseline_date='2020-01')
    print(f"✓ 计算完成，共 {len(monthly_epu)} 个月的数据")
    print()
    
    # 4. 扩展数据到完整时间序列
    print("[步骤 4/5] 生成完整的月度时间序列（2020-01 至 2023-12）...")
    
    # 合并真实数据和扩展数据
    last_date = monthly_epu['date'].max()
    next_month = last_date + pd.DateOffset(months=1)
    
    extended_df = generate_additional_data(
        start_date=next_month.strftime('%Y-%m'),
        end_date='2023-12',
        base_df=monthly_epu
    )
    
    # 合并数据
    full_epu_df = pd.concat([
        monthly_epu[['date', 'epu_index', 'epu_index_smooth', 'epu_count', 
                     'total_count', 'epu_ratio', 'avg_confidence']],
        extended_df[['date', 'epu_index', 'epu_index_smooth', 'epu_count',
                    'total_count', 'epu_ratio', 'avg_confidence']]
    ], ignore_index=True)
    
    print(f"✓ 生成完整时间序列，共 {len(full_epu_df)} 个月")
    print(f"  时间范围：{full_epu_df['date'].min().strftime('%Y-%m')} 至 {full_epu_df['date'].max().strftime('%Y-%m')}")
    print()
    
    # 5. 生成市场数据
    print("[步骤 5/5] 生成相关市场数据...")
    final_df = generate_market_data(full_epu_df)
    print("✓ 市场数据生成完成")
    print()
    
    # 6. 保存结果
    output_file = 'data/epu_index.csv'
    final_df.to_csv(output_file, index=False)
    final_df.to_csv(cell_dir / 'epu_index.csv', index=False)
    classified_df.to_csv(cell_dir / 'classified_news.csv', index=False)
    news_df.to_csv(cell_dir / 'sample_news.csv', index=False)
    print(f"✓ 结果已保存到：{output_file}")
    print()
    
    # 7. 输出统计摘要
    print("=" * 60)
    print("数据统计摘要")
    print("=" * 60)
    print(f"EPU指数 (Deepseek):")
    print(f"  均值：{final_df['EPU_Deepseek'].mean():.3f}")
    print(f"  标准差：{final_df['EPU_Deepseek'].std():.3f}")
    print(f"  最小值：{final_df['EPU_Deepseek'].min():.3f}")
    print(f"  最大值：{final_df['EPU_Deepseek'].max():.3f}")
    print()
    print(f"EPU指数 (Baker):")
    print(f"  均值：{final_df['EPU_Baker'].mean():.3f}")
    print(f"  标准差：{final_df['EPU_Baker'].std():.3f}")
    print()
    print(f"市场波动率 (Volatility):")
    print(f"  均值：{final_df['Volatility'].mean():.4f}")
    print(f"  标准差：{final_df['Volatility'].std():.4f}")
    print()
    print(f"换手率 (TurnoverRate):")
    print(f"  均值：{final_df['TurnoverRate'].mean():.3f}")
    print(f"  标准差：{final_df['TurnoverRate'].std():.3f}")
    print()
    print(f"利率 (Interest):")
    print(f"  均值：{final_df['Interest'].mean():.4f}")
    print(f"  标准差：{final_df['Interest'].std():.4f}")
    print()
    print("=" * 60)
    print("程序执行完成！")
    print("=" * 60)
    print()
    print("下一步：运行 2_EPU_analysis.py 进行数据分析和图表生成")
    
    summary_path = cell_dir / 'summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        f.write("程序1：EPU 指数生成摘要\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"新闻条数: {len(news_df)}\n")
        f.write(f"EPU相关新闻条数: {epu_related_count}\n")
        f.write(f"EPU相关新闻比例: {epu_ratio:.1f}%\n")
        f.write(f"平均置信度: {classified_df['confidence'].mean():.3f}\n\n")
        f.write("EPU 指数统计（Deepseek）:\n")
        f.write(f"- 均值: {final_df['EPU_Deepseek'].mean():.3f}\n")
        f.write(f"- 标准差: {final_df['EPU_Deepseek'].std():.3f}\n")
        f.write(f"- 最小值: {final_df['EPU_Deepseek'].min():.3f}\n")
        f.write(f"- 最大值: {final_df['EPU_Deepseek'].max():.3f}\n")


if __name__ == "__main__":
    main()

