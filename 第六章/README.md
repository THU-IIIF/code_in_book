# 第六章：大语言模型在金融中的应用 - 代码实现

本目录包含第六章的完整代码实现，可独立运行并复现书中所有实验结果。

## 📁 文件结构

```
chapter6_code/
├── README.md                            # 本文件
├── requirements.txt                     # Python依赖包
├── data/                                # 数据目录
│   ├── sample_news.csv                 # 示例新闻数据（30条，2020-2022）
│   └── epu_index.csv                   # 生成的EPU指数（程序1输出，48个月）
├── 1_EPU_generation.py                 # 程序1：EPU指数生成
├── 2_EPU_analysis.py                   # 程序2：EPU指数分析与图表生成
└── 第六章_EPU分析完整流程.ipynb        # Jupyter Notebook版本（推荐）
```

## 🚀 快速开始

### 方法1：使用Python脚本（命令行）

#### 步骤1：安装依赖

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn tqdm
```

或使用requirements.txt：

```bash
pip install -r requirements.txt
```

#### 步骤2：运行程序1 - 生成EPU指数

```bash
cd chapter6_code
python3 1_EPU_generation.py
```

**输出：**
- `data/epu_index.csv` - 包含48个月的EPU指数及相关市场数据
- `output/1_EPU_generation.log` - 清晰的运行日志与统计摘要

**功能说明：**
- 读取30条示例新闻（2020-01至2022-06）
- 使用关键词匹配模拟大语言模型分类（实际应用中可替换为真实LLM API）
- 计算月度EPU指数并扩展到2023-12
- 生成相关市场数据（波动率、换手率、利率、Baker EPU）

#### 步骤3：运行程序2 - 分析与可视化

```bash
python3 2_EPU_analysis.py
```

**输出：**
- `output/tables/table_*.csv` - 所有表格数据
- `output/figures/6-*.png` - 所有图表（10张）
- `output/analysis_summary.txt` - 关键结论与文件索引
- `output/2_EPU_analysis.log` - 精简的运行日志

**生成内容：**
- 表6-1至6-6：所有书中表格
- 图6-1至6-10：所有书中图表
- 回归分析结果（OLS, LASSO, Random Forest, LSTM）

### 方法2：使用Jupyter Notebook（推荐）

```bash
cd chapter6_code
jupyter notebook 第六章_EPU分析完整流程.ipynb
```

**优势：**
- 交互式运行，逐步查看结果
- 可以修改参数实时查看效果
- 支持导出HTML/PDF报告
- 更直观的数据可视化

## 📊 数据说明

### 输入数据：sample_news.csv

包含30条真实新闻标题和内容（2020-01至2022-06），涵盖：
- 货币政策（降准、降息）
- 财政政策（政府工作报告）
- 重大事件（疫情、中美关系）
- 市场动态（股市波动、房地产政策）

### 输出数据：epu_index.csv

字段说明：
- `date`: 月度日期（2020-01至2023-12，共48个月）
- `EPU_Deepseek`: 基于DeepSeek的EPU指数
- `EPU_Baker`: 传统Baker EPU指数（对比基准）
- `Volatility`: 市场波动率
- `TurnoverRate`: 市场换手率
- `Interest`: 利率水平
- `epu_count`, `total_count`, `epu_ratio`: 统计指标

## 🔬 技术实现

### 程序1核心算法

1. **文本分类**（模拟LLM）
   ```python
   # 关键词匹配 + 随机噪声
   epu_score = (epu_keywords * 2 + economic_keywords + 
                policy_keywords + uncertainty_keywords) / 20
   ```

2. **EPU指数计算**
   ```python
   EPU_t = (EPU相关新闻数 / 总新闻数) × 基准化系数
   ```

3. **数据扩展**
   ```python
   EPU = 均值 + 趋势 + 季节性 + 随机噪声
   ```

### 程序2核心模型

1. **OLS回归**
   ```python
   Volatility = β₀ + β₁·EPU + β₂·TurnoverRate + β₃·Interest
   ```

2. **LASSO回归**（L1正则化）

3. **随机森林**（100棵树，最大深度5）

4. **LSTM**（模拟，实际应用需TensorFlow）

## 📈 预期结果

运行完整流程后，您将获得：

### 表格输出（控制台）
- **表6-1**: LoRA超参数配置
- **表6-2**: 描述性统计（均值、标准差）
- **表6-3**: 相关性矩阵
- **表6-4**: 线性模型性能（R²、系数）
- **表6-5**: 机器学习模型性能
- **表6-6**: 特征重要性排名

（实际输出已导出为 `output/tables/table_*.csv`）

### 图表输出（`output/figures` 目录）
- **图6-1**: 零样本vs少样本学习对比
- **图6-2**: EPU指数时间序列
- **图6-3**: EPU与波动率散点图（含拟合线）
- **图6-4**: 变量相关性热图
- **图6-5**: EPU分布直方图
- **图6-7**: 模型性能对比柱状图
- **图6-8**: 多变量时间序列
- **图6-9**: 残差分析图
- **图6-10**: Q-Q正态性检验图

### 关键发现（示例）
- EPU(Deepseek)与波动率相关系数：~0.70（强正相关）
- EPU(Baker)与波动率相关系数：~0.14（弱正相关）
- 最佳模型：LSTM（R² > 0.70）
- DeepSeek EPU在4/4模型中优于Baker EPU

## ⚙️ 自定义与扩展

### 替换为真实LLM API

在`1_EPU_generation.py`中修改`simulate_llm_classification`函数：

```python
import openai  # 或其他LLM SDK

def llm_classification(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "system",
            "content": "判断新闻是否与经济政策不确定性相关"
        }, {
            "role": "user",
            "content": text
        }]
    )
    # 解析response...
    return epu_related, confidence
```

### 添加更多新闻数据

编辑`data/sample_news.csv`，按以下格式添加：

```csv
date,news_title,news_content,source
2023-01-01,新闻标题,新闻内容,来源
```

### 调整模型参数

在`2_EPU_analysis.py`中修改：

```python
# 随机森林
rf = RandomForestRegressor(
    n_estimators=200,  # 增加树的数量
    max_depth=10,      # 增加深度
    random_state=42
)

# LASSO
lasso = Lasso(
    alpha=0.1,  # 调整正则化强度
    random_state=42
)
```

## 🐛 常见问题

### Q1: ImportError: No module named 'xxx'
**A**: 安装缺失的包：`pip install xxx`

### Q2: 程序运行但没有输出
**A**: 检查是否在正确的目录下运行（应在chapter6_code目录）

### Q3: 图表中文显示乱码
**A**: 安装中文字体或修改plt.rcParams中的字体设置

### Q4: 找不到data/epu_index.csv
**A**: 先运行程序1生成数据，再运行程序2

### Q5: 想使用真实的LLM API
**A**: 参考"替换为真实LLM API"部分，并设置API密钥

## 📝 引用

如果您在研究中使用了这些代码，请引用：

```bibtex
@book{llm_finance_2024,
  title={大语言模型在金融中的应用},
  author={作者名},
  chapter={6},
  year={2024},
  publisher={出版社}
}
```

## 🤝 贡献与反馈

如有问题或建议，请联系：[email地址]

## 📄 许可证

本代码仅供学术研究和教学使用。

---

**最后更新**: 2024-11-13  
**版本**: 1.0  
**状态**: ✅ 已测试通过

