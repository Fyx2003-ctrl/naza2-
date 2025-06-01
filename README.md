# 哪吒2电影评论数据分析项目

## 项目简介
本项目旨在对电影《哪吒2》的豆瓣影评数据进行全面的数据分析与可视化。项目涵盖了从数据爬取、预处理到文本分析和数据可视化的整个流程。通过分析观众评论，旨在深入了解观众对电影的看法、情感倾向以及关注焦点，为电影行业提供有价值的市场洞察和决策支持。

## 项目结构
```
naza2_analysis/
├── data/                 # 原始数据目录
├── src/                  # 源代码
│   ├── naza2.py          # 豆瓣影评爬虫
│   ├── preprocessing.py  # 数据预处理
│   ├── analysis.py       # 文本分析
│   ├── visualization.py  # 数据可视化
│   └── utils.py          # 工具函数
├── output/              # 输出结果
│   ├── processed_data/  # 处理后的数据
│   └── visualizations/  # 可视化图表
├── docs/                # 文档
│   └── analysis_report.md # 分析报告
├── cache/              # 缓存文件
├── models/             # 模型文件
├── notebooks/          # Jupyter notebooks
└── requirements.txt    # 项目依赖
```

## 功能特点
1. 数据爬取
   - 从豆瓣电影页面爬取影评数据 (支持同步/异步方式)
   - 数据清洗和初步处理

2. 数据预处理
   - 文本清洗 (使用正则表达式和停用词过滤)
   - 情感分析 (使用基于Transformers库的模型)
   - 分词处理 (使用jieba库)
   - 数据标准化

3. 文本分析
   - 情感趋势分析
   - 关键词提取 (使用jieba库)
   - 主题模型分析 (使用gensim库的LDA模型)
   - 负面评论分析

4. 数据可视化
   - 情感趋势图 (使用matplotlib和seaborn库)
   - 词云图 (使用wordcloud库，并指定中文字体)
   - 主题分布图
   - 负面评论分析图

## 环境要求
- Python 3.8+
- 依赖包：见 requirements.txt

## 安装说明
1. 克隆项目
```bash
git clone https://github.com/Fyx2003-ctrl/naza2-.git
cd naza2_analysis
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用说明
1. 数据爬取
```bash
python src/naza2.py
```

2. 数据预处理
```bash
python src/preprocessing.py
```

3. 文本分析
```bash
python src/analysis.py
```

4. 生成可视化
```bash
python src/visualization.py
```

## 分析结果
详细的分析结果请参见 [分析报告](docs/analysis_report.md)

## 主要发现
1. 评论情感分布
   - 正面评论: 88.0%
   - 中性评论: 4.2%
   - 负面评论: 7.8%

2. 时间趋势
   - 评论热度持续稳定
   - 日均评论量保持较高水平

3. 主题分析
   - 剧情评价
   - 人物塑造
   - 视觉效果
   - 情感表达
   - 制作质量

## 贡献指南
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证
MIT License

## 联系方式
vx:fyxacxg1379 