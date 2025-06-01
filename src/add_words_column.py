# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter
import jieba

# --- 设置路径 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, 'output')
processed_data_path = os.path.join(output_dir, 'processed_data.csv')
output_processed_data_path = os.path.join(output_dir, 'processed_data.csv')  # 直接覆盖原文件

# --- 加载情感分析模型的 tokenizer ---
# 注意：这里仅加载tokenizer用于分词过滤，不进行情感分析
local_model_path = os.path.join(project_root, 'models', 'sentiment_model')

try:
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    print("情感分析模型的 tokenizer 加载成功。")
except Exception as e:
    print(f"情感分析模型的 tokenizer 加载失败: {str(e)}")
    # 在实际运行中，如果分词依赖tokenizer，加载失败应退出或使用备选方案
    # exit() # 如果分词依赖tokenizer，可以 uncomment 此行

# --- 定义分词函数 ---
def tokenize_text(text, tokenizer):
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        # 使用jieba进行分词
        words = jieba.cut(text)
        
        # 停用词列表 - 优化后的版本（与可视化部分保持一致）
        stopwords = set([
            # 程度词
            '很', '太', '非常', '十分', '特别', '比较', '更加', '最', '更', '极', '极其', '相当', '稍微', '有点',
            # 时间词
            '已经', '曾经', '总是', '经常', '从来', '一直', '马上', '立刻', '刚刚', '终于',
            # 代词
            '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '自己', '别人', '大家',
            # 连词
            '和', '与', '及', '而', '或', '但', '然而', '因此', '所以', '因为', '由于', '如果', '那么', '虽然', '可是',
            # 语气词
            '的', '了', '是', '在', '有', '就', '都', '也', '还', '又', '才', '并', '没有', '不是',
            # 数量词
            '一个', '一种', '一些', '有些', '这个', '那个', '那些', '这些', '这', '那', '之', '总', '各', '位', '种', '些',
            # 其他无意义词
            '等等', '一样', '一般', '随便', '任何', '什么', '怎样', '如何', '为什么', '哪里', '何时', '多久', '多少', '什么样', '怎么样',
            '一方面', '另一方面', '与此同时', '况且', '何况', '再说', '而是', '不及', '与其', '为了', '以便', '以免', '不论', '不管',
            '即使', '既然', '自从', '直到', '因为', '由于', '除非', '与其说', '不如说', '之类', '以内', '以外', '以上', '以下',
            '之前', '之后', '之间', '之中', '之外', '以来'
        ])
        
        # 过滤停用词、标点、数字、英文
        filtered_words = [
            word for word in words
            if word not in stopwords
            and len(word.strip()) > 1  # 只保留长度大于1的词
            and not word.isdigit()  # 过滤纯数字
            and not re.match(r'^[a-zA-Z]+$', word)  # 过滤纯英文字母
            and not re.match(r'^\W+$', word)  # 过滤纯符号
        ]

        return filtered_words
    except Exception as e:
        print(f"分词函数错误: {str(e)}")
        return []

# --- 主逻辑 ---
if __name__ == "__main__":
    print(f"正在从 {processed_data_path} 加载数据...")
    try:
        df = pd.read_csv(processed_data_path, encoding='utf-8-sig')
        # 确保 comment_time 是 datetime 类型，以便后续使用 .dt 访问器
        df['comment_time'] = pd.to_datetime(df['comment_time'])
        print("数据加载成功。")
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        exit()

    print("正在为 'content' 列进行分词并生成 'words' 列...")
    # 使用 tqdm 的 progress_apply 显示进度条
    tqdm.pandas(desc="分词进度")
    # 注意：tokenize_text 函数不再需要tokenizer参数，因为jieba不依赖 transformers tokenizer
    # 如果您的jieba分词需要加载自定义词典，请确保在此之前加载
    df['words'] = df['content'].progress_apply(lambda x: tokenize_text(x, None)) # tokenizer参数传入None或移除
    print("'words' 列已生成。")

    # 检查新列
    print("\nDataFrame 列信息（包含新添加的 'words' 列）：")
    df.info()
    print("\n'words' 列前 5 行示例：")
    print(df['words'].head())

    # 定义情感类别（如果sentiment_category列还不存在）
    # 注意：这里只是根据sentiment_score创建类别列，sentiment_score本身应该在preprocessing.py中计算
    if 'sentiment_category' not in df.columns:
        print("正在生成 'sentiment_category' 列...")
        def get_sentiment_category(score):
            if score > 0.6:
                return '正面'
            elif 0.4 <= score <= 0.6:
                return '中性'
            else:
                return '负面'
        df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_category)
        print("'sentiment_category' 列已生成。")

    # 保存处理后的数据（直接覆盖原文件）
    print(f"\n正在保存处理后的数据到 {output_processed_data_path}...")
    try:
        # 在保存前，如果 comment_time 是 datetime 对象，转换为字符串格式
        if pd.api.types.is_datetime64_any_dtype(df['comment_time']):
            df['comment_time'] = df['comment_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df.to_csv(output_processed_data_path, index=False, encoding='utf-8-sig')
        print("数据保存成功。")
    except Exception as e:
        print(f"保存数据时出错: {str(e)}")

    print("\n'add_words_column.py' 脚本执行完成。")
