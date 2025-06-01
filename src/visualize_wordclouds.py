# 代码块：词云图生成
# 确保您已在当前环境中安装 wordcloud 和 Pillow：
# pip install wordcloud Pillow matplotlib pandas

import pandas as pd
import matplotlib.pyplot as plt
import os
import ast  # 用于转换 words 列
from wordcloud import WordCloud
import re # 用于过滤词语
from collections import Counter # 可以用于辅助检查词频，虽然词云图不直接用它

# --- 设置数据文件路径 ---
# 假设您在项目根目录或 notebooks 目录下运行此代码
# 如果您在 src 目录下运行，请调整路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 尝试找到项目根目录 (假设 output 目录在项目根目录下)
project_root = os.path.dirname(script_dir) # 如果在src下运行
# 如果在notebooks下运行，可能需要调整为：
# project_root = os.path.dirname(script_dir)
# 如果在项目根目录运行，project_root = script_dir


output_dir = os.path.join(project_root, 'output')
processed_data_with_words_path = os.path.join(output_dir, 'processed_data_with_words.csv')

# --- 加载数据 ---
print(f"正在从 {processed_data_with_words_path} 加载数据...")
try:
    # 检查文件是否存在
    if not os.path.exists(processed_data_with_words_path):
        raise FileNotFoundError(f"文件未找到: {processed_data_with_words_path}")

    df = pd.read_csv(processed_data_with_words_path, encoding='utf-8-sig')
    print("数据加载成功！")

    # 确保 'words' 列已从字符串转换为列表
    if df['words'].dtype == 'object' and isinstance(df['words'].iloc[0], str):
         df['words'] = df['words'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
         print("'words' 列已从字符串转换为列表。")

    # 确保 'comment_time' 列是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(df['comment_time']):
         df['comment_time'] = pd.to_datetime(df['comment_time'])
         print("'comment_time' 列已转换为 datetime 类型。")

    # 确保 'sentiment_category' 列存在（如果数据预处理时未生成）
    if 'sentiment_category' not in df.columns:
         print("正在生成 'sentiment_category' 列...")
         # 确保 sentiment_score 列存在
         if 'sentiment_score' not in df.columns:
             print("错误: 找不到 'sentiment_score' 列，无法生成 'sentiment_category' 列。请先运行数据预处理脚本。")
             exit()
         def get_sentiment_category(score):
             if score > 0.6:
                 return '正面'
             elif 0.4 <= score <= 0.6:
                 return '中性'
             else:
                 return '负面'
         df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_category)
         print("'sentiment_category' 列已生成。")


except FileNotFoundError:
    print(f"错误：未找到文件 {processed_data_with_words_path}")
    print("请检查文件路径是否正确，并确保已成功运行数据预处理脚本生成该文件。")
    print("如果您在 src 目录下运行，请检查 processed_data_with_words.csv 是否在 output 目录下。")
    exit() # 如果文件未找到，终止脚本
except Exception as e:
    print(f"加载或处理数据时发生错误：{e}")
    import traceback
    traceback.print_exc() # 打印详细错误栈
    exit() # 如果加载或处理失败，终止脚本


# --- 定义词语处理和过滤函数（与add_words_column.py中的过滤逻辑相同） ---
# 定义停用词列表 (与 add_words_column.py 中的保持一致)
stop_words = set([
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

# 将 words 列表转换为用于词云生成的文本字符串，并应用过滤
def prepare_text_for_wordcloud(words_list):
    # 确保words_list是列表且非空
    if not isinstance(words_list, list) or not words_list:
        return ""
    # 过滤停用词、标点、数字、英文（与add_words_column.py中的过滤逻辑相同）
    processed_words = [word for word in words_list
            if word not in stop_words
            and len(word.strip()) > 1  # 只保留长度大于1的词
            and not (isinstance(word, (int, float)) or (isinstance(word, str) and word.isdigit())) # 过滤数字
            and not (isinstance(word, str) and re.match(r'^[a-zA-Z]+$', word))  # 过滤纯英文字母
            and not (isinstance(word, str) and re.match(r'^\W+$', word))  # 过滤纯符号
    ]
    return ' '.join(processed_words)


# --- 生成词云图函数 ---
def generate_wordcloud(text, title, output_filename=None):
    """
    从文本生成并显示词云图。
    Args:
        text (str): 用于生成词云的文本字符串。
        title (str): 图表标题。
        output_filename (str, optional): 如果指定，将词云保存到 output 目录下。Defaults to None.
    """
    try:
        # !!! 设置中文字体文件路径 !!!
        # 请将这里的 'simhei.ttf' 替换为您系统中实际的、支持中文的字体文件路径
        # 例如：'C:/Windows/Fonts/simhei.ttf' 或 'C:/Windows/Fonts/simsun.ttc'
        # 您可以将字体文件复制到脚本所在目录，或者指定完整路径
        font_path = 'simhei.ttf' # 默认尝试当前脚本目录下的simhei.ttf

        # 如果simhei.ttf不在当前目录，请修改为完整路径，例如：
        # font_path = 'C:/Windows/Fonts/simhei.ttf'
        # 或者使用simsun.ttc
        # font_path = 'C:/Windows/Fonts/simsun.ttc'

        # 检查字体文件是否存在，如果不存在则给出警告并尝试使用默认字体
        if not os.path.exists(font_path) or font_path is None:
             print(f"警告: 未找到指定的字体文件 {font_path if font_path else 'None'}，词云图可能无法正常显示中文。请确保字体文件存在且路径正确。")
             font_path = None # 如果找不到指定字体，让wordcloud使用默认字体

        wordcloud = WordCloud(
            font_path=font_path,        # 指定字体路径
            width=800,                  # 宽度
            height=400,                 # 高度
            background_color='white',   # 背景颜色
            max_words=100,              # 最大词数
            # colormap='viridis',         # 配色方案 (可选，如果导致问题可以注释掉)
            prefer_horizontal=0.9,      # 优先水平排列
            min_font_size=10,           # 最小字号
            max_font_size=100,          # 最大字号
            random_state=42             # 随机种子，保证每次生成结果一致 (可选)
        )

        # 确保文本不为空或只包含空白字符
        if not text or not text.strip():
            print(f"警告：'{title}' 的文本为空或只包含空白字符，跳过生成词云。")
            return

        # 生成词云
        wordcloud.generate_from_text(text)

        # 显示或保存词云图
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off') # 不显示坐标轴
        plt.title(title, fontsize=15, pad=20) # 设置标题

        if output_filename:
             # 确保 output 目录存在
             os.makedirs(output_dir, exist_ok=True)
             output_path = os.path.join(output_dir, output_filename)
             try:
                 plt.savefig(output_path, dpi=300, bbox_inches='tight')
                 print(f"词云图已保存到: {output_path}")
             except Exception as e:
                  print(f"保存词云图到文件时出错: {str(e)}")


        plt.show() # 显示图表

    except Exception as e:
        # 打印详细的错误信息
        import traceback
        traceback.print_exc()
        print(f"生成词云图时出错 ('{title}'): {str(e)}")


# --- 执行词云图生成 ---

print("\n--- 开始生成词云图 ---")

# 1. 生成整体评论的词云图
print("\n正在生成整体评论词云图...")
# 将所有评论的 words 列表展平并处理成一个大字符串
all_words_list = [word for sublist in df['words'] if isinstance(sublist, list) for word in sublist]
all_text = prepare_text_for_wordcloud(all_words_list)
generate_wordcloud(all_text, '整体评论词云图', 'wordcloud_overall.png')


# 2. 按情感类别生成词云图
print("\n正在生成情感类别词云图...")
for sentiment in ['正面', '中性', '负面']:
    group = df[df['sentiment_category'] == sentiment]
    if not group.empty:
        # 合并该情感类别的所有词列表并处理
        group_words_list = [word for sublist in group['words'] if isinstance(sublist, list) for word in sublist]
        group_text = prepare_text_for_wordcloud(group_words_list)
        generate_wordcloud(group_text, f'{sentiment}评论词云图', f'wordcloud_{sentiment}.png')
    else:
        print(f"警告：没有找到 {sentiment} 评论，跳过生成词云。")


# 3. 按评分生成词云图
print("\n正在生成评分词云图...")
# 对评分进行排序，确保图表顺序一致
for score in sorted(df['score'].unique()):
    group = df[df['score'] == score]
    if not group.empty:
        # 合并该评分的所有词列表并处理
        group_words_list = [word for sublist in group['words'] if isinstance(sublist, list) for word in sublist]
        group_text = prepare_text_for_wordcloud(group_words_list)
        generate_wordcloud(group_text, f'{score}星评论词云图', f'wordcloud_score_{score}.png')
    else:
        print(f"警告：没有找到 {score} 星评论，跳过生成词云。")


# 4. 分析高互动评论（点赞数高于中位数）并生成词云图
print("\n正在分析高互动评论关键词...")
if not df.empty:
    median_useful = df['useful_count'].median()
    # 确保 median_useful 是数值类型，避免比较错误
    if pd.api.types.is_numeric_dtype(df['useful_count']) and not pd.isna(median_useful):
        high_interaction = df[df['useful_count'] > median_useful]
        if not high_interaction.empty:
            # 合并高互动评论的所有词列表并处理
            high_interaction_words_list = [word for sublist in high_interaction['words'] if isinstance(sublist, list) for word in sublist]
            high_interaction_text = prepare_text_for_wordcloud(high_interaction_words_list)
            generate_wordcloud(high_interaction_text, '高互动评论词云图', 'wordcloud_high_interaction.png')
        else:
            print("警告：没有找到高互动评论（点赞数高于中位数），跳过生成词云。")
    else:
         print("警告：'useful_count' 列数据类型异常或中位数为NaN，无法分析高互动评论。")

else:
     print("警告：DataFrame为空，无法分析高互动评论。")


print("\n--- 所有词云图生成请求已发出 ---")
