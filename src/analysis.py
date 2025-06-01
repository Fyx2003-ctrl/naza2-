import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import jieba.analyse
from datetime import datetime, timedelta
import json
import os

class TextAnalyzer:
    def __init__(self, data_path):
        """
        初始化文本分析器
        :param data_path: 处理后的数据文件路径
        """
        self.data = pd.read_csv(data_path)
        self.data['comment_time'] = pd.to_datetime(self.data['comment_time'])
        
    def get_time_series_analysis(self):
        """
        时间序列分析
        :return: 按时间统计的情感得分和评论数量
        """
        # 按天统计
        daily_stats = self.data.groupby(self.data['comment_time'].dt.date).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'score': 'mean'
        }).reset_index()
        
        daily_stats.columns = ['date', 'avg_sentiment', 'sentiment_std', 'comment_count', 'avg_score']
        
        return daily_stats
    
    def get_keywords_analysis(self, top_n=20):
        """
        关键词分析
        :param top_n: 返回前N个关键词
        :return: 关键词及其TF-IDF值
        """
        # 合并所有评论内容
        all_text = ' '.join(self.data['content'].fillna(''))
        
        # 使用jieba提取关键词
        keywords = jieba.analyse.extract_tags(all_text, topK=top_n, withWeight=True)
        
        return keywords
    
    def get_sentiment_trend(self, window_size='1D'):
        """
        情感趋势分析
        :param window_size: 时间窗口大小
        :return: 情感趋势数据
        """
        # 按时间窗口计算情感得分
        sentiment_trend = self.data.set_index('comment_time').rolling(window=window_size).agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).reset_index()
        
        sentiment_trend.columns = ['time', 'avg_sentiment', 'sentiment_std', 'comment_count']
        
        return sentiment_trend
    
    def get_topic_modeling(self, num_topics=5):
        """
        主题模型分析
        :param num_topics: 主题数量
        :return: 主题词及其权重
        """
        # 准备文档
        texts = self.data['words'].tolist()
        
        # 创建词典
        dictionary = corpora.Dictionary(texts)
        
        # 创建文档-词频矩阵
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # 训练LDA模型
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary,
            passes=10
        )
        
        # 获取每个主题的关键词
        topics = []
        for topic_id in range(num_topics):
            topic_words = lda_model.show_topic(topic_id, topn=10)
            topics.append({
                'topic_id': topic_id,
                'words': topic_words
            })
        
        return topics
    
    def get_negative_review_analysis(self):
        """
        负面评论分析
        :return: 负面评论的关键词和主题
        """
        # 筛选负面评论（情感得分小于0.3）
        negative_reviews = self.data[self.data['sentiment_score'] < 0.3]
        
        # 提取负面评论关键词
        negative_text = ' '.join(negative_reviews['content'].fillna(''))
        negative_keywords = jieba.analyse.extract_tags(negative_text, topK=20, withWeight=True)
        
        # 对负面评论进行主题分析
        negative_texts = negative_reviews['words'].tolist()
        dictionary = corpora.Dictionary(negative_texts)
        corpus = [dictionary.doc2bow(text) for text in negative_texts]
        
        lda_model = models.LdaModel(
            corpus=corpus,
            num_topics=3,
            id2word=dictionary,
            passes=10
        )
        
        negative_topics = []
        for topic_id in range(3):
            topic_words = lda_model.show_topic(topic_id, topn=10)
            negative_topics.append({
                'topic_id': topic_id,
                'words': topic_words
            })
        
        return {
            'keywords': negative_keywords,
            'topics': negative_topics,
            'review_count': len(negative_reviews)
        }
    
    def save_analysis_results(self, output_dir):
        """
        保存分析结果
        :param output_dir: 输出目录
        """
        results = {
            'time_series': self.get_time_series_analysis().to_dict(orient='records'),
            'keywords': self.get_keywords_analysis(),
            'sentiment_trend': self.get_sentiment_trend().to_dict(orient='records'),
            'topics': self.get_topic_modeling(),
            'negative_analysis': self.get_negative_review_analysis()
        }
        
        # 保存为JSON
        output_path = os.path.join(output_dir, 'analysis_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"分析结果已保存到 {output_path}")

def main():
    # 设置路径
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(current_dir, 'output', 'processed_reviews.csv')
    output_dir = os.path.join(current_dir, 'output')
    
    # 创建分析器实例
    analyzer = TextAnalyzer(data_path)
    
    # 执行分析并保存结果
    analyzer.save_analysis_results(output_dir)

if __name__ == '__main__':
    main() 