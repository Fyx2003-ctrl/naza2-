import os
import json
import pandas as pd
from datetime import datetime
import re
from tqdm import tqdm
# import jieba # 如果新模型自带分词，可能不再需要jieba，但很多模型依赖外部分词或有自己的tokenizer
# from snownlp import SnowNLP # 移除snownlp
import shutil
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
import sys
import gc
import numpy as np
import logging

# 导入 transformers 相关的库
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class DataPreprocessor:
    def __init__(self, data_dir):
        """
        初始化数据预处理器
        :param data_dir: 原始数据目录
        """
        self.data_dir = data_dir
        self.raw_data = []
        self.processed_data = None
        self.cache_dir = os.path.join(os.path.dirname(data_dir), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.cache_dir, 'checkpoint.json')
        self.performance_log_file = os.path.join(self.cache_dir, 'performance.log')

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(os.path.dirname(data_dir), 'preprocessing.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 加载情感分析模型
        self._load_sentiment_model()

        # 清理系统资源
        self._cleanup_resources()

    def _load_sentiment_model(self):
        """加载深度学习情感分析模型和分词器"""
        self.logger.info("正在加载情感分析模型...")
        try:
            # 修改为您的本地模型路径
            local_model_path = os.path.join(os.path.dirname(self.data_dir), 'models', 'sentiment_model') # 请确保这个路径与您实际保存文件的路径一致
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.logger.info(f"情感分析模型加载成功，使用设备: {self.device}，从本地路径 {local_model_path}")
        except Exception as e:
            self.logger.error(f"情感分析模型加载失败: {str(e)}")
            self.logger.error("请检查本地模型路径是否正确，并确保该目录下包含了完整的模型文件。")
            raise

    def _print_progress(self, message):
        """打印进度信息，确保立即显示"""
        print(message, flush=True)
        sys.stdout.flush()

    def _get_cache_path(self, operation):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f'{operation}_cache.pkl')

    def _load_cache(self, operation):
        """加载缓存数据"""
        cache_path = self._get_cache_path(operation)
        # --- 注意：如果您修改了数据处理逻辑（例如更换情感模型），请手动删除 cache 目录下的相关文件，特别是 process_cache.pkl ---
        if os.path.exists(cache_path):
            try:
                self._print_progress(f"正在加载缓存: {cache_path} ...")
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self._print_progress("缓存加载成功。")
                    return data
            except Exception as e:
                self._print_progress(f"缓存加载失败: {e}。将重新进行处理。")
                # 如果缓存加载失败，可以考虑删除损坏的缓存文件
                # os.remove(cache_path)
                return None
        return None

    def _save_cache(self, operation, data):
        """保存缓存数据"""
        cache_path = self._get_cache_path(operation)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self._print_progress(f"缓存已保存到: {cache_path}")
        except Exception as e:
            self._print_progress(f"保存缓存 {cache_path} 失败: {e}")


    def _load_checkpoint(self):
        """加载检查点数据"""
        # --- 注意：如果您修改了数据处理逻辑（例如更换情感模型），旧的检查点可能不再适用，请手动删除 checkpoint.json ---
        if os.path.exists(self.checkpoint_file):
            try:
                self._print_progress(f"正在加载检查点: {self.checkpoint_file} ...")
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    self._print_progress("检查点加载成功。")
                    return checkpoint_data
            except Exception as e:
                self._print_progress(f"检查点加载失败: {e}。将从头开始处理。")
                # 如果检查点加载失败，可以考虑删除损坏的检查点文件
                # os.remove(self.checkpoint_file)
                return None
        return None

    def _save_checkpoint(self, processed_ids, total_count):
        """保存检查点数据"""
        checkpoint_data = {
            'processed_ids': list(processed_ids),
            'total_count': total_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
            self._print_progress(f"检查点已保存到: {self.checkpoint_file}")
        except Exception as e:
            self._print_progress(f"保存检查点 {self.checkpoint_file} 失败: {e}")


    def _cleanup_resources(self):
        """清理系统资源"""
        try:
            # 强制进行垃圾回收
            gc.collect()

            # 清理可能存在的jieba缓存（如果还用到的话）
            # if hasattr(jieba, 'dt'):
            #     jieba.dt.cache_file = None
            #     jieba.dt.initialize()

            # 深度学习模型通常不需要额外的清理，只要加载一次即可
            print("系统资源清理完成")
        except Exception as e:
            print(f"清理资源时出错: {str(e)}")

    def _log_performance(self, processed_count, elapsed_time, batch_size):
        """记录性能数据"""
        try:
            speed = processed_count / elapsed_time if elapsed_time > 0 else 0
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.performance_log_file, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} - 处理速度: {speed:.2f}条/秒, "
                       f"批次大小: {batch_size}, "
                       f"已处理: {processed_count}条\n")
        except Exception as e:
            print(f"记录性能数据时出错: {str(e)}")

    def merge_json_files(self):
        """合并所有JSON文件并保存原始数据副本"""
        # 检查缓存
        cached_data = self._load_cache('merge')
        if cached_data is not None:
            print("使用缓存的合并数据...")
            self.raw_data = cached_data
            return

        print("正在合并JSON文件...")
        all_reviews = []

        # 遍历所有JSON文件
        for file in tqdm(os.listdir(self.data_dir)):
            if file.endswith('.json'):
                file_path = os.path.join(self.data_dir, file)
                try: # 添加文件读取错误处理
                    with open(file_path, 'r', encoding='utf-8') as f:
                         # 尝试加载JSON，处理空文件或格式错误
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_reviews.extend(data)
                            else:
                                print(f"警告: 文件 {file} 内容不是列表，跳过。")
                        except json.JSONDecodeError:
                            print(f"警告: 文件 {file} 不是有效的JSON格式，跳过。")
                except Exception as e:
                    print(f"读取文件 {file} 时出错: {str(e)}")


        # 保存原始数据副本
        raw_data_path = os.path.join(os.path.dirname(self.data_dir), 'output', 'raw_data')
        os.makedirs(raw_data_path, exist_ok=True)

        # 保存为JSON
        with open(os.path.join(raw_data_path, 'raw_reviews.json'), 'w', encoding='utf-8') as f:
            json.dump(all_reviews, f, ensure_ascii=False, indent=4)

        # 保存为CSV
        # 转换为DataFrame时，处理可能的嵌套结构，确保所有键都存在
        df_raw = pd.json_normalize(all_reviews)
        # 扁平化嵌套的author和interaction字段，如果它们还是字典的话
        if 'author' in df_raw.columns and isinstance(df_raw['author'].iloc[0], dict):
             df_raw = pd.json_normalize(all_reviews) # 重新使用json_normalize处理
        elif 'interaction' in df_raw.columns and isinstance(df_raw['interaction'].iloc[0], dict):
             df_raw = pd.json_normalize(all_reviews) # 重新使用json_normalize处理

        df_raw.to_csv(os.path.join(raw_data_path, 'raw_reviews.csv'), index=False, encoding='utf-8-sig')

        print(f"原始数据已保存，共 {len(all_reviews)} 条评论")
        self.raw_data = all_reviews

        # 保存缓存
        self._save_cache('merge', all_reviews)


    def clean_text(self, text):
        """
        清理文本数据
        :param text: 原始文本
        :return: 清理后的文本
        """
        if not isinstance(text, str):
            return ""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 移除特殊字符，保留中文标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?，。！？\n]', '', text)
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def standardize_score(self, score):
        """
        标准化评分
        :param score: 原始评分
        :return: 标准化后的评分（1-5）
        """
        # 确保score是字符串或数字，并尝试转换为整数
        try:
            score_int = int(score)
            if 1 <= score_int <= 5:
                return score_int
        except (ValueError, TypeError):
            pass # 忽略转换错误

        # 如果是字符串，使用映射
        score_map = {
            '很差': 1,
            '较差': 2,
            '还行': 3,
            '推荐': 4,
            '力荐': 5
        }
        return score_map.get(str(score), 3)  # 默认值为3，确保使用字符串键

    # --- 修改点：修改处理单条评论的逻辑，使用新的情感分析模型并添加分词 ---
    def _process_single_review(self, review):
        """处理单条评论"""
        try:
            # 清理文本
            title = self.clean_text(review.get('title', ''))
            content = self.clean_text(review.get('comment_content', ''))

            # 跳过空评论
            if not content or len(content.strip()) == 0:
                return {'review_id': review.get('review_id', 'unknown'), 'filtered': 'empty_content'}

            # 标准化时间
            comment_time = review.get('comment_time')
            if comment_time:
                try:
                    comment_time = datetime.strptime(comment_time, '%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError) as e:
                    self.logger.error(f"时间解析错误 (ID: {review.get('review_id', 'unknown')}): {str(e)}")
                    comment_time = None
            else:
                comment_time = None

            # 标准化评分
            try:
                score = self.standardize_score(review.get('score', 3))
            except Exception as e:
                self.logger.error(f"评分标准化错误 (ID: {review.get('review_id', 'unknown')}): {str(e)}")
                score = 3

            # --- 修改点：添加分词逻辑 ---
            # words = []
            # if content and self.tokenizer: # 确保内容不为空且 tokenizer 已加载
            #      try:
            #          # 使用 tokenizer 进行分词。不同 tokenizer 的分词方式可能不同。
            #          # 例如，对于许多 tokenizer，可以直接使用 encode() 或 tokenize() 方法
            #          # 或者获取 input_ids 并转换为 tokens
            #          # 这里使用 tokenize() 作为示例，它返回一个字符串列表
            #          tokens = self.tokenizer.tokenize(content)
            #          # 如果需要进一步处理，例如去除标点和停用词，可以在这里添加逻辑
            #          # 暂时先将 token 作为 words 返回
            #          words = tokens
            #      except Exception as e:
            #          self.logger.error(f"分词失败 (ID: {review.get('review_id', 'unknown')}): {str(e)}")
            #          words = [] # 分词失败则返回空列表


            # 计算情感得分 (使用新的模型)
            sentiment_score = 0.5  # 默认中性
            if content and self.model and self.tokenizer: # 确保内容不为空且模型和tokenizer已加载
                try:
                    inputs = self.tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        scores = torch.softmax(outputs.logits, dim=1)
                        sentiment_score = scores[0][1].item()  # 获取正面情感的概率
                except Exception as e:
                    self.logger.error(f"情感分析失败 (ID: {review.get('review_id', 'unknown')}): {str(e)}")
                    sentiment_score = 0.5

            # 处理互动数据
            interaction_data = review.get('interaction', {})
            try:
                useful_count = int(interaction_data.get('useful_count', 0))
                useless_count = int(interaction_data.get('useless_count', 0))
                reply_count = int(interaction_data.get('reply_count', 0))
            except (ValueError, TypeError) as e:
                self.logger.error(f"互动数据解析错误 (ID: {review.get('review_id', 'unknown')}): {str(e)}")
                useful_count = useless_count = reply_count = 0

            # 构建处理后的评论数据字典
            processed_review = {
                'review_id': review.get('review_id', 'unknown'),
                'title': title,
                'content': content,
                'comment_time': comment_time,
                'score': score,
                'sentiment_score': sentiment_score,
                #'words': words,  # --- 修改点：添加 'words' 列 ---
                'author_name': review.get('author', {}).get('name', ''),
                'author_url': review.get('author', {}).get('url', ''),
                'useful_count': useful_count,
                'useless_count': useless_count,
                'reply_count': reply_count,
                'url': review.get('url', '')
            }

            return processed_review

        except Exception as e:
            self.logger.error(f"处理评论 {review.get('review_id', 'unknown')} 时发生未知错误: {str(e)}")
            return {'review_id': review.get('review_id', 'unknown'), 'error': str(e)}


    def process_data(self):
        """处理数据"""
        # 检查缓存
        # --- 重要：如果您修改了数据处理逻辑（例如更换情感模型、清洗逻辑等），请手动删除 cache 目录下的 process_cache.pkl 和 checkpoint.json 文件！ ---
        cached_data = self._load_cache('process')
        if cached_data is not None:
            self._print_progress("使用缓存的处理后数据...")
            if isinstance(cached_data, list): # 兼容旧的list缓存格式
                self.processed_data = pd.DataFrame(cached_data)
            else:
                self.processed_data = cached_data # dataframe缓存格式
            return

        self._print_progress("正在处理数据...")

        # 加载检查点
        # --- 重要：如果您修改了数据处理逻辑（例如更换情感模型、清洗逻辑等），旧的检查点可能不再适用，请手动删除 checkpoint.json 和 process_cache.pkl 文件！ ---
        checkpoint = self._load_checkpoint()
        processed_ids = set(checkpoint.get('processed_ids', [])) if checkpoint else set() # 使用get处理可能缺失的键
        total_count = checkpoint.get('total_count', len(self.raw_data)) if checkpoint else len(self.raw_data)


        if processed_ids:
            self._print_progress(f"发现未完成的处理任务，已处理 {len(processed_ids)}/{total_count} 条数据")
            # 过滤出未处理的数据
            remaining_data = [review for review in self.raw_data
                            if review.get('review_id', 'unknown') not in processed_ids] # 使用get处理可能缺失的键
        else:
            remaining_data = self.raw_data

        # 增加进程数，但保持合理范围
        # 对于深度学习模型，如果模型较大，并行进程数可能需要减少，以免内存不足
        # 如果有GPU，通常设置device=0并在单个进程中处理，或者使用GPU多进程库
        # 在CPU模式下，适当的进程数可以加快处理速度，但要考虑内存和CPU核心数
        optimal_workers = max(1, min(os.cpu_count(), 4)) # 限制最大进程数，例如不超过CPU核心数且不超过4个
        # 如果检测到GPU，并且想使用GPU加速，将工作进程设为1，因为pipeline会自动处理批次
        if torch.cuda.is_available():
             optimal_workers = 1 # 如果有GPU，通常只用1个进程，让GPU处理

        self._print_progress(f"使用 {optimal_workers} 个工作进程进行并行处理 (情感分析在每个进程中独立加载模型)")
        # --- 注意：如果模型加载很耗时或占用内存大，多进程模式下每个进程都会加载模型，可能导致内存或启动时间问题。
        # 如果遇到问题，可以考虑将 optimal_workers 设为 1，或者寻找支持模型共享的多进程处理方式。 ---


        processed_results = []
        start_time = time.time()
        last_update = start_time
        last_progress = 0
        processed_count = 0
        error_count = 0 # 记录处理失败的评论数量
        filtered_count = 0 # 记录因空内容等被过滤的评论数量


        # 使用进程池并行处理数据
        # 注意：深度学习模型在多进程环境下可能需要特殊的处理，尤其是GPU。
        # 对于CPU模式，进程池是可行的。但如果模型加载占用大量内存，进程数过多可能导致OOM。
        # 如果遇到内存问题，请将optimal_workers设置为1
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # 将数据分成批次处理 (对于进程池，批次大小可以稍微大一些)
            batch_size = 500  # 每批处理500条数据
            total_batches = (len(remaining_data) + batch_size - 1) // batch_size

            futures = []
            batch_review_ids = {} # 记录每个future对应的review_id

            # 提交所有批次的任务
            self._print_progress("提交处理任务到进程池...")
            # 对于每个评论，提交一个独立的处理任务
            for review in remaining_data:
                 future = executor.submit(self._process_single_review, review)
                 futures.append(future)
                 batch_review_ids[future] = review.get('review_id', 'unknown')


            self._print_progress(f"共提交 {len(futures)} 个处理任务")

            # 收集结果并处理
            # 使用 as_completed 来按任务完成顺序处理结果
            results_iterator = tqdm(as_completed(futures), total=len(futures),
                                     desc="处理进度",
                                     unit="条",
                                     mininterval=0.5) # 调整进度条更新频率


            for future in results_iterator:
                 # 在获取结果前，先尝试获取对应的 review_id，用于错误日志
                 review_id = batch_review_ids.get(future, 'unknown')
                 try:
                     result = future.result()
                     # 检查返回结果是否是错误或过滤标记
                     if isinstance(result, dict) and 'filtered' in result:
                         filtered_count += 1
                     elif isinstance(result, dict) and 'error' in result:
                         error_count += 1
                         self._print_progress(f"处理评论 {review_id} 失败: {result.get('error', '未知错误信息')}")
                     elif result is not None: # 成功处理，且结果不是None
                         processed_results.append(result)
                         processed_ids.add(review_id)
                         processed_count += 1
                     else: # 返回结果是None，也计入错误
                          error_count += 1
                          self._print_progress(f"处理评论 {review_id} 返回结果为 None")


                     # 每处理固定数量的评论后更新检查点和进度条
                     if processed_count % 100 == 0 and processed_count > last_progress: # 每处理100条更新一次
                         self._save_checkpoint(processed_ids, total_count)
                         current_time = time.time()
                         elapsed_time = current_time - start_time
                         if elapsed_time > 0 and processed_count > 0:
                             speed = processed_count / elapsed_time
                             results_iterator.set_postfix(speed=f"{speed:.1f}条/s", errors=error_count, filtered=filtered_count)
                         last_progress = processed_count

                 except Exception as e:
                     # 处理从 future.result() 获取结果时发生的未知错误
                     error_count += 1
                     self._print_progress(f"获取评论 {review_id} 结果时发生未知错误: {str(e)}")
                     # 可以在这里添加更详细的错误记录

            # 最终保存检查点和缓存
            self._save_checkpoint(processed_ids, total_count) # 最终保存一次检查点

            # 处理完成后，将最终结果转换为DataFrame
            if processed_results:
                self.processed_data = pd.DataFrame(processed_results)
                self._save_cache('process', self.processed_data)
                self._print_progress(f"\n数据处理完成，共处理 {len(self.processed_data)} 条有效评论")
                self._print_progress(f"过滤掉 {filtered_count} 条无效评论，处理失败 {error_count} 条")
            else:
                self._print_progress("未处理任何有效评论")
                self._print_progress(f"过滤掉 {filtered_count} 条无效评论，处理失败 {error_count} 条")

        # 在主进程中手动触发一次垃圾回收和资源清理
        gc.collect()
        self._cleanup_resources()



    def save_processed_data(self, output_dir):
        """保存处理后的数据"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        print("正在保存处理后的数据...")

        try:
            # 确保数据是DataFrame格式
            if not isinstance(self.processed_data, pd.DataFrame):
                 if isinstance(self.processed_data, list):
                    self.processed_data = pd.DataFrame(self.processed_data)
                 else:
                     print("警告: 处理后的数据不是DataFrame或列表，无法保存。")
                     return

            # 保存为CSV文件
            csv_path = os.path.join(output_dir, 'processed_data.csv')
            # 确保时间列格式正确再保存
            if 'comment_time' in self.processed_data.columns and pd.api.types.is_datetime64_any_dtype(self.processed_data['comment_time']):
                 self.processed_data['comment_time'] = self.processed_data['comment_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

            self.processed_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"数据已保存到CSV文件: {csv_path}")

            # 保存为JSON文件
            json_path = os.path.join(output_dir, 'processed_data.json')
            self.processed_data.to_json(json_path, orient='records', force_ascii=False, indent=4)
            print(f"数据已保存到JSON文件: {json_path}")

        except Exception as e:
            print(f"保存数据时出错: {str(e)}")
            # 尝试保存备份数据
            try:
                backup_path = os.path.join(output_dir, 'processed_data_backup.json')
                if isinstance(self.processed_data, pd.DataFrame):
                    data_to_save = self.processed_data.to_dict('records')
                elif isinstance(self.processed_data, list):
                    data_to_save = self.processed_data
                else:
                     data_to_save = [] # 无法保存，设为空列表

                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=4)
                print(f"已保存备份数据到: {backup_path}")
            except Exception as backup_error:
                print(f"保存备份数据时出错: {str(backup_error)}")

    def generate_comparison_report(self, output_dir):
        """
        生成数据对比报告
        :param output_dir: 输出目录
        """
        if self.processed_data is not None and not self.processed_data.empty: # 检查数据是否为空
            report_dir = os.path.join(output_dir, 'reports')
            os.makedirs(report_dir, exist_ok=True)

            # 计算基本统计信息
            stats = {
                '原始数据': {
                    '总评论数': len(self.raw_data) if self.raw_data is not None else 0,
                    '有效评论数': len(self.processed_data),
                    '过滤率': f"{((len(self.raw_data) - len(self.processed_data)) / (len(self.raw_data) or 1) * 100):.2f}%" # 处理raw_data为空的情况
                },
                # 确保score列存在且非空
                '评分分布': self.processed_data['score'].value_counts().to_dict() if 'score' in self.processed_data.columns and not self.processed_data['score'].empty else {},
                 # 确保sentiment_score列存在且非空
                '情感得分': {
                    '平均值': self.processed_data['sentiment_score'].mean() if 'sentiment_score' in self.processed_data.columns and not self.processed_data['sentiment_score'].empty else None,
                    '中位数': self.processed_data['sentiment_score'].median() if 'sentiment_score' in self.processed_data.columns and not self.processed_data['sentiment_score'].empty else None,
                    '标准差': self.processed_data['sentiment_score'].std() if 'sentiment_score' in self.processed_data.columns and not self.processed_data['sentiment_score'].empty else None
                }
            }

            # 保存报告
            report_path = os.path.join(report_dir, 'preprocessing_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=4)

            print(f"数据对比报告已保存到 {report_path}")
        else:
            print("处理后的数据为空，跳过生成报告。")

    def analyze_sentiment(self, text):
        """使用预训练模型进行情感分析"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                sentiment_score = scores[0][1].item()  # 获取正面情感的概率
                
            return sentiment_score
        except Exception as e:
            self.logger.error(f"情感分析失败: {str(e)}")
            return 0.5  # 发生错误时返回中性分数

    def process_comments(self, df):
        """处理评论数据"""
        self.logger.info("开始处理评论数据...")
        
        # 情感分析
        self.logger.info("正在进行情感分析...")
        df['sentiment_score'] = df['content'].apply(self.analyze_sentiment)
        
        # 计算情感分布
        sentiment_stats = {
            'positive': len(df[df['sentiment_score'] > 0.6]),
            'neutral': len(df[(df['sentiment_score'] >= 0.4) & (df['sentiment_score'] <= 0.6)]),
            'negative': len(df[df['sentiment_score'] < 0.4])
        }
        
        self.logger.info(f"情感分析完成，分布情况：{sentiment_stats}")
        
        return df


def main():
    # 设置路径
    # 确保这里的路径与您的项目结构一致
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(current_dir, 'data') # 假设原始json文件在 data 目录下
    output_dir = os.path.join(current_dir, 'output') # 处理后的文件保存在 output 目录下

    # 创建预处理器实例
    preprocessor = DataPreprocessor(data_dir)

    # 执行数据处理流程
    preprocessor.merge_json_files() # 合并原始数据
    preprocessor.process_data() # 处理数据，包括情感分析
    preprocessor.save_processed_data(output_dir) # 保存处理后的数据
    preprocessor.generate_comparison_report(output_dir) # 生成报告

if __name__ == '__main__':
    main()
