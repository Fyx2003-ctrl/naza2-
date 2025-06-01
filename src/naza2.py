import requests
from lxml import etree
import pandas as pd
import time
from urllib.parse import urljoin
import json
import random
from datetime import datetime
import re
import aiohttp  # 用于异步HTTP请求
import asyncio  # 用于协程操作
from aiohttp import ClientTimeout
import aiofiles  # 用于异步文件操作
from asyncio import Semaphore  # 用于控制并发数量

# 配置常量
MAX_CONCURRENT_REQUESTS = 5  # 最大并发请求数
REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
RETRY_TIMES = 3  # 重试次数

#UA伪装和cookie配置
cookie = 'bid=hS1mV-zLr9k; _pk_id.100001.4cf6=0b7a911aa330ae27.1743917069.; __utmz=30149280.1743917069.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmz=223695111.1743917069.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); ll="108305"; _vwo_uuid_v2=D8189A6EAEA3B83B65F553FF27FA9F890|17856c277b6038a483fabb122fb0046b; ap_v=0,6.0; _pk_ses.100001.4cf6=1; __utma=30149280.2020998894.1743917069.1747567436.1747700843.14; __utmc=30149280; __utma=223695111.146931643.1743917069.1747567436.1747700843.14; __utmb=223695111.0.10.1747700843; __utmc=223695111; push_noty_num=0; push_doumail_num=0; __utmv=30149280.27917; __utmt=1; __utmb=30149280.14.10.1747700843; dbcl2="279173128:++oSEV0Lwk4"; ck=Msc7'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://movie.douban.com/',
    'Cookie': cookie
}

def get_page(url, retry=3):
    """获取页面内容，添加随机延迟和重试机制"""
    for i in range(retry):
        try:
            # 添加随机延迟，避免被封
            time.sleep(random.uniform(1, 3))
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'
            
            # 检查是否需要登录
            if '登录' in response.text and '注册' in response.text:
                print("检测到需要登录，请更新cookie")
                return None
                
            return response.text
        except Exception as e:
            print(f"请求失败，正在重试 ({i+1}/{retry}): {str(e)}")
            time.sleep(5)  # 失败后等待更长时间
    return None

def clean_text(text):
    """清理文本数据，但保留换行符"""
    if not text:
        return ""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 保留换行符，只清理多余的空白字符
    text = re.sub(r'[ \t]+', ' ', text)
    # 移除特殊字符，但保留中文标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?，。！？\n]', '', text)
    return text.strip()

def get_reviews_url(movie_url):
    """获取影评页面URL"""
    try:
        html = get_page(movie_url)
        if not html:
            return None
        html = etree.HTML(html)
        reviews_url = html.xpath('//*[@id="reviews-wrapper"]/header/h2/span/a/@href')
        if reviews_url:
            return urljoin(movie_url, reviews_url[0])
        return None
    except Exception as e:
        print(f"获取影评URL失败: {str(e)}")
        return None

def extract_reviews(reviews_url, max_pages=390):
    """提取所有评论"""
    all_reviews = []
    page = 0
    error_count = 0
    max_errors = 5  # 最大连续错误次数
    
    while page < max_pages and error_count < max_errors:
        try:
            current_url = f"{reviews_url}?start={page*20}" if page > 0 else reviews_url
            print(f"正在处理第 {page + 1}/{max_pages} 页: {current_url}")
            
            html = get_page(current_url)
            if not html:
                error_count += 1
                print(f"获取页面失败，错误计数: {error_count}")
                time.sleep(5)  # 增加等待时间
                continue
            
            html = etree.HTML(html)
            # 修改XPath以匹配实际的评论列表
            review_items = html.xpath('//div[contains(@class, "review-list")]//div[contains(@class, "review-item")]')
            
            if not review_items:
                error_count += 1
                print(f"未找到评论，错误计数: {error_count}")
                time.sleep(5)  # 增加等待时间
                continue
            
            error_count = 0  # 重置错误计数
            print(f"找到 {len(review_items)} 条评论")
            
            for item in review_items:
                try:
                    review_id = item.xpath('./@id')[0]
                    
                    # 获取评论标题
                    title = item.xpath('.//h2/a/text()')[0].strip()
                    title = clean_text(title)
                    
                    # 获取评分
                    score_elements = item.xpath('.//span[contains(@class, "allstar")]/@title')
                    score = score_elements[0] if score_elements else "未评分"
                    
                    # 获取评论时间
                    comment_time = item.xpath('.//span[@class="main-meta"]/text()')[0].strip()
                    try:
                        comment_time = datetime.strptime(comment_time, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        comment_time = None
                    
                    # 获取评论者信息
                    author_name = item.xpath('.//a[@class="name"]/text()')[0].strip()
                    author_url = item.xpath('.//a[@class="name"]/@href')[0]
                    
                    # 获取评论内容
                    short_content = item.xpath('.//div[@class="short-content"]/text()')
                    short_content = '\n'.join([p.strip() for p in short_content if p.strip()])
                    short_content = clean_text(short_content)
                    
                    # 获取互动数据，增加空值处理
                    useful_count = item.xpath('.//span[@id="r-useful_count-{}"]/text()'.format(review_id))
                    useful_count = int(useful_count[0].strip()) if useful_count and useful_count[0].strip() else 0
                    
                    useless_count = item.xpath('.//span[@id="r-useless_count-{}"]/text()'.format(review_id))
                    useless_count = int(useless_count[0].strip()) if useless_count and useless_count[0].strip() else 0
                    
                    reply_count = item.xpath('.//a[contains(@class, "reply")]/text()')
                    reply_count = int(reply_count[0].strip().replace('回应', '')) if reply_count and reply_count[0].strip() else 0
                    
                    # 获取详细评论内容
                    comment_url = item.xpath('.//h2/a/@href')[0]
                    comment_html = get_page(comment_url)
                    if comment_html:
                        comment_tree = etree.HTML(comment_html)
                        comment_content = comment_tree.xpath('//div[@class="review-content clearfix"]/p/text()')
                        if not comment_content:
                            comment_content = [short_content]
                        comment_content = '\n'.join([p.strip() for p in comment_content])
                        comment_content = clean_text(comment_content)
                    else:
                        comment_content = short_content
                    
                    # 构建评论数据字典
                    review_data = {
                        'review_id': review_id,
                        'title': title,
                        'score': score,
                        'comment_time': comment_time,
                        'comment_content': comment_content,
                        'author': {
                            'name': author_name,
                            'url': author_url
                        },
                        'interaction': {
                            'useful_count': useful_count,
                            'useless_count': useless_count,
                            'reply_count': reply_count
                        },
                        'url': comment_url
                    }
                    
                    # 只添加非空评论
                    if comment_content and len(comment_content.strip()) > 0:
                        all_reviews.append(review_data)
                        print(f"成功提取评论: {title}")
                    
                except Exception as e:
                    print(f"处理评论项时出错: {str(e)}")
                    continue
            
            # 每处理10页保存一次数据
            if (page + 1) % 10 == 0:
                save_to_csv(all_reviews, f'naza2_reviews_page_{page+1}.csv')
                save_to_json(all_reviews, f'naza2_reviews_page_{page+1}.json')
                print(f"已保存第 {page + 1} 页数据")
            
            page += 1
            time.sleep(random.uniform(2, 4))  # 增加页面间延迟
            
        except Exception as e:
            print(f"处理页面时出错: {str(e)}")
            error_count += 1
            time.sleep(5)  # 增加错误后等待时间
            continue
    
    return all_reviews

def save_to_csv(reviews, output_file):
    """保存评论数据到CSV文件"""
    try:
        flattened_reviews = []
        for review in reviews:
            flat_review = {
                'review_id': review['review_id'],
                'title': review['title'],
                'score': review['score'],
                'comment_time': review['comment_time'],
                'comment_content': review['comment_content'].replace('\n', '\\n'),  # 将换行符转换为字符串
                'author_name': review['author']['name'],
                'author_url': review['author']['url'],
                'useful_count': review['interaction']['useful_count'],
                'useless_count': review['interaction']['useless_count'],
                'reply_count': review['interaction']['reply_count'],
                'url': review['url']
            }
            flattened_reviews.append(flat_review)
        
        df = pd.DataFrame(flattened_reviews)
        # 删除完全为空的行
        df = df.dropna(how='all')
        # 删除评论内容为空的行
        df = df.dropna(subset=['comment_content'])
        
        # 保存CSV时添加空行分隔
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            df.to_csv(f, index=False, encoding='utf-8-sig')
            f.write('\n')  # 添加额外的空行
        
        print(f"数据已保存到 {output_file}")
        
    except Exception as e:
        print(f"保存CSV文件时出错: {str(e)}")

def save_to_json(reviews, output_file):
    """保存评论数据到JSON文件"""
    try:
        # 在JSON中保持换行符
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, ensure_ascii=False, indent=4)
            f.write('\n\n')  # 添加额外的空行
        print(f"数据已保存到 {output_file}")
    except Exception as e:
        print(f"保存JSON文件时出错: {str(e)}")

def main():
    movie_url = 'https://movie.douban.com/subject/34780991/'
    output_file = 'naza2_reviews.csv'
    
    reviews_url = get_reviews_url(movie_url)
    if not reviews_url:
        print("无法获取影评页面URL")
        return
    
    print("开始提取评论...")
    reviews = extract_reviews(reviews_url)
    
    if reviews:
        print(f"共获取到 {len(reviews)} 条评论")
        # 保存最终数据
        save_to_csv(reviews, output_file)
        save_to_json(reviews, output_file.replace('.csv', '.json'))
    else:
        print("未获取到任何评论")

if __name__ == '__main__':
    main()

async def get_page_async(url, session, semaphore, retry=RETRY_TIMES):
    """
    异步获取页面内容
    :param url: 目标URL
    :param session: aiohttp会话对象
    :param semaphore: 信号量对象，用于控制并发
    :param retry: 重试次数
    :return: 页面内容或None
    """
    async with semaphore:  # 使用信号量控制并发
        for i in range(retry):
            try:
                # 添加随机延迟
                await asyncio.sleep(random.uniform(1, 3))
                async with session.get(url, headers=headers, timeout=ClientTimeout(total=REQUEST_TIMEOUT)) as response:
                    text = await response.text()
                    
                    # 检查是否需要登录
                    if '登录' in text and '注册' in text:
                        print("检测到需要登录，请更新cookie")
                        return None
                    
                    return text
            except Exception as e:
                print(f"请求失败，正在重试 ({i+1}/{retry}): {str(e)}")
                await asyncio.sleep(5)
    return None

async def get_reviews_url_async(movie_url, session, semaphore):
    """
    异步获取影评页面URL
    """
    try:
        html = await get_page_async(movie_url, session, semaphore)
        if not html:
            return None
        html = etree.HTML(html)
        reviews_url = html.xpath('//*[@id="reviews-wrapper"]/header/h2/span/a/@href')
        if reviews_url:
            return urljoin(movie_url, reviews_url[0])
        return None
    except Exception as e:
        print(f"获取影评URL失败: {str(e)}")
        return None

async def process_review_item(item, session, semaphore):
    """
    异步处理单个评论项
    """
    try:
        review_id = item.xpath('./@id')[0]
        
        # 获取评论基本信息
        title = item.xpath('.//h2/a/text()')[0].strip()
        title = clean_text(title)
        
        score_elements = item.xpath('.//span[contains(@class, "allstar")]/@title')
        score = score_elements[0] if score_elements else "未评分"
        
        comment_time = item.xpath('.//span[@class="main-meta"]/text()')[0].strip()
        try:
            comment_time = datetime.strptime(comment_time, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
        except:
            comment_time = None
        
        author_name = item.xpath('.//a[@class="name"]/text()')[0].strip()
        author_url = item.xpath('.//a[@class="name"]/@href')[0]
        
        # 获取评论内容
        short_content = item.xpath('.//div[@class="short-content"]/text()')
        short_content = '\n'.join([p.strip() for p in short_content if p.strip()])
        short_content = clean_text(short_content)
        
        # 获取互动数据
        useful_count = item.xpath('.//span[@id="r-useful_count-{}"]/text()'.format(review_id))
        useful_count = int(useful_count[0].strip()) if useful_count and useful_count[0].strip() else 0
        
        useless_count = item.xpath('.//span[@id="r-useless_count-{}"]/text()'.format(review_id))
        useless_count = int(useless_count[0].strip()) if useless_count and useless_count[0].strip() else 0
        
        reply_count = item.xpath('.//a[contains(@class, "reply")]/text()')
        reply_count = int(reply_count[0].strip().replace('回应', '')) if reply_count and reply_count[0].strip() else 0
        
        # 异步获取详细评论内容
        comment_url = item.xpath('.//h2/a/@href')[0]
        comment_html = await get_page_async(comment_url, session, semaphore)
        
        if comment_html:
            comment_tree = etree.HTML(comment_html)
            comment_content = comment_tree.xpath('//div[@class="review-content clearfix"]/p/text()')
            if not comment_content:
                comment_content = [short_content]
            comment_content = '\n'.join([p.strip() for p in comment_content])
            comment_content = clean_text(comment_content)
        else:
            comment_content = short_content
        
        return {
            'review_id': review_id,
            'title': title,
            'score': score,
            'comment_time': comment_time,
            'comment_content': comment_content,
            'author': {
                'name': author_name,
                'url': author_url
            },
            'interaction': {
                'useful_count': useful_count,
                'useless_count': useless_count,
                'reply_count': reply_count
            },
            'url': comment_url
        }
    except Exception as e:
        print(f"处理评论项时出错: {str(e)}")
        return None

async def extract_reviews_async(reviews_url, max_pages=390):
    """
    异步提取所有评论
    """
    all_reviews = []
    page = 0
    error_count = 0
    max_errors = 5
    
    # 创建信号量控制并发
    semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 创建异步会话
    async with aiohttp.ClientSession() as session:
        while page < max_pages and error_count < max_errors:
            try:
                current_url = f"{reviews_url}?start={page*20}" if page > 0 else reviews_url
                print(f"正在处理第 {page + 1}/{max_pages} 页: {current_url}")
                
                html = await get_page_async(current_url, session, semaphore)
                if not html:
                    error_count += 1
                    print(f"获取页面失败，错误计数: {error_count}")
                    await asyncio.sleep(5)
                    continue
                
                html = etree.HTML(html)
                review_items = html.xpath('//div[contains(@class, "review-list")]//div[contains(@class, "review-item")]')
                
                if not review_items:
                    error_count += 1
                    print(f"未找到评论，错误计数: {error_count}")
                    await asyncio.sleep(5)
                    continue
                
                error_count = 0
                print(f"找到 {len(review_items)} 条评论")
                
                # 并发处理所有评论项
                tasks = [process_review_item(item, session, semaphore) for item in review_items]
                page_reviews = await asyncio.gather(*tasks)
                
                # 过滤掉None值并添加到总列表
                valid_reviews = [r for r in page_reviews if r and r['comment_content'] and len(r['comment_content'].strip()) > 0]
                all_reviews.extend(valid_reviews)
                
                # 每处理10页保存一次数据
                if (page + 1) % 10 == 0:
                    await save_to_csv_async(all_reviews, f'naza2_reviews_page_{page+1}.csv')
                    await save_to_json_async(all_reviews, f'naza2_reviews_page_{page+1}.json')
                    print(f"已保存第 {page + 1} 页数据")
                
                page += 1
                await asyncio.sleep(random.uniform(2, 4))
                
            except Exception as e:
                print(f"处理页面时出错: {str(e)}")
                error_count += 1
                await asyncio.sleep(5)
                continue
    
    return all_reviews

async def save_to_csv_async(reviews, output_file):
    """
    异步保存评论数据到CSV文件
    """
    try:
        flattened_reviews = []
        for review in reviews:
            flat_review = {
                'review_id': review['review_id'],
                'title': review['title'],
                'score': review['score'],
                'comment_time': review['comment_time'],
                'comment_content': review['comment_content'].replace('\n', '\\n'),
                'author_name': review['author']['name'],
                'author_url': review['author']['url'],
                'useful_count': review['interaction']['useful_count'],
                'useless_count': review['interaction']['useless_count'],
                'reply_count': review['interaction']['reply_count'],
                'url': review['url']
            }
            flattened_reviews.append(flat_review)
        
        df = pd.DataFrame(flattened_reviews)
        df = df.dropna(how='all')
        df = df.dropna(subset=['comment_content'])
        
        # 使用异步文件操作
        async with aiofiles.open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            await f.write(df.to_csv(index=False, encoding='utf-8-sig'))
            await f.write('\n')
        
        print(f"数据已保存到 {output_file}")
        
    except Exception as e:
        print(f"保存CSV文件时出错: {str(e)}")

async def save_to_json_async(reviews, output_file):
    """
    异步保存评论数据到JSON文件
    """
    try:
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(reviews, ensure_ascii=False, indent=4))
            await f.write('\n\n')
        print(f"数据已保存到 {output_file}")
    except Exception as e:
        print(f"保存JSON文件时出错: {str(e)}")

async def main_async():
    """
    异步主函数
    """
    movie_url = 'https://movie.douban.com/subject/34780991/'
    output_file = 'naza2_reviews.csv'
    
    # 创建信号量
    semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # 创建异步会话
    async with aiohttp.ClientSession() as session:
        reviews_url = await get_reviews_url_async(movie_url, session, semaphore)
        if not reviews_url:
            print("无法获取影评页面URL")
            return
        
        print("开始提取评论...")
        reviews = await extract_reviews_async(reviews_url)
        
        if reviews:
            print(f"共获取到 {len(reviews)} 条评论")
            # 保存最终数据
            await save_to_csv_async(reviews, output_file)
            await save_to_json_async(reviews, output_file.replace('.csv', '.json'))
        else:
            print("未获取到任何评论")

def main():
    """
    主函数入口
    """
    asyncio.run(main_async())

if __name__ == '__main__':
    main()
