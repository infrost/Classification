#!/usr/bin/env python
# encoding=utf-8
import sys
import os.path
from datetime import datetime
import json
import hanlp

# 初始化腾讯NLU引擎
module_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(module_dir + '/lib/')
print('初始化腾讯NLU引擎...')
from tencent_ai_texsmart import *
engine = NluEngine(module_dir + '/data/nlu/kb/', 1)

# 初始化 hanlp 的语义相似度分析
print('初始化hanlp语义相似度模块分析')
HanLP = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)

# 加载配置文件中的分类定义并进行预处理
def load_and_preprocess_categories(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    preprocessed_categories = {}
    category_entities = {}
    category_related = {}
    entity_to_category = {}

    for category, chars in categories.items():
        preprocessed_chars = chars  # 不进行去重操作，直接赋值

        # 对分类字符进行实体识别
        output = engine.parse_text(preprocessed_chars)
        entities = [entity.type.name for entity in output.entities()]

        # 提取related字段并将其内容合并到分类字符中
        related_content = []
        for entity in output.entities():
            try:
                meaning_json = json.loads(entity.meaning)
                if 'related' in meaning_json:
                    related_content.extend(meaning_json['related'])
            except json.JSONDecodeError:
                continue

        preprocessed_chars += ''.join(related_content)
        preprocessed_chars = ''.join(dict.fromkeys(preprocessed_chars))  # 再次删除重复字符
        preprocessed_categories[category] = preprocessed_chars
        category_entities[category] = entities
        category_related[category] = related_content

        # 建立实体到分类的映射表
        for entity in entities:
            if entity not in entity_to_category:
                entity_to_category[entity] = []
            entity_to_category[entity].append(category)

    return preprocessed_categories, category_entities, category_related, entity_to_category

# 从配置文件中加载并预处理分类
categories, category_entities, category_related, entity_to_category = load_and_preprocess_categories('config.json')

def text_classification(word):
    category_counts = {category: 0 for category in categories}

    # 统计每个分类中包含的字符个数
    for char in word:
        for category, chars in categories.items():
            if char in chars:
                category_counts[category] += 1

    # 找出字符个数最多的分类
    max_category = 'other'
    max_count = 0
    for category, count in category_counts.items():
        if count > max_count:
            max_count = count
            max_category = category

    return max_category

def load_processed_words(output_file):
    processed_words = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as outfile:
            for line in outfile:
                processed_word = line.split(':')[0].strip()
                processed_words.add(processed_word)
    except FileNotFoundError:
        pass  # 如果文件不存在，则认为没有已经处理过的单词
    return processed_words

def log_preprocessing_results(categories, category_entities, category_related, log_file):
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"预处理时间: {datetime.now()}\n")
        log.write("预处理后的分类结果:\n")
        for category, chars in categories.items():
            log.write(f"{category}: {chars}\n")
        log.write("分类的实体识别结果:\n")
        for category, entities in category_entities.items():
            log.write(f"{category}: {', '.join(entities)}\n")
        log.write("分类的相关内容:\n")
        for category, related in category_related.items():
            log.write(f"{category}: {', '.join(related)}\n")
        log.write("\n")

def log_classification_results(category_counts, suspect_counts, log_file):
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"分类处理时间: {datetime.now()}\n")
        log.write("分类结果统计:\n")
        for category, count in category_counts.items():
            log.write(f"{category}: {count} 个词条\n")
        log.write("疑似分类统计:\n")
        for category, count in suspect_counts.items():
            log.write(f"疑似 {category}: {count} 个词条\n")
        log.write("\n")

def classify_word(word, log_file, output_file, category_counts, suspect_counts):
    with open(output_file, 'a', encoding='utf-8') as outfile, open(log_file, 'a', encoding='utf-8') as log:
        try:
            log.write(f"处理词条: {word}\n")
            result = None

            if len(word) <= 20:
                # 使用腾讯AI的实体识别引擎进行命名实体识别（NER）
                output = engine.parse_text(word)
                entities = [entity.type.name for entity in output.entities()]
                log.write(f"识别到的实体: {', '.join(entities)}\n")
            else:
                entities = []

            if entities:
                # 匹配到实体，分类处理
                entity_matches = {}
                for entity in entities:
                    if entity in entity_to_category:
                        for cat in entity_to_category[entity]:
                            if cat not in entity_matches:
                                entity_matches[cat] = 0
                            entity_matches[cat] += 1
                if entity_matches:
                    # 找到包含该实体数量最少的分类
                    best_category = min(entity_matches, key=entity_matches.get)
                    result = f"{best_category}【实体识别】"
                    category_counts[best_category] += 1
                    
            if not result:
                # 基于语义的相似性检测
                similarities = []
                for cat, cat_content in categories.items():
                    similarity = HanLP([(word, cat_content)])[0]
                    similarities.append((cat, similarity))
                # 找出最高相似度的分类
                most_similar_category, confidence = max(similarities, key=lambda x: x[1])
                if abs(1-confidence) > 0.80:  # 使用一个很小的阈值来比较浮点数
                    result = f"other无法判断"
                else:
                    result = f"other疑似【{most_similar_category}, {confidence:.2f}】"
                    suspect_counts[most_similar_category] += 1
            
            if result == "other无法判断":
                category = text_classification(word)
                if category != 'other':
                    result = f"{category}【字符匹配】"
                    category_counts[category] += 1

            print(f"成功处理：{word}: {result}")  # 打印成功结果
            outfile.write(f"{word}: {result}\n")
            outfile.flush()  # 确保内容立刻写入文件
        except Exception as e:
            print(f"请求错误：{e}. 正在重试...")  # 打印错误信息
            log.write(f"处理词条 {word} 时出现错误：{e}\n")  # 记录错误信息

def classify_words(input_file, output_file, log_file):
    processed_words = load_processed_words(output_file)
    category_counts = {category: 0 for category in categories}
    suspect_counts = {category: 0 for category in categories}

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            word = line.strip()
            if word and word not in processed_words:  # 忽略空行和已经处理过的单词
                classify_word(word, log_file, output_file, category_counts, suspect_counts)

    log_classification_results(category_counts, suspect_counts, log_file)
    return category_counts, suspect_counts

# 日志文件路径
log_file = 'log.txt'

# 记录预处理结果
log_preprocessing_results(categories, category_entities, category_related, log_file)

# 调用函数处理keywords.txt并输出到analysis.txt，同时记录日志
category_counts, suspect_counts = classify_words('keywords.txt', 'analysis.txt', log_file)

# 提示用户继续输入词条进行分类
print("处理完毕，可以继续输入词条进行分类。输入exit结束程序。")
while True:
    user_input = input("请输入词条：")
    if user_input.lower() == 'exit':
        break
    classify_word(user_input.strip(), log_file, 'analysis.txt', category_counts, suspect_counts)
print("程序已结束。")
