# %%
import pandas as pd

tfidf_path = 'tfidf_search_results_normalized.csv'
semantic_path = 'semantic_search_results.csv'

tfidf_csv = pd.read_csv(tfidf_path)
semantic_csv = pd.read_csv(semantic_path)

df = pd.DataFrame({
    '搜尋詞': tfidf_csv['搜尋詞'],
    'Rank': tfidf_csv['Rank'],
    'tf-idf': tfidf_csv['tf-idf'],
    'ner_relevancy_1': tfidf_csv['ner_relevancy_1'],
    'semantic_model': semantic_csv['semantic_model'],
    'ner_relevancy_2': semantic_csv['ner_relevancy_2']
})

df

# %%
from src import inference_api
import torch, src.config as config

# all_attribute = ['品牌', '名稱', '產品', '產品序號', '顏色', '材質', '對象與族群', '適用物體、事件與場所', 
#                      '特殊主題', '形狀', '圖案', '尺寸', '重量', '容量', '包裝組合', '功能與規格']

# set device
config.string_device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.device = torch.device(config.string_device)

# load model
model, tokenizer = inference_api.load_model("clw8998/Product-Name-NER-model", device=config.device)

# Relevancy Calculation Function: Handles the comparison and calculation of relevancy scores
def ner_relevancy(df, index, all_results, check_att, margin, target_weights):
    print(df.loc[index, '搜尋詞'])

    if df.loc[index, '搜尋詞'].lower() not in all_results:
        print(f"Warning(query): '{df.loc[index, '搜尋詞']}' not found in NER results.")
        return df
    if df.loc[index, 'tf-idf'].lower() not in all_results:
        print(f"Warning(tf-idf): '{df.loc[index, 'tf-idf']}' not found in NER results.")
        return df
    if df.loc[index, 'semantic_model'].lower() not in all_results:
        print(f"Warning(semantic_model): '{df.loc[index, 'semantic_model']}' not found in NER results.")
        return df

    query_tags_dict = all_results[df.loc[index, '搜尋詞'].lower()]
    tfidf_tags_dict = all_results[df.loc[index, 'tf-idf'].lower()]
    semantic_tags_dict = all_results[df.loc[index, 'semantic_model'].lower()]

    # 計算 query tags 與 tf-idf 的 tags 相關性
    tfidf_relevancy_score = calculate_weighted_relevancy(query_tags_dict, tfidf_tags_dict, check_att, margin, target_weights)
    df.loc[index, 'ner_relevancy_1'] = 2 if tfidf_relevancy_score >= 0.7 else (1 if tfidf_relevancy_score >= 0.35 else 0)

    # 計算 query tags 與 semantic model 的 tags 相關性
    semantic_relevancy_score = calculate_weighted_relevancy(query_tags_dict, semantic_tags_dict, check_att, margin, target_weights)
    df.loc[index, 'ner_relevancy_2'] = 2 if semantic_relevancy_score >= 0.7 else (1 if semantic_relevancy_score >= 0.35 else 0)
    
    return df

# %%
def flatten_tags(tag_dict, att):
    return set(ent for ents in tag_dict.get(att, []) for ent in ents)

def calculate_weighted_relevancy(query_pool, target_pool, check_att, margin, tag_weights):
    # Use tag weights if provided, otherwise default to equal weighting
    tag_weights = tag_weights or {att: 1 for att in check_att}

    relevancy_score = 0
    total_weight = sum(tag_weights[att] for att in check_att)  # Total weight for normalizing

    # Loop over the required attribute categories
    for att in check_att:

        query_tag_set = flatten_tags(query_pool, att)  # Flattened query tags
        target_tag_set = flatten_tags(target_pool, att)   # Flattened target tags

        # Calculate intersection based on set size
        intersection_size = len(query_tag_set & target_tag_set)

        # Calculate relevancy for current attribute based on margin and weights
        if intersection_size >= len(query_tag_set) * margin:
            relevancy_score += tag_weights[att]  # Fully match
        elif intersection_size > 0:
            relevancy_score += tag_weights[att] * 0.5  # Partial match

    return relevancy_score / total_weight  # Normalize score

# %%
check_att = ['品牌', '產品', '顏色', '適用物體、事件與場所', '功能與規格']
all_results = inference_api.get_ner_tags(
        model, 
        tokenizer, 
        list(set(df['搜尋詞'].tolist() + df['tf-idf'].tolist() + df['semantic_model'].tolist())), 
        check_att)

# %%
tag_weights = {'品牌': 1.5, '產品': 1.5, '顏色': 0.5, '適用物體、事件與場所': 1.0, '功能與規格': 1.5}

for index, row in df.iterrows():
    df = ner_relevancy(df, index, all_results, check_att, margin=0.3, target_weights = tag_weights)
df

# %%
df.to_csv('NER-relevancy_modify.csv', index=False, encoding='utf-8-sig')