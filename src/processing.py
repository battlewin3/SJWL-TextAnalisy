import re
import os
import sys
from collections import Counter
from itertools import combinations
import pandas as pd
import spacy
import jieba
import networkx as nx
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from functools import lru_cache

# --- Global Initializations ---

try:
    nlp = spacy.load('zh_core_web_sm')
except OSError:
    print("Warning: spaCy Chinese model 'zh_core_web_sm' not found.")
    print("Please run: python -m spacy download zh_core_web_sm")
    nlp = None

@lru_cache(maxsize=None)
def get_default_stopwords():
    """
    Loads the default stopword list from the resources folder.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords_path = os.path.join(base_dir, '..', 'resources', '停用词表.txt')
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    except Exception as e:
        print(f"Error reading default stopwords: {e}")
        return set()

# --- Sentiment Analysis Setup ---

@lru_cache(maxsize=None)
def get_default_sentiment_dict():
    """
    Loads the default sentiment lexicon from the resources folder.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lexicon_path = os.path.join(base_dir, '..', 'resources', '通用中英文六维语义情感词典', 'Semantic_dimension_prediction', 'Chinese', 'feature.csv')
    try:
        df = pd.read_csv(lexicon_path, encoding='utf-8-sig')
        return pd.Series(df.Emotion.values, index=df.word).to_dict()
    except Exception as e:
        print(f"Error reading default sentiment lexicon: {e}")
        return {}

def load_custom_sentiment_dict(file_path):
    """
    Loads a user-provided custom sentiment dictionary.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        if 'word' in df.columns and 'Emotion' in df.columns:
            return pd.Series(df.Emotion.values, index=df.word).to_dict()
        else:
            print("Error: Custom sentiment dictionary must have 'word' and 'Emotion' columns.")
            return {}
    except Exception as e:
        print(f"Error reading custom sentiment lexicon: {e}")
        return {}

# --- Core Text Processing Functions ---

def text_cleaner(text: str, user_dict_path: str = None, user_stopwords_path: str = None) -> str:
    """
    Cleans text by removing non-Chinese characters and stopwords.
    """
    if user_dict_path:
        try: jieba.load_userdict(user_dict_path)
        except Exception as e: print(f"Error loading user dictionary: {e}")

    stopwords = get_default_stopwords().copy()
    if user_stopwords_path:
        try:
            with open(user_stopwords_path, 'r', encoding='utf-8') as f:
                stopwords.update({line.strip() for line in f})
        except Exception as e: print(f"Error loading user stopwords: {e}")
            
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    tokens = jieba.lcut(text)
    tokens = [t for t in tokens if t.strip() and t not in stopwords and len(t.strip()) > 1]
    return " ".join(tokens)

def analyze_sentiment(cleaned_text: str, custom_sentiment_path: str = None):
    """
    Performs sentiment analysis using a combination of default and custom lexicons.
    """
    sentiment_dict = get_default_sentiment_dict().copy()
    if not sentiment_dict:
        return None, "默认情感词典未加载，无法进行分析。"

    if custom_sentiment_path:
        custom_dict = load_custom_sentiment_dict(custom_sentiment_path)
        sentiment_dict.update(custom_dict)

    words = cleaned_text.split()
    if not words:
        return None, "文本内容为空或被过滤，无法进行情感分析。"

    total_score, sentiment_words = 0, []
    for word in words:
        score = sentiment_dict.get(word)
        if score is not None:
            total_score += score
            sentiment_words.append({'词语': word, '情感分数': score})
    
    avg_score = total_score / len(sentiment_words) if sentiment_words else 0

    if avg_score > 0: label = "正面"
    elif avg_score < 0: label = "负面"
    else: label = "中性"

    pos_words = sorted([d for d in sentiment_words if d['情感分数'] > 0], key=lambda x: x['情感分数'], reverse=True)
    neg_words = sorted([d for d in sentiment_words if d['情感分数'] < 0], key=lambda x: x['情感分数'])

    return {
        "overall_score": avg_score, "label": label,
        "pos_words": pos_words, "neg_words": neg_words,
        "word_count": len(sentiment_words)
    }, None

def calculate_word_frequency(cleaned_text: str) -> list:
    words = cleaned_text.split()
    return [] if not words else list(Counter(words).items())

def calculate_tfidf(cleaned_text: str) -> pd.DataFrame:
    words = cleaned_text.split()
    if not words: return pd.DataFrame(columns=['词语', 'TF-IDF值'])
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    df = pd.DataFrame({
        '词语': vectorizer.get_feature_names_out(),
        'TF-IDF值': tfidf_matrix.toarray().flatten()
    })
    return df.sort_values(by='TF-IDF值', ascending=False)

def perform_lda(cleaned_text: str, num_topics: int = 5, top_words: int = 10) -> (dict, str):
    words = cleaned_text.split()
    if len(words) < num_topics: return None, "文本过短，无法进行主题建模。"
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform([cleaned_text])
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = {f"主题 {i+1}": [feature_names[j] for j in topic.argsort()[:-top_words-1:-1]] 
              for i, topic in enumerate(lda.components_)}
    return topics, None

def perform_ner(raw_text: str) -> (list, str):
    if not nlp: return None, "spaCy模型未加载，无法执行NER。"
    doc = nlp(raw_text)
    entities = [{"实体": e.text, "类型": e.label_, "起始位置": e.start_char, "结束位置": e.end_char}
                for e in doc.ents if len(e.text.strip()) > 1]
    return entities, None

def create_cooccurrence_network(raw_text: str) -> (nx.Graph, str):
    if not nlp: return None, "spaCy模型未加载，无法执行共现分析。"
    doc = nlp(raw_text)
    G = nx.Graph()
    for sent in doc.sents:
        entities = [e.text for e in sent.ents if len(e.text.strip()) > 1]
        for pair in combinations(set(entities), 2):
            u, v = sorted(pair)
            G.add_edge(u, v, weight=G.get_edge_data(u, v, {'weight': 0})['weight'] + 1)
    return (G, None) if G.nodes() else (None, "未找到足够的实体来构建网络图。")

def _find_font() -> str:
    font_paths = {'nt': ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']}
    for font_path in font_paths.get(os.name, []):
        if os.path.exists(font_path): return font_path
    return None

def generate_wordcloud(cleaned_text: str) -> (WordCloud, str):
    font_path = _find_font()
    if not font_path: return None, "生成词云失败: 未在系统中找到支持中文的字体。"
    try:
        wc = WordCloud(width=800, height=400, background_color='white',
                       font_path=font_path, collocations=False).generate(cleaned_text)
        return wc, None
    except Exception as e:
        return None, f"生成词云失败: {e}."
