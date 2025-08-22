import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import chardet
import tempfile
import os
import io
from src.processing import (
    text_cleaner, 
    calculate_word_frequency, 
    calculate_tfidf, 
    perform_lda, 
    perform_ner,
    create_cooccurrence_network,
    generate_wordcloud,
    analyze_sentiment
)

# --- Global Setup ---

def setup_matplotlib_font():
    font_names = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Zen Hei']
    try:
        for font in font_names:
            try:
                matplotlib.font_manager.findfont(font)
                plt.rcParams['font.sans-serif'] = [font]
                break
            except: continue
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        st.warning(f"无法自动设置中文字体，图形中的中文可能无法正确显示。错误: {e}")

setup_matplotlib_font()

# --- Main Application UI ---

def main():
    st.title("文本分析与可视化工具")

    with st.sidebar:
        st.header("⚙️ 设置")
        with st.expander("文件上传", expanded=True):
            uploaded_files = st.file_uploader("上传一个或多个文本文件 (.txt)", type=["txt"], accept_multiple_files=True)
            encoding_option = st.selectbox("文件编码 (单个文件上传时生效)", ('自动检测', 'UTF-8', 'GBK', 'GB18030', 'BIG5'))

        with st.expander("词典配置"):
            user_dict_file = st.file_uploader("上传自定义分词词典", type=["txt"])
            st.caption("格式: `自定义词 权重 词性` (后两项可选), 每行一个。")
            user_stopwords_file = st.file_uploader("上传自定义停止词 (将覆盖默认词表)", type=["txt"])
            st.caption("格式: 每行一个停止词。")
            user_sentiment_file = st.file_uploader("上传自定义情感词典", type=["csv"])
            st.caption("格式: CSV文件, 包含 `word` 和 `Emotion` 两列。例如: `高兴,10`")

        st.header("📊 分析方法")
        analysis_type = st.radio("选择分析", ("情感分析", "词频分析", "TF-IDF关键词提取", "LDA主题建模", "实体识别 (NER)", "实体共现网络", "词云可视化"))

    if uploaded_files:
        all_raw_text, success_count, error_files = process_uploaded_files(uploaded_files, encoding_option)
        
        if error_files: st.warning(f"以下文件解码失败，已被跳过: {', '.join(error_files)}")
        
        if success_count > 0:
            st.success(f"已成功加载并合并 {success_count} 个文件。")
            with st.expander("原始文本预览 (已合并)", expanded=False):
                st.text_area("原始文本", all_raw_text, height=200)

            custom_files = {"dict": user_dict_file, "stopwords": user_stopwords_file, "sentiment": user_sentiment_file}
            paths = handle_custom_files(custom_files)
            
            run_analysis(analysis_type, all_raw_text, paths)
            
            cleanup_temp_files(paths.values())
        else:
            st.error("所有上传的文件都无法解码，请检查文件格式和编码设置。")
    else:
        st.info("请在左侧上传一个或多个 .txt 文件以开始分析。")

# --- Helper Functions ---

def decode_with_fallback(raw_bytes, initial_encoding):
    encodings_to_try = list(dict.fromkeys([initial_encoding, 'utf-8', 'gbk', 'gb18030', 'big5']))
    for encoding in encodings_to_try:
        if not encoding: continue
        try: return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, TypeError): continue
    return None

def process_uploaded_files(files, encoding_option):
    all_text, success_count, error_files = "", 0, []
    is_batch = len(files) > 1
    if is_batch: st.info("检测到多个文件，将为每个文件自动检测并尝试多种编码。")

    for file in files:
        raw_bytes = file.read()
        primary_encoding = get_encoding(raw_bytes, '自动检测' if is_batch else encoding_option)
        decoded_text = decode_with_fallback(raw_bytes, primary_encoding)
        if decoded_text is not None:
            all_text += decoded_text + "\n"
            success_count += 1
        else:
            error_files.append(file.name)
    return all_text, success_count, error_files

def get_encoding(raw_bytes, encoding_option):
    if encoding_option == '自动检测': return chardet.detect(raw_bytes)['encoding']
    return encoding_option

def handle_custom_files(files):
    paths = {"dict": None, "stopwords": None, "sentiment": None}
    for key, file in files.items():
        if file:
            paths[key] = save_temp_file(file)
            st.sidebar.success(f"已加载: {file.name}")
    return paths

def save_temp_file(uploaded_file):
    suffix = ".csv" if uploaded_file.type == "text/csv" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def cleanup_temp_files(paths):
    for path in paths:
        if path and os.path.exists(path): os.remove(path)

def run_analysis(analysis_type, raw_text, paths):
    analyses = {
        "情感分析": run_sentiment_analysis, "词频分析": run_word_frequency_analysis, 
        "TF-IDF关键词提取": run_tfidf_analysis, "LDA主题建模": run_lda_analysis, 
        "实体识别 (NER)": run_ner_analysis, "实体共现网络": run_cooccurrence_network_analysis, 
        "词云可视化": run_wordcloud_analysis
    }
    target_function = analyses.get(analysis_type)
    if not target_function: st.error(f"未知的分析类型: {analysis_type}"); return

    if analysis_type == "情感分析":
        with st.spinner('正在进行文本预处理...'):
            cleaned_text = text_cleaner(raw_text, paths["dict"], paths["stopwords"])
        target_function(cleaned_text, paths["sentiment"])
    elif analysis_type not in ["实体识别 (NER)", "实体共现网络"]:
        with st.spinner('正在进行文本预处理...'):
            cleaned_text = text_cleaner(raw_text, paths["dict"], paths["stopwords"])
        target_function(cleaned_text)
    else:
        target_function(raw_text)

# --- Analysis-Specific UI Functions ---

def run_sentiment_analysis(cleaned_text, custom_sentiment_path):
    st.header("情感分析结果")
    with st.spinner('正在分析情感...'):
        results, error = analyze_sentiment(cleaned_text, custom_sentiment_path)
    
    if error: st.error(f"情感分析失败: {error}"); return
    if not results: st.warning("无法计算情感分数，可能是因为文本过短或不包含有效的情感词。"); return

    st.subheader("总体情感")
    color = "green" if results['label'] == "正面" else "red" if results['label'] == "负面" else "blue"
    st.markdown(f"情感倾向: <font color='{color}'>**{results['label']}**</font>", unsafe_allow_html=True)
    st.metric("平均情感分数", f"{results['overall_score']:.4f}")
    
    if results['label'] == '中性' and results['word_count'] > 0:
        st.info(f"检测到 {results['word_count']} 个情感词，但正面和负面词的情感分数相互抵消，使总体情感倾向为中性。")
    elif results['word_count'] == 0:
        st.warning("未在文本中检测到任何已知的情感词。")

    with st.expander("查看与导出详细结果", expanded=True):
        df_pos = pd.DataFrame(results['pos_words'])
        df_neg = pd.DataFrame(results['neg_words'])
        
        df_all = pd.concat([df_pos, df_neg], ignore_index=True)
        st.download_button("📥 导出为 CSV", df_all.to_csv(index=False).encode('utf-8-sig'), 'sentiment_analysis.csv', 'text/csv')

        st.subheader("正面情感词")
        st.dataframe(df_pos)
        st.subheader("负面情感词")
        st.dataframe(df_neg)

def run_word_frequency_analysis(cleaned_text):
    st.header("词频分析结果")
    with st.spinner('正在计算...'):
        word_freq = calculate_word_frequency(cleaned_text)
        if not word_freq: st.warning("文本内容为空或被完全过滤，无法计算词频。"); return
        df = pd.DataFrame(word_freq, columns=['词语', '频率']).sort_values(by='频率', ascending=False)

    with st.expander("查看与导出", expanded=True):
        st.download_button("📥 导出为 CSV", df.to_csv(index=False).encode('utf-8-sig'), 'word_frequency.csv', 'text/csv')
        top_n = st.slider("显示Top N词数", 5, min(50, len(df)), 20)
        st.dataframe(df.head(top_n))
        st.bar_chart(df.head(top_n).set_index('词语'))

def run_tfidf_analysis(cleaned_text):
    st.header("TF-IDF关键词提取")
    with st.spinner('正在计算...'):
        df = calculate_tfidf(cleaned_text)
        if df.empty: st.warning("文本内容为空或被完全过滤，无法计算TF-IDF。"); return

    with st.expander("查看与导出", expanded=True):
        st.download_button("📥 导出为 CSV", df.to_csv(index=False).encode('utf-8-sig'), 'tfidf_keywords.csv', 'text/csv')
        top_n = st.slider("显示Top N关键词", 5, min(50, len(df)), 20)
        st.dataframe(df.head(top_n))
        st.bar_chart(df.head(top_n).set_index('词语'))

def run_lda_analysis(cleaned_text):
    st.header("LDA主题建模")
    num_topics = st.slider("主题数量", 2, 10, 5)
    top_words = st.slider("每个主题的关键词数", 5, 20, 10)

    with st.spinner('正在提取主题...'):
        topics, error = perform_lda(cleaned_text, num_topics, top_words)

    if error: st.warning(error)
    else:
        df = pd.DataFrame(topics)
        with st.expander("查看与导出", expanded=True):
            st.download_button("📥 导出为 CSV", df.to_csv(index=False).encode('utf-8-sig'), 'lda_topics.csv', 'text/csv')
            st.dataframe(df)

def run_ner_analysis(raw_text):
    st.header("实体识别 (NER)")
    with st.spinner('正在识别实体...'):
        entities, error = perform_ner(raw_text)

    if error: st.error(error)
    elif not entities: st.warning("未能在文本中识别出任何实体。")
    else:
        df = pd.DataFrame(entities)
        with st.expander("查看与导出", expanded=True):
            col1, col2 = st.columns(2)
            col1.download_button("📥 导出为 CSV", df.to_csv(index=False).encode('utf-8-sig'), 'ner_results.csv', 'text/csv')
            col2.download_button("📥 导出为 JSON", df.to_json(orient='records', indent=4, force_ascii=False).encode('utf-8'), 'ner_results.json', 'application/json')
            st.dataframe(df)
            st.bar_chart(df['类型'].value_counts())

def run_cooccurrence_network_analysis(raw_text):
    st.header("实体共现网络")
    with st.spinner('正在构建网络...'):
        G, error = create_cooccurrence_network(raw_text)

    if error: st.warning(error)
    else:
        with st.expander("查看与导出", expanded=True):
            graphml_string = "\n".join(nx.generate_graphml(G))
            st.download_button("📥 导出为 GraphML", graphml_string, "cooccurrence_network.graphml", "application/xml")
            
            fig, ax = plt.subplots(figsize=(12, 10))
            pos = nx.spring_layout(G, k=0.8, iterations=50)
            nx.draw_networkx(G, pos, with_labels=True, node_size=[G.degree(n) * 100 for n in G.nodes()], width=[G[u][v]['weight'] * 0.5 for u, v in G.edges()], node_color='skyblue', edge_color='gray', alpha=0.8, font_size=10)
            st.pyplot(fig)
            
            edges_df = pd.DataFrame([(u, v, d['weight']) for u, v, d in G.edges(data=True)], columns=['实体1', '实体2', '共现次数']).sort_values(by='共现次数', ascending=False)
            st.dataframe(edges_df)

def run_wordcloud_analysis(cleaned_text):
    st.header("词云可视化")
    if not cleaned_text.strip(): st.warning("文本内容为空或被完全过滤，无法生成词云。"); return

    with st.spinner('正在生成词云...'):
        wc, error = generate_wordcloud(cleaned_text)

    if error: st.error(error)
    else:
        with st.expander("查看与导出", expanded=True):
            img_buffer = io.BytesIO()
            wc.to_image().save(img_buffer, format='PNG')
            st.download_button("📥 导出为 PNG", img_buffer.getvalue(), "wordcloud.png", "image/png")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
