import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gradio as gr

# 简单的知识库
knowledge_base = [
    "感冒是一种常见的呼吸道病毒感染。",
    "感冒症状包括流鼻涕、咳嗽、喉咙痛和打喷嚏。",
    "感冒通常持续7-10天。",
    "治疗感冒的方法包括休息、多喝水和服用非处方药如布洛芬。",
    "如果症状持续超过两周，应该咨询医生。",
    "高血压是指血液对血管壁的压力持续过高。",
    "正常血压范围是收缩压低于120mmHg，舒张压低于80mmHg。",
    "高血压患者应该减少盐的摄入，定期服用降压药物。"
]

# 加载嵌入模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 为知识库生成嵌入向量
knowledge_embeddings = model.encode(knowledge_base)

def ask_question(question):
    # 将问题转换为嵌入向量
    question_embedding = model.encode([question])
    
    # 计算与知识库中每个条目的相似度
    similarities = cosine_similarity(question_embedding, knowledge_embeddings)
    
    # 找到最相似的条目
    most_similar_idx = np.argmax(similarities)
    answer = knowledge_base[most_similar_idx]
    
    return answer

# 创建界面
iface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(label="请输入您的问题"),
    outputs=gr.Textbox(label="回答"),
    title="简化版医学问答助手",
    description="这是一个基于语义相似度的简单医学问答演示"
)

# 启动应用
iface.launch()