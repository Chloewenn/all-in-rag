from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 使用与 03_llamaindex_vector.py 建立索引时相同的嵌入模型。
# 文档向量已保存在磁盘中；这里的模型用于将查询文字转换为向量。
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 从脚本所在目录定位索引文件夹，避免受终端工作目录影响。
persist_path = Path(__file__).resolve().parent / "llamaindex_index_store"

# StorageContext 加载文档存储、向量存储、索引存储等组件。
# load_index_from_storage 根据保存的结构恢复索引，不重新分块或计算文档向量。
storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
index = load_index_from_storage(storage_context)
print(f"LlamaIndex 索引已加载：{persist_path}")

# 3. 创建检索器并搜索。similarity_top_k=1 表示最多返回 1 个相似节点。
# 检索器只进行相似性搜索，这个示例不需要聊天模型或 API Key。
retriever = index.as_retriever(similarity_top_k=1)
query = "LlamaIndex是做什么的？"
results = retriever.retrieve(query)

# 4. 每个结果是 NodeWithScore：node 保存原文，score 是相似度分数。
# 在当前默认向量存储中使用余弦相似度；分数不是答案正确的概率。
print(f"\n查询：{query}")
print("相似度最高的文本：")
for result in results:
    print(f"- {result.node.get_content()}")
    if result.score is not None:
        print(f"  相似度：{result.score:.4f}")
