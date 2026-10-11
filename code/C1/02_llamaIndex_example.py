import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
# Settings：提供默认聊天模型、嵌入模型、分块器等配置，供后面的组件使用。
from llama_index.core.node_parser import TokenTextSplitter  # 按 token 长度控制文本分块。
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 加载配置。
load_dotenv()

# 2. 配置负责生成回答的 LLM（Large Language Model，大语言模型）。使用 AIHubmix
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 3. 配置嵌入模型 与分块器。
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

Settings.node_parser = TokenTextSplitter(chunk_size=1024, chunk_overlap=200)
# chunk_size=1024 控制每块的 token 预算；metadata 也会占用预算，正文可能更短。
# chunk_overlap=200 设置相邻块的目标重叠长度，以减少分块边界处的上下文丢失。
# 本例的默认 tokenizer 使用 tiktoken；它用于估算长度，不是聊天模型本身。
# 这与前例按 len() 计算字符数的 RecursiveCharacterTextSplitter 有所不同。

# 4. 配置读取器
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()
# Document 包含正文与 metadata（文件路径等附加信息）；本例尚未在这里分块。
# LlamaIndex 可用 doc.text 或 doc.get_content() 获取正文，不是 LangChain 的 page_content。

# 5. 建立向量索引
index = VectorStoreIndex.from_documents(docs)

# 6. 建立查询引擎
query_engine = index.as_query_engine()
# 当前版本默认 similarity_top_k=2，即最多取相似度最高的 2 个节点，不是 2 个例子。

# 7. 查看框架内置的提示词模板。
print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))