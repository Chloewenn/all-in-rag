# 学习目标：用 LlamaIndex 从本地 Markdown 文档构建一个最小的 RAG 问答应用。
# RAG（Retrieval-Augmented Generation，检索增强生成）先检索资料，再根据资料生成回答。
# 本例没有训练或微调聊天模型，而是在提问时把检索到的原文交给模型参考。
#
# 先看两条数据流，再按下面的编号逐步学习：
# 索引阶段：文件 -> docs（Document 列表）-> Nodes（文本块）-> 向量 -> index（索引）。
# 查询阶段：问题 -> 问题向量 -> 相关 Nodes -> 提示词 -> 聊天模型 -> 回答。
#
# 对照 01_langchain_example.py：
# 前例显式调用分块器、添加向量、拼接正文、填充模板和调用模型；
# 本例由 from_documents() 封装分块与向量化，由 query() 封装检索与生成。
# 这些步骤仍然存在，只是由框架在内部执行。
#
# Python 小提示：SimpleDirectoryReader 是类，SimpleDirectoryReader(...) 创建对象；
# .load_data() 是对象的方法，调用它才会真正读取文件。
# Settings 是框架提供的共享配置对象；Settings.llm = ... 是给它的属性赋值。

import os  # os.getenv(...) 用于读取环境变量，例如 API Key。
# Hugging Face 是模型托管平台；若需要使用下载镜像，可取消下一行注释。
# 这行应放在导入模型相关库之前，仅用于选择下载地址。
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv  # 把 .env 中的配置加载到当前进程的环境变量。
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# VectorStoreIndex：组织文档分块及其向量，并提供向量检索入口。
# SimpleDirectoryReader：从目录或指定文件中读取 Document 对象。
# Settings：提供默认聊天模型、嵌入模型、分块器等配置，供后面的组件使用。
from llama_index.core.node_parser import TokenTextSplitter  # 按 token 长度控制文本分块。
from llama_index.llms.openai_like import OpenAILike  # 对接 OpenAI 风格的第三方模型 API。
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 封装本地嵌入模型。

# 1. 加载配置。
# 默认从脚本所在目录开始向上查找 .env，通常不会覆盖已经存在的环境变量。
# 本例可以在 .env 中填写 AIHUBMIX_API_KEY=你的密钥，供下面的 getenv() 读取。
load_dotenv()

# 2. 配置负责生成回答的 LLM（Large Language Model，大语言模型）。
# OpenAILike 是 API 客户端封装，不会将聊天模型下载到本机。
# 创建对象主要是配置连接；真正的生成请求在后面的 query() 内部发生。
# Settings.llm 设置当前进程中的共享默认模型；后续组件没有单独传 llm 时会使用它。
# 它不是写入磁盘的配置，也不会在程序退出后自动保留。
Settings.llm = OpenAILike(
    # 服务平台上的模型标识，和后面的 BAAI 嵌入模型不是同一个模型。
    model="glm-4.7-flash-free",
    # getenv() 返回字符串值；未配置时返回 None。变量名要与 .env 中的名称一致。
    # 密钥必须属于下面 api_base 对应的平台，本例使用 AIHubmix 的密钥。
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    # 请求发送到 AIHubmix；API 格式与 OpenAI 兼容不代表实际服务商是 OpenAI。
    api_base="https://aihubmix.com/v1",
    # 告诉框架这是聊天模型，按带角色的消息调用聊天接口。
    # 这是客户端的配置描述，不会改变远端模型的能力，也不会自动保留多轮对话。
    is_chat_model=True
)

# 下面是停用的另一种配置示例。切换平台时，模型名、密钥和地址要一起对应更换。
# 注意：OpenAI 没有在本文件中导入，不能仅取消下面的注释就直接运行。
# 使用兼容接口时也可以沿用上面的 OpenAILike，并修改相应参数。
# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 3. 配置嵌入模型与分块器。
# Embedding（嵌入）把文字转换为固定长度的数字列表，即向量，便于比较语义相似度。
# 聊天模型负责生成答案，嵌入模型负责编码文档和问题，两者用途不同。
# HuggingFaceEmbedding 内部使用 Sentence Transformers 加载 BAAI 的中文嵌入模型。
# 括号中的字符串是 model_name 参数；首次运行通常下载模型，后续使用本地缓存。
# 本例未指定 device，由框架根据硬件选择运行设备；当前版本默认会归一化向量。
# 注意类名是 HuggingFaceEmbedding，前一个 LangChain 示例的类名以 Embeddings 结尾。
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 默认分块器 SentenceSplitter 在当前版本中使用 NLTK 句子处理资源。
# 如果这些资源未安装，自动下载又被代理安全检查拦截，会在建立索引时失败。
# 这里显式使用 TokenTextSplitter，不需要 NLTK 的 stopwords 或 punkt_tab 资源。
# token 是分词器处理文字的单位，不等同于一个汉字、一个英文单词或一个字符。
# chunk_size=1024 控制每块的 token 预算；metadata 也会占用预算，正文可能更短。
# chunk_overlap=200 设置相邻块的目标重叠长度，以减少分块边界处的上下文丢失。
# 它尝试按空格、换行等边界切分，必要时细分到字符，不保证保留完整句子。
# 本例的默认 tokenizer 使用 tiktoken；它用于估算长度，不是聊天模型本身。
# 这与前例按 len() 计算字符数的 RecursiveCharacterTextSplitter 有所不同。
Settings.node_parser = TokenTextSplitter(chunk_size=1024, chunk_overlap=200)

# 4. 加载知识源：文件 -> list[Document]。
# input_files 是文件路径列表，只读取指定文件；读整个目录时可使用 input_dir 参数。
# 相对路径从终端的当前工作目录计算，所以本例应在 code/C1 目录下运行。
# SimpleDirectoryReader(...) 配置读取器，紧接的 load_data() 才执行读取并返回列表。
# 当前版本对这里的 .md 文件按文本读取；其他格式可能由对应的文件读取器处理。
# Document 包含正文与 metadata（文件路径等附加信息）；本例尚未在这里分块。
# LlamaIndex 可用 doc.text 或 doc.get_content() 获取正文，不是 LangChain 的 page_content。
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 5. 建立向量索引：Document -> Node -> 向量与存储。
# from_documents() 是类方法：直接通过类调用，并返回一个 VectorStoreIndex 对象。
# 它内部先使用 Settings.node_parser 将文档切成 Nodes，再用 Settings.embed_model 编码。
# Node 是 LlamaIndex 的检索单元；本例主要是 TextNode，可以理解为带来源信息的文本块。
# Document 代表加载的原始文档，Node 代表供检索使用的块；Node 不代表一整页。
# 节点正文可用 node.text 或 node.get_content() 查看，还可保留 metadata 和来源关系。
# 当前默认配置在内存中保存节点与向量，没有连接独立数据库，也没有训练聊天模型。
# 本例没有调用 index.storage_context.persist(...)，程序结束后不会自动保存索引。
index = VectorStoreIndex.from_documents(docs)

# 6. 创建查询引擎，组合“检索器”和“回答生成器”。
# as_query_engine() 返回可调用 query() 的对象，此时还没有提交问题。
# 检索器负责寻找相关 Nodes；回答生成器负责组织上下文、填充提示词并调用 Settings.llm。
# 当前版本默认 similarity_top_k=2，即最多取相似度最高的 2 个节点，不是 2 个例子。
# 想提供更多资料，可尝试 index.as_query_engine(similarity_top_k=3)，但会增加输入长度。
# 默认 response_mode='compact' 尝试合并资料；资料太长时可能分多次调用模型并完善回答。
# 因此一次 query() 不一定只对应一次远程 API 请求。
query_engine = index.as_query_engine()

# 7. 查看框架内置的提示词模板。
# get_prompts() 返回“模板名称 -> 模板对象”的字典，这一步不会发起模型请求。
# 常见键为 response_synthesizer:text_qa_template 和 response_synthesizer:refine_template：
# 前者指导模型根据资料回答，后者用于在追加资料时完善已有回答。
# 打印的是模板配置，不是这次提问最终发送的全部消息，里面的占位符还没有填入。
# 与前例手写 ChatPromptTemplate 不同，本例没有显式定义模板，而是使用框架默认模板。
print(query_engine.get_prompts())

# 8. 提问并打印回答。
# query() 内部执行：问题向量化 -> 检索相关 Nodes -> 组织提示词 -> 调用 LLM -> 返回结果。
# 本例的模型不会自己读取本地文件，参考资料由查询引擎从索引中检索后提供。
# “文中有哪些例子”是全文汇总问题；默认只取 2 个节点，可能漏掉其他块中的例子。
# 输出也受提示词和模型能力影响，仍需要对照原文核实。
# 默认的非流式查询返回 Response 对象，print() 会调用其字符串表示，显示回答正文。
# 这与前例 print(AIMessage) 时同时打印元数据的行为不同。
# 如需研究检索结果，可改为 response = query_engine.query(...)，再查看：
#   response.response：回答正文。
#   response.source_nodes：生成回答时使用的来源节点及其检索分数。
#   response.source_nodes[0].node.get_content()：第一个来源节点的内容（结果非空时）。
# 查询引擎不是聊天历史管理器，本脚本每次运行只提交下面这一个问题。
print(query_engine.query("文中举了哪些例子?"))
