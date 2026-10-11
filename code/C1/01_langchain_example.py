# 学习目标：用一篇本地 Markdown 文档，搭建一个最小的 RAG 问答流程。
# RAG（Retrieval-Augmented Generation，检索增强生成）先找资料，再让模型根据资料回答。
# 本例不会训练或微调大模型，而是在每次提问时把检索到的资料放进提示词。
#
# 可以先沿着这条数据流阅读，再回头看每一步的细节：
# 文件 -> docs（文档）-> chunks（分块）-> vectorstore（向量存储）
# question（问题）-> retrieved_docs（相关分块）-> docs_content（参考资料）
# 问题 + 参考资料 -> prompt（提示词）-> llm（大语言模型）-> answer（回答）
#
# Python 小提示：TextLoader、ChatOpenAI 等是类，相当于一类工具的定义；
# TextLoader(...) 会创建一个对象，loader 是指向这个对象的变量；
# loader.load() 调用对象的方法，让它执行“读取文档”这个具体动作。

import os  # 用于访问环境变量：os.getenv(...) 读取配置，os.environ[...] 设置配置。
from pathlib import Path  # 用对象表示文件路径，方便拼接路径和定位父目录。
# Hugging Face 是模型托管平台，下面的设置用于选择模型下载地址。
# 若需要使用镜像，取消下一行注释；应放在导入模型相关库之前。
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv  # 将 .env 文件中的配置加载到当前进程的环境变量中。
from langchain_community.document_loaders import TextLoader  # 把文本文件转换为 Document。
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 将长文档切成小块。
from langchain_huggingface import HuggingFaceEmbeddings  # 用本地嵌入模型把文本转换为向量。
from langchain_core.vectorstores import InMemoryVectorStore  # 在内存中保存向量并提供检索。
from langchain_core.prompts import ChatPromptTemplate  # 定义可填入问题和资料的提示词模板。
from langchain_openai import ChatOpenAI  # 封装聊天模型 API 的配置和调用。

# 1. 加载配置并确定知识源的位置。
# load_dotenv() 默认从脚本所在目录开始向上查找 .env，通常不会覆盖已有环境变量。
# 本例的 .env 可填写 AIHUBMIX_API_KEY=你的密钥，供后面的 os.getenv(...) 读取。
load_dotenv()

# __file__ 是当前脚本的路径，resolve() 将它转换为解析后的绝对路径。
# parents[0] 是 code/C1，parents[1] 是 code，parents[2] 是项目根目录。
# Path 对象上的 / 表示拼接路径；因此从不同工作目录启动也能定位到同一篇文档。
markdown_path = Path(__file__).resolve().parents[2] / "data/C1/markdown/easy-rl-chapter1.md"

# 2. 加载文档：文件 -> list[Document]。
# TextLoader 将 Markdown 当作普通文本读取，不解析标题、表格等 Markdown 结构。
# 这里用于检索的是文字内容，因此普通文本加载即可，也不需要 NLTK 分词资源。
# str(markdown_path) 把 Path 转成字符串；encoding="utf-8" 指定读取中文的编码。
# 创建 loader 只配置加载器，真正读取文件发生在下一行的 load()。
loader = TextLoader(str(markdown_path), encoding="utf-8")
# load() 返回 Document 对象的列表；本例整个文件先成为一个 Document。
# Document 是 LangChain 传递文档的常用数据结构，主要关注两个字段：
#   page_content：正文字符串（“page” 不代表这里一定是一页纸）。
#   metadata：附加信息；本例包含 source，即原始文件路径。
docs = loader.load()

# 3. 文本分块：list[Document] -> 更多、更短的 Document。
# 整篇文章可能很长；切成小块后，可以只检索与问题相关的部分，减少提示词长度。
# RecursiveCharacterTextSplitter 优先尝试按段落分隔（\n\n），再按换行（\n）、
# 空格，最后按字符切分；只有某部分仍然过长时，才继续用更细的分隔方式处理。
# 当前安装版本的默认参数为 chunk_size=4000、chunk_overlap=200，长度按 len 计算。
# chunk_size 是每块的长度上限，chunk_overlap 是相邻块的目标重叠长度：
# 保留部分重复文字可减少边界处的上下文丢失，实际重叠量受分隔位置影响。
# 这里按字符长度计算，不是按 token 数；token 是模型处理文本时使用的单位。
# 想实验不同粒度，可改成 RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)。
text_splitter = RecursiveCharacterTextSplitter()
# split_documents() 读取每个 Document 的正文，分块后保留其 metadata。
# chunks 仍是 list[Document]；与 docs 的区别是每个对象只包含原文的一部分。
chunks = text_splitter.split_documents(docs)

# 4. 创建嵌入模型，并把分块存入向量存储（索引阶段）。
# Embedding（嵌入）把一段文字转换为固定长度的数字列表，即“向量”。
# 同一模型生成的向量可用于比较文本的语义相似度，帮助找到表达相关意思的文字。
# 嵌入模型负责“把文字转成数字”，后面的聊天模型负责“生成回答”，用途不同。
# HuggingFaceEmbeddings 是 LangChain 的封装，内部使用 Sentence Transformers。
# 初始化时会加载指定模型；首次运行通常下载文件，后续通常复用本地缓存。
embeddings = HuggingFaceEmbeddings(
    # Hugging Face 上的模型标识：组织 BAAI / 模型 bge-small-zh-v1.5，面向中文文本。
    model_name="BAAI/bge-small-zh-v1.5",
    # model_kwargs 传给底层模型；device='cpu' 表示在本机 CPU 上运行嵌入计算。
    model_kwargs={'device': 'cpu'},
    # encode_kwargs 控制文本编码；normalize_embeddings=True 将向量归一化为单位长度。
    # 归一化调整向量的数值尺度，不是把原始文字做格式清洗。
    encode_kwargs={'normalize_embeddings': True}
)
  
# InMemoryVectorStore 将向量、正文和 metadata 保存在当前进程的内存中。
# 构造时传入 embeddings，让存储知道如何编码文档和后续的问题。
# 此时尚未添加资料；程序退出后这些内存数据不会自动保存，下次运行会重新构建。
vectorstore = InMemoryVectorStore(embeddings)
# add_documents() 内部调用 embeddings.embed_documents(...)，为每个分块计算向量，
# 再保存向量和原文。返回值是文档 ID 列表，本例不使用这个返回值。
# 到这一步，知识源已准备好；这仍然是索引数据，没有训练或修改聊天模型。
vectorstore.add_documents(chunks)

# 5. 定义提示词模板：决定如何把参考资料和问题交给聊天模型。
# Prompt（提示词）就是发给模型的输入文字，包括任务说明、资料和问题。
# from_template() 根据字符串创建 ChatPromptTemplate；这里创建的是一个 human 消息模板。
# {context} 和 {question} 是占位符，稍后由 format() 填入，不是这里立即求值的 f-string。
# 创建模板不会调用 API；同一个模板可以复用于不同问题和不同检索结果。
# “只根据上下文回答”是在约束模型的行为，但提示词本身不能保证回答一定准确。
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 6. 配置负责生成回答的大语言模型（LLM，Large Language Model）。
# ChatOpenAI 是客户端封装，不是下载到本机的模型；本例通过 AIHubmix 调用远程模型。
# 它使用 OpenAI 风格的接口，实际服务平台由 base_url 决定，模型由 model 决定。
# 构造对象会检查配置并准备客户端；真正发送问题发生在后面的 invoke()。
llm = ChatOpenAI(
    # 服务平台提供的模型标识；它与前面的 BAAI 嵌入模型是两个不同的模型。
    model="glm-4.7-flash-free",
    # 控制生成时的采样随机性：通常越低越稳定，越高越多样；并非“准确率”设置。
    temperature=0.7,
    # 限制生成部分的 token 预算，不是字符数，也不是输入文档长度上限。
    # 对推理模型，这个预算可能包含推理 token；具体计数方式由接口和模型决定。
    max_tokens=4096,
    # getenv() 返回环境变量的字符串值；变量不存在时返回 None。
    # 变量名要与 .env 一致，密钥要属于下面 base_url 对应的服务平台。
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    # API 的基础地址：请求发往 AIHubmix，而不是 Hugging Face 或本机模型。
    base_url="https://aihubmix.com/v1"
)

# 另一种配置示例：改用 DeepSeek 官方接口时，模型名、密钥和地址应一起对应更换。
# 以下代码保持注释状态，不会执行。
# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

# 7. 检索相关资料，再生成回答（查询阶段）。
# 可以修改 question 来实验；这个简单脚本每次运行只回答一个问题，没有多轮对话历史。
question = "文中举了哪些例子？"

# similarity_search() 内部先调用 embeddings.embed_query(question)，将问题转成向量，
# 再与存储中的分块向量计算余弦相似度，返回最相似的最多 k 个 Document。
# k=3 表示最多取 3 个分块，不是 3 个句子或 3 个例子；这里只返回文档，不返回分数。
# 本例按相似度排序取前几块，没有设置“相似度必须高于多少”的筛选阈值。
# 对“文中有哪些例子”这类全文汇总问题，例子可能分散在其他块中，取 3 块可能漏掉内容。
# 增大 k 能提供更多资料，但也会增加输入长度；检索结果决定模型能看到哪些原文。
retrieved_docs = vectorstore.similarity_search(question, k=3)
# doc.page_content for doc in retrieved_docs 是生成器表达式，逐个取出检索分块的正文。
# "\n\n".join(...) 用两个换行连接正文，得到一个字符串，供模板的 {context} 使用。
# 这里仅拼接正文，没有把 Document 的 metadata 一起交给模型。
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 这一行按“先执行内层，再执行外层”的顺序完成两个动作：
#   1. prompt.format(...) 将 question 和 docs_content 填入模板，返回完整的提示词字符串。
#      ChatPromptTemplate.format() 会把消息格式化为带角色标记的文字；这里不是消息列表。
#   2. llm.invoke(...) 把该字符串作为用户输入发给远程模型，并等待完整响应。
# 模型在本例中不会自己读取文件或查询向量存储，参考资料来自上面填入的 context。
# invoke() 返回 AIMessage 对象，而不是单纯的字符串。
answer = llm.invoke(prompt.format(question=question, context=docs_content))
# print(answer) 会显示整个消息对象：content 是回答正文，response_metadata 和
# usage_metadata 等字段包含服务返回的信息及 token 用量，所以终端输出会比较长。
# 只想阅读正文时，可改为 print(answer.content)，正文中的换行会正常显示。
# 想查看检索是否合适，可在调用模型前逐个打印 retrieved_docs 中的 page_content。
print(answer)
