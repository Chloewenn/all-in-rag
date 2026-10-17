"""对照 06_embedding_based_routing.py，用 LlamaIndex 实现嵌入相似性路由。

设置 DEEPSEEK_API_KEY 后运行 python 06_llama.py。
路由使用本地嵌入模型，只有生成最终回答时才调用 DeepSeek。
首次运行可能需要下载 BAAI/bge-small-zh-v1.5，无需向量数据库。
"""

import os

from llama_index.core.base.embeddings.base import BaseEmbedding, similarity
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek


# 1. 将原示例的两条处理链封装为 LlamaIndex 查询引擎。
class CuisineQueryEngine(CustomQueryEngine):
    llm: LLM
    instruction: str

    def custom_query(self, query_str: str) -> str:
        response = self.llm.chat(
            [ChatMessage(role="user", content=self.instruction.format(query=query_str))]
        )
        return response.message.content or ""


# 2. 自定义路由引擎：显式计算余弦相似度，不使用 LLM Selector。
class EmbeddingRouterQueryEngine(CustomQueryEngine):
    embed_model: BaseEmbedding
    tools: list[QueryEngineTool]
    route_embeddings: list[list[float]]

    def custom_query(self, query_str: str) -> Response:
        query_embedding = self.embed_model.get_query_embedding(query_str)
        scores = [
            similarity(query_embedding, embedding)
            for embedding in self.route_embeddings
        ]
        chosen_index = max(range(len(scores)), key=scores.__getitem__)
        # 分数是余弦相似度，并非分类概率；与原示例一样，总是选择最高分。
        # 本例只有川菜/粤菜，没有“其他”分支，也没有最低相似度阈值。
        response = self.tools[chosen_index].query_engine.query(query_str)
        response.metadata = response.metadata or {}
        response.metadata.update(route_index=chosen_index, similarity_scores=scores)
        return response


def build_router(llm: LLM, embed_model: BaseEmbedding) -> EmbeddingRouterQueryEngine:
    # description 用于计算路由相似度；instruction 用于生成专家回答。
    tools = [
        QueryEngineTool.from_defaults(
            query_engine=CuisineQueryEngine(
                llm=llm,
                instruction="你是一位川菜大厨。请用正宗的川菜做法，回答关于「{query}」的问题。",
            ),
            name="川菜",
            description="你是一位处理川菜的专家。用户的问题是关于麻辣、辛香、重口味的菜肴，例如水煮鱼、麻婆豆腐、鱼香肉丝、宫保鸡丁、花椒、海椒等。",
        ),
        QueryEngineTool.from_defaults(
            query_engine=CuisineQueryEngine(
                llm=llm,
                instruction="你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{query}」的问题。",
            ),
            name="粤菜",
            description="你是一位处理粤菜的专家。用户的问题是关于清淡、鲜美、原汁原味的菜肴，例如白切鸡、老火靓汤、虾饺、云吞面等。",
        ),
    ]
    # 路由描述只在初始化时编码一次；每次查询只需编码用户问题。
    route_embeddings = embed_model.get_text_embedding_batch(
        [tool.metadata.description for tool in tools]
    )
    return EmbeddingRouterQueryEngine(
        embed_model=embed_model, tools=tools, route_embeddings=route_embeddings
    )


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请先设置 DEEPSEEK_API_KEY 环境变量")

    llm = DeepSeek(model="deepseek-chat", temperature=0, api_key=api_key)
    # 空指令与原 LangChain 示例保持一致，避免自动添加 BGE 查询前缀。
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        query_instruction="",
        text_instruction="",
    )
    router = build_router(llm, embed_model)
    print("已定义 2 个路由: 川菜, 粤菜")
    print("LlamaIndex 嵌入相似性路由引擎创建成功。\n")

    demo_queries = [
        "水煮鱼怎么做才嫩？",
        "如何做一碗清淡的云吞面？",
        "麻婆豆腐的核心调料是什么？",
    ]
    for i, query in enumerate(demo_queries, 1):
        print(f"\n--- 问题 {i}: {query} ---")
        try:
            response = router.query(query)
            metadata = response.metadata
            for tool, score in zip(router.tools, metadata["similarity_scores"]):
                print(f"{tool.metadata.name}相似度: {score:.4f}")
            chosen_tool = router.tools[metadata["route_index"]]
            print(f"路由决策: 检测到问题与“{chosen_tool.metadata.name}”最相似。")
            print(f"回答: {response}")
        except Exception as e:
            print(f"执行错误: {e}")


if __name__ == "__main__":
    main()
