"""用 LlamaIndex 实现与 05_llm_based_routing.py 对应的 LLM 意图路由。

运行前设置 DEEPSEEK_API_KEY，然后执行 python 05_llama.py。
依赖项目 requirements.txt 中的 llama-index-core、llama-index-llms-deepseek。
本例仅路由到不同提示模板，不需要向量模型或向量数据库。
"""

import os

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.query_engine import CustomQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.deepseek import DeepSeek


# 1. 定义不同菜系的处理引擎，对应 LangChain 中的 prompt | llm | parser。
class CuisineQueryEngine(CustomQueryEngine):
    """将某个专家角色的提示模板封装为可被路由的查询引擎。"""

    llm: LLM
    instruction: str

    def custom_query(self, query_str: str) -> str:
        response = self.llm.chat(
            [ChatMessage(role="user", content=self.instruction.format(question=query_str))]
        )
        return response.message.content or ""


def build_router(llm: LLM) -> tuple[RouterQueryEngine, list[QueryEngineTool]]:
    # 工具描述会交给分类用的 LLM，而 instruction 用于选中后的专家回答。
    tools = [
        QueryEngineTool.from_defaults(
            query_engine=CuisineQueryEngine(
                llm=llm,
                instruction="你是一位川菜大厨。请用正宗的川菜做法，回答关于「{question}」的问题。",
            ),
            name="sichuan",
            description="川菜：回答川菜菜品和烹饪方法，例如麻婆豆腐、回锅肉、水煮鱼。",
        ),
        QueryEngineTool.from_defaults(
            query_engine=CuisineQueryEngine(
                llm=llm,
                instruction="你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{question}」的问题。",
            ),
            name="cantonese",
            description="粤菜：回答粤菜菜品和烹饪方法，例如白切鸡、清蒸鱼、叉烧。",
        ),
        QueryEngineTool.from_defaults(
            query_engine=CuisineQueryEngine(
                llm=llm,
                instruction="你是一个美食助手。请回答关于「{question}」的问题。",
            ),
            name="general",
            description="其他：回答不属于川菜或粤菜、菜系不明确的通用美食问题，例如番茄炒蛋是否放糖。",
        ),
    ]

    # 2. Selector 对应分类器；RouterQueryEngine 对应分类并分发的完整链。
    # LLMSingleSelector 通过文本/JSON 选择一个工具，不要求模型支持函数调用。
    # general 也是由 LLM 选择的候选项，并非分类失败时自动执行的异常兜底。
    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
        llm=llm,
    )
    return router, tools


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请先设置 DEEPSEEK_API_KEY 环境变量")

    llm = DeepSeek(model="deepseek-chat", temperature=0, api_key=api_key)
    router, tools = build_router(llm)
    print("LlamaIndex 路由查询引擎创建成功。\n")

    # 3. 保留 LangChain 示例中的三个问题，方便比较。
    demo_questions = [
        "麻婆豆腐怎么做？",          # 预期：川菜
        "白切鸡的正宗做法是什么？",  # 预期：粤菜
        "番茄炒蛋需要放糖吗？",      # 预期：其他
    ]
    topics = {"sichuan": "川菜", "cantonese": "粤菜", "general": "其他"}
    for i, question in enumerate(demo_questions, 1):
        print(f"\n--- 问题 {i}: {question} ---")
        try:
            # 只执行一次完整路由；从响应元数据读取真实执行的决策，避免重复分类。
            response = router.query(question)
            selection = response.metadata["selector_result"]
            tool_name = tools[selection.ind].metadata.name
            print(f"路由决策: {topics[tool_name]} ({tool_name})")
            print(f"选择理由: {selection.reason}")
            print(f"回答: {response}")
        except Exception as e:
            print(f"执行错误: {e}")


if __name__ == "__main__":
    main()
