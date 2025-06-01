# -*- coding: utf-8 -*-
import os
from kag.builder import DocumentIndexer
from kag.solver import HybridRetriever
from langchain.agents import AgentExecutor, Tool
from langchain_community.chat_models import ChatOpenAI

# -------------------- 1.KAG服务初始化 --------------------
os.environ["OPENSPG_HOST"] = "http://localhost:8887"  # OpenSPG服务地址
os.environ["BGE_API_KEY"] = "your_api_key"  # 硅基流动API密钥

# -------------------- 2.知识库构建 --------------------
def build_knowledge_base():
    # 初始化文档处理器（网页7）
    indexer = DocumentIndexer(
        kg_host="neo4j://localhost:7687",
        vector_store="milvus",
        embedding_model="BAAI/bge-m3",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 加载气候文档并构建知识图谱（网页2）
    indexer.load_documents([
        "https://weather.gov.sg/climate-climate-of-singapore/",
        "singapore_climate_reports.pdf"
    ])
    
    # 执行知识抽取与对齐（网页6）
    indexer.build(
        enable_ner=True,  # 启用实体识别
        relation_depth=2,  # 关系抽取层级
        semantic_alignment=True  # 语义对齐
    )

# -------------------- 3.工具定义 --------------------
class ClimateTools:
    @classmethod
    def create_tools(cls):
        # 混合检索器（网页3）
        retriever = HybridRetriever(
            vector_top_k=5,
            kg_relation_depth=2,
            numerical_rules=[
                "temperature > 30 → heat_wave_alert",
                "precipitation > 50 → flood_warning"
            ]
        )
        
        return [
            Tool(
                name="Climate_Knowledge",
                func=retriever.query,
                description="新加坡气候知识库，支持多跳推理和数值计算"
            ),
            Tool(
                name="Weather_Alert",
                func=cls.check_weather_alert,
                description="极端天气预警系统"
            )
        ]
    
    @staticmethod
    def check_weather_alert(query: str) -> str:
        """ 调用气象API进行实时校验 """
        # 示例代码，需接入真实API
        return "当前无极端天气预警"

# -------------------- 4.Agent初始化 --------------------
def init_agent():
    llm = ChatOpenAI(
        model="glm-4",
        temperature=0.3,
        max_tokens=1024
    )
    
    return AgentExecutor.from_agent_and_tools(
        agent=HybridRetriever.create_agent(llm=llm),
        tools=ClimateTools.create_tools(),
        verbose=True,
        max_iterations=3  # 限制推理步骤（网页4）
    )

# -------------------- 5.执行示例 --------------------
if __name__ == "__main__":
    # 构建知识库（首次运行需执行）
    # build_knowledge_base()
    
    agent = init_agent()
    response = agent.invoke({
        "input": "新加坡过去五年中，哪年的极端高温天数最多？结合气候知识分析成因"
    })
    
    print(f"最终答案：\n{response['output']}")
    
    # 显示推理路径（网页3）
    if hasattr(agent, 'reasoning_path'):
        print("\n推理路径：")
        for step in agent.reasoning_path:
            print(f"- {step['action']} → {step['result'][:50]}...")
