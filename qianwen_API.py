from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [
    {"role": "system", "content": "你是一个专升本题库同义改写模型，擅长生成语义相似但表达不同的高等数学、英语、语文、管理学题目。"},
    {"role": "user", "content": "请改写以下题目：以下函数是偶函数的是(\quad). A y=2\sin x B y=cos2x C y={{x}^{3}}sinx D y=|sinx|cosx"}
 ]

reply = client.chat.completions.create(
    model="qwen-flash",  # 可以按需更换为其它深度思考模型
    messages=messages,
    # extra_body={"enable_thinking": True},
    # stream=True
)
print(reply.choices[0].message.content)