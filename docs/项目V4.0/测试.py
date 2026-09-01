#react循环实现
import math

def react_agent(user_query, llm, tools, max_steps=10):
    messages = [
        {
            "role": "system",
            "content": """
            你是一个ReAct Agent,
            每一步可以选择：
            1、调用工具：输出TOOL：工具名，参数
            2、执行根据： 输出final：答案
            """
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    for step in range(max_steps):
        response = llm.invoke(messages)

        if response.startwith("FINAL:"):
            return response[len("FINAL:"):]
            # return response[6:]

        tool_name, tool_args = parse_tool_call(response)
        tool = tools[tool_name]
        observation = tool(**tool_args)

        messages.append({
            "role": "assistant",
            "content": response,
        })

        messages.append({
            "role": "tool",
            "content": str(observation)
        })

    return "没有结果"


#余弦相似度实现
def cosine_similarity(a,b):
    dot_product = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

print(cosine_similarity([1,2,3],[1,2,3]))