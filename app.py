from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

resp1 = llm.invoke(
    "We are building an AI system for processing medical insurance claims."
)
print("---Response 1---")
print(resp1.content)

resp2 = llm.invoke("What are the main risks in this system?")
print("---Response 2---")
print(resp2.content)

"""
Q: Why the second question may fail or behave inconsistently without conversation history.
A:  It behave Inconsistently or to fail to provide specific medical insurance risks because "LLM are stateless" it doesn't have memory.
    'resp2' is sent as isolated string, without a memory object or list of previous messages passed into the prompt. the model has no idea what this system refers to.
    It may give generic risk about softwares or may ask for clarification depending on models we are using, as the context of resp1 was never sent back to next API call.
"""

messages = [
    SystemMessage(
        content="You are a senior AI architect reviewing production systems."
    ),
    HumanMessage(
        content="We are building an AI system for processing medical insurance claims."
    ),
    HumanMessage(content="What are the main risks in this system?"),
]
print("---Response 2---")
response = llm.invoke(messages)
print(response.content)
print("--------------------------------")

"""
Reflection:

1. Why did string-based invocation fail?
A: String based invocation is stateless. The LLM processes each request as cold start/ random question. When we sent resp2 by then llm had forgotten the previous instance  of resp1. 
    It lacks memory buffer which need to give as feedback with variable or function call i.e resp1.

2. Why does message-based invocation work?
A: Since we had provided manually Human Message as a Conversation History message based invocation worked. While querying second question first human message is treated as the context or transript of what happened so far.
    It doesn't have actually remembers rather simply re-reads upon each invocation.

3. What would break in a production AI system if we ignore message history?
A: 
1. User expects a chat/ conversational flow. re-iterrating the context/ entire problem statement in every message is frustrating.
2. User often use it, the previous one, the first one without conversation history, LLM can not resolve these references.
3. If we had set some security layer in system message but do not include in subsequent calls, the llm may behave in unprofessional or unstructured way.
"""
