---
title: "Building an Agentic Workflow with AWS Bedrock, LangChain, and LangGraph"
date: 2025-11-22T08:00:30-04:00
categories:
  - Generative AI
  - AWS
classes: wide
excerpt: Implementation of agentic workflow using AWS Bedrock, Langchain and LangGraph

---

## Introduction
As Large Language Models are becoming extremely capable in using external tools and performing reasoning tasks, Agentic Workflow has become the new norm in designing intelligent systems. Instead of relying on a single prompt-response interaction, agentic systems utilize multiple specialized agents, introduce decision-making logic, and dynamically route execution based on intermediate outputs.

In this blog, I will be implementing an agentic workflow implementation by using AWS Bedrock API, LangChain and LangGraph.

## Architecture for Implementation
![Model Deployment Screenshot](/images/blog-images/bedrock-langgraph-agents/graph_visualization.png)

An example use case of clothing recommendation based on the user inputs will be implemented. It consists of:
1. Supervisor Agent: Extracts key features (gender, occasion, color, season, personal_fashion_taste) from user input.
2. Router: Checks if a color was mentioned.
3. Color Recommender (Conditional): If no color was provided, this agent suggests one based on the season and occasion.
4. Clothing Recommender: Takes all the data and provides the final styling advice.

## Library Imports
To implement the AWS Bedrock model invocation, organize the prompt, message and orchestrate the whole workflow, following imports are needed. (Note: If there are any missing libraries, install them in your environment.)
```
import json
from typing import Literal
import boto3
from botocore.config import Config

from langchain_aws import ChatBedrock
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
```

## Setting Up AWS Bedrock Client
In this step, a Bedrock client is created using boto3 that accesses your locally configured AWS credentials for accessing the Foundational Models from Bedrock. It is then integrated with LangChain’s ChatBedrock for initializing the client to perform all the invocations. We will be using the Amazon Nova Micro model for all the LLM invocations.
```
class BedrockClient:
    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client(service_name='bedrock-runtime', 
                                   region_name=region_name, 
                                   config=Config(retries={"max_attempts": 10}))

    def invoke_model(self, prompt: str, params: dict = {}):
        lc_client = ChatBedrock(client=self.client, model_id=params.get("model_id", "amazon.nova-micro-v1:0"))
        response = lc_client.invoke(prompt)

        return response
    
bedrock_client = BedrockClient()
```

## Defining Agents
In LangGraph, each agent is a “Node.” Nodes receive the current State (a list of messages), perform an action, and return an update to that state.

### Supervisor Agent: Information Extraction
This agent understands the natural language from human inputs and extracts the key feature attributes if available in JSON format.
```
def supervisor_agent(state: MessagesState):
    system_prompt = SystemMessagePromptTemplate.from_template(
        """# Instruction
        You are tasked with extracting value for key features from the input text. Key features include: gender, color, occasion, season or personal 
        fashion taste. If no value is found for respective categories, use "not_available"

        #Output 
        Use the following JSON format for output in all lowercase. Do not add any additional explanations, punctuations or reasonings. 
        ## JSON Format
        {{"gender": "gender mentioned in the text, else assume male", 
        "color": "color mentioned in the text for clothes or footwear",
        "occasion": "occasion that the clothes or footwear is intended for",
        "season": "season that is mentioned or occasion that usually occurs in particular season", 
        "personal_fashion_taste": "any additional fashion taste or clues in the inputs"}}
        """
    )

    user_prompt_template = HumanMessagePromptTemplate.from_template(
    """
    #Inputs
    input_text : {input_text}
    """
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt_template])

    formatted_prompt = chat_prompt.format_prompt(input_text=state["messages"][0].content)
    response = bedrock_client.invoke_model(prompt=formatted_prompt)

    return {"messages": [response]}
```

### Color Recommender Agent: Color Recommendation
The Color Recommender Agent only runs when required. Its main objective is to recommend the best possible color based on given gender, season, and occasion attributes.
```
def color_recommender_agent(state: MessagesState):
    supervisor_state_message = json.loads(state["messages"][-1].content)

    system_prompt = SystemMessagePromptTemplate.from_template(
        """System Prompt
        # Instruction
        You are tasked with recommending the best color for clothes and footwear  based on the gender, season and occasion. 

        #Output 
        Use the following JSON format for output in all lowercase. Do not add any additional explanations, punctuations or reasonings. 
        ## JSON Format
        {{"color": ""}}
        """
    )

    user_prompt_template = HumanMessagePromptTemplate.from_template(
    """
    #Inputs
    gender: {gender}
    occasion: {occasion}
    season: {season}
    """
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt_template])

    formatted_prompt = chat_prompt.format_prompt(gender=supervisor_state_message.get("gender"), 
                                           occasion=supervisor_state_message.get("occasion"), 
                                           season=supervisor_state_message.get("season"))
    response = bedrock_client.invoke_model(formatted_prompt)

    return {"messages": [response]}
```

### Clothing Recommender Agent: Final Clothing Recommendation
The final agent is responsible for evaluating every key feature attribute and providing the best possible recommendation for clothing.
```
def clothing_recommender_agent(state: MessagesState):
    state_message = json.loads(state["messages"][-1].content)

    if state_message["color"] == "not_available":
        supervisor_state_message = json.loads(state["messages"][1].content)
        recommended_color = json.loads(state["messages"][2].content)["color"]
    
    else:
        supervisor_state_message = state_message
        recommended_color = supervisor_state_message["color"]

    system_prompt = SystemMessagePromptTemplate.from_template(
        """System Prompt
        # Instruction
        You are tasked with recommending the clothing and styling summary based on the available key features: gender, color, occasion, season or personal fashion taste.

        #Output 
        Use the following JSON format for output in all lowercase. Do not add any additional explanations, punctuations or reasonings. 
        ## JSON Format
        {{"clothing": "recommend and explain the styling in less than 30 words"}}
        """
    )

    user_prompt_template = HumanMessagePromptTemplate.from_template(
    """
    #Inputs
    gender: {gender}
    occasion: {occasion}
    season: {season}
    color: {color}
    personal_fashion_taste: {personal_fashion_taste}
    """
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt_template])

    formatted_prompt = chat_prompt.format_prompt(gender=supervisor_state_message.get("gender"), 
                                           occasion=supervisor_state_message.get("occasion"), 
                                           season=supervisor_state_message.get("season"),  
                                           color=recommended_color,  
                                           personal_fashion_taste=supervisor_state_message.get("personal_fashion_taste"))
    response = bedrock_client.invoke_model(formatted_prompt)

    return {"messages": [response]}
```

### Router Agent: Explicit Control Flow
The Router isn’t an LLM call. It’s a Python function that determines the next step in the graph based on the output from “Supervisor Agent”. If the extracted color is not_available, execution is routed to the “Color Recommender”. Otherwise, the workflow skips directly to the “Clothing Recommender Agent”.
```
def router_agent(state: MessagesState) -> Literal["color_available", "color_unavailable"]:
    supervisor_state_message = json.loads(state["messages"][1].content)
    if supervisor_state_message["color"] == "not_available":
        return "color_unavailable"
    
    else:
        return "color_available"
```

## Orchestrating with LangGraph
Finally, we will initialize LangGraph’s StateGraph, which treats your workflow like a flowchart where each step is a node and each transition is an edge. Here we initialize StateGraph with MessagesState, which acts as the shared memory for all agents.

Nodes are the functional building blocks of our graph, so we will register each of our python functions for agents as nodes to the graph.

Edges determine the path the data takes or executes through the graph.
1. The Entry Point: graph.add_edge(START, “supervisor_agent”) tells the graph to always begin with the supervisor.
2. Conditional Logic: Instead of a straight line, the router_agent acts as a traffic controller. If the supervisor determines the user didn’t mention a color, the router_agent returns “color_unavailable”, forcing the graph to visit the color_recommender_agent first.
3. The Exit: Finally, every path converges at the clothing_recommender_agent before reaching the END.

The final step, graph.compile(), transforms your defined nodes and edges into a single executable object. This compiled graph can then be invoked with an initial message, and it will automatically handle the routing and state management until it reaches the end of the chain.

```
if __name__ == "__main__":
    
    graph = StateGraph(MessagesState)

    # LangGraph Node Definitions
    graph.add_node(supervisor_agent, name="supervisor_agent")
    graph.add_node(router_agent, name="router_agent")
    graph.add_node(color_recommender_agent, name="color_recommender_agent")
    graph.add_node(clothing_recommender_agent, name="clothing_recommender_agent")

    # LangGraph Edge Definitions
    graph.add_edge(START, "supervisor_agent")

    graph.add_conditional_edges(
        "supervisor_agent", 
        router_agent, 
        {"color_unavailable": "color_recommender_agent", 
         "color_available": "clothing_recommender_agent"})
    
    graph.add_edge("color_recommender_agent", "clothing_recommender_agent")
    graph.add_edge("clothing_recommender_agent", END)

    # LangGraph Compilation
    graph = graph.compile()

    # LangGraph Input State and Graph Invoke
    input_text = "I am male looking for fashion ideas to attend my friend's wedding in Fall.."
    input_state = {"messages": [HumanMessage(content=input_text)]}

    result = graph.invoke(input_state)
    
    print(f"Input user text: {input_text}")
    print(f"Agentic Response: {result["messages"][-1].content}")
```

## Save compiled LangGraph
Optionally, you can save the compiled graph which visualizes the nodes and edges in our implementation.
```
png_data = graph.get_graph().draw_mermaid_png()

# Save the bytes to a file
 with open("outputs/graph_visualization.png", "wb") as f:
    f.write(png_data)
```

## Conclusion
By using LangGraph with AWS Bedrock, we’ve moved away from a “black box” chat and toward a transparent, reliable workflow. With this implementation reference, you should be able to create your own agents with multiple logical nodes & edges, conditional edges and utilize langchain_aws library.