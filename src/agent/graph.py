import boto3
import json
from typing import Any, Dict
from langchain_community.chat_models import BedrockChat
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from state import State
from configuration import Configuration
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#import logging
#logging.basicConfig(level=logging.DEBUG)

pdf_path = "/home/sagemaker-user/langgraph/lang-risk-assess/src/creditReport.pdf" 

# Initialize Bedrock Chat
llm = BedrockChat(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    streaming=True,
#    client=bedrock_client,
#    callbacks=[StreamingStdOutCallbackHandler()]  # Add streaming callback
)
print("Bedrock Chat initialized")
def load_and_split_pdf(state: State):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(pages)
    text_chunks = [chunk.page_content for chunk in chunks]

    print("Completed node 1")
    return {"messages": [], "pdf_chunks": text_chunks}

def analyze_credit_report(state: Dict[str, Any]):
    pdf_chunks = state.get("pdf_chunks", [])

    if not pdf_chunks:
        raise ValueError("PDF chunks are empty. Ensure the PDF content was loaded correctly.")

    combined_content = "\n\n".join(pdf_chunks)

    analysis_prompt = f"""You are a credit analyst reviewing a credit report. 
    Please analyze the following credit report information and provide:
    1. Overall credit score assessment
    2. Key findings and red flags
    3. Payment history analysis
    4. Credit utilization review
    5. Recommendations for improvement

    Credit Report Content:
    {combined_content}
    """
    messages = [{"role": "user", "content": analysis_prompt}]

    response = llm.invoke(messages)
    print("Completed node 2")
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response}], 
        "pdf_chunks": state["pdf_chunks"]
    }

def loan_officer_report(state: Dict[str, Any]):
    state_messages = state.get("messages", [])

    if not state_messages:
        raise ValueError("State Messages are empty.")
    
    financial_credit_factors = state_messages[-1]["content"] 
    
    financial_prompt = f"""You are an experienced loan officer responsible for evaluating loan applications.

    ### Task:
    Based on the provided financial and credit factors, generate a **concise and precise** loan decision report.

    ### Input:
    **Financial and Credit Factors:** {financial_credit_factors}

    ### Requirements:
    - Provide an **loan decision report** of the applicant.
    - Identify **potential risk factors** affecting their ability to repay.
    - Offer a **clear recommendation** on whether to approve, deny, or modify the loan terms.
    - The report should be **limited to one page or less** for clarity and efficiency.

    ### Tone & Style:
    - Be professional, objective, and fact-based.
    - Avoid unnecessary details—keep it **concise and actionable**.

    ### Example Input:
    **Financial and Credit Factors:** 
    - Credit Score: 680
    - Debt-to-Income Ratio: 45%
    - Payment History: 2 missed payments in the last 12 months
    - Employment: Stable, 5+ years at current job

    ### Expected Output Example:
    "The applicant demonstrates moderate creditworthiness with a fair credit score (680) and stable employment history. However, the high debt-to-income ratio (45%) and recent missed payments pose a repayment risk. Recommendation: Loan approval with conditions—require a lower loan amount or additional collateral to mitigate risk."

    Now, generate the **Credit Decision Report** based on the provided financial and credit factors.
    """

    messages = [{"role": "user", "content": financial_prompt}]

    response = llm.invoke(messages)
    print("Completed node 3")
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": response}], 
        "pdf_chunks": []
    }

# Define the graph with channels and specify output channels upfront
# Define the graph
workflow = StateGraph(dict, config_schema=Configuration)  # Change State to dict

# Add the node for credit report analysis load_and_split_pdf
workflow.add_node("load_and_split_pdf", load_and_split_pdf)
workflow.add_node("analyze_credit_report", analyze_credit_report)
workflow.add_node("loan_officer_report", loan_officer_report)

# Set the entrypoint
workflow.add_edge("__start__", "load_and_split_pdf")
workflow.add_edge("load_and_split_pdf","analyze_credit_report")
workflow.add_edge("analyze_credit_report","loan_officer_report")

# Set analyze_credit_report as a final node
workflow.set_finish_point("loan_officer_report")

# Compile the workflow
graph = workflow.compile()

graph.name = "Credit Report Analysis Graph"

state_input = {"messages": [{"role": "user", "content": "start"}], "pdf_chunks": []}
result = graph.invoke(state_input)

ai_message1 = result['messages'][-2]["content"]
print(ai_message1.content)

ai_message2 = result['messages'][-1]["content"]
print(ai_message2.content)



    
