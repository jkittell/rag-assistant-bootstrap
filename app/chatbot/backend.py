"""Chatbot backend handling request processing and response generation."""
from typing import Dict, Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from opentelemetry import trace
from ..data.processor import DataProcessor

class ChatbotBackend:
    def __init__(self, config, llm, data_processor: DataProcessor):
        self.config = config
        self.llm = llm
        self.data_processor = data_processor
        self.tracer = trace.get_tracer(__name__)
        
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.data_processor.vector_store.as_retriever(),
            return_source_documents=True
        )
        
        # Default prompt template
        self.prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know.
            
            Context: {context}
            
            Question: {question}
            
            Answer: """,
            input_variables=["context", "question"]
        )
    
    async def process_input(self, query: str) -> Dict[str, Any]:
        """Process user input and generate response."""
        with self.tracer.start_as_current_span("process_input") as span:
            span.set_attribute("query", query)
            
            # Get response from QA chain
            result = self.qa_chain({"query": query})
            
            # Format response
            response = {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]],
                "metadata": [doc.metadata for doc in result["source_documents"]]
            }
            
            return response
    
    def update_prompt(self, new_template: str) -> None:
        """Update the prompt template."""
        self.prompt_template = PromptTemplate(
            template=new_template,
            input_variables=["context", "question"]
        )
