from typing import List    #environment setup and dependencies
from pydantic import BaseModel, Field
from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv
import datetime

load_dotenv()

class Document(BaseModel):
    content: str = Field(..., description="The content of the document")
    metadata: dict = Field(default_factory=dict, description="Document metadata")

class QueryResult(BaseModel):
    answer: str = Field(..., description="Response to the query")
    source_nodes: List[str] = Field(..., description="Source references used")

class QueryPreprocessor:
    def __init__(self):
 
        pass

    def preprocess(self, query: str) -> str:
        """Clean and standardize the query."""
        query = query.strip().lower()
        return query

class CustomAIAssistant:
    def __init__(self, data_path: str, index_path: str = "index"):
        """
        Initialize the AI Assistant
        :param data_path: Path to your document directory
        :param index_path: Path where the vector index will be stored
        """
        self.data_path = data_path
        self.index_path = index_path
        
        self.system_prompt = """
        You are a helpful AI Assistant specialized in:
        1. Providing guidance on job applications (e.g., how to apply, resume tips, interview prep).
        2. Analyzing job market trends (e.g., which industries are hiring, salary expectations, market demand).
        3. Offering general advice related to the job market, career paths, and skill development.
        """

        self.configure_settings()
        self.index = None
        self.agent = None
        self.load_or_create_index()

    def configure_settings(self):
        """Configure LLM and embedding settings"""
       
        Settings.llm = Groq(model="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        
        # Set up embedding model (e.g., Jina)
        Settings.embed_model = JinaEmbedding(
            api_key=os.getenv("JINA_API_KEY"),
            model="jina-embeddings-v2-base-en",
        )

    def load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path):
            self.load_index()
        else:
            self.create_index()
        self._create_agent()

    def load_index(self):
        """Load existing vector index"""
        storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
        self.index = load_index_from_storage(storage_context)

    def create_index(self):
        """Create new vector index from documents"""
        documents = SimpleDirectoryReader(
            self.data_path,
            recursive=True,
        ).load_data()

        if not documents:
            raise ValueError("No documents found in the specified path")
        
        self.index = VectorStoreIndex.from_documents(documents)
        self.save_index()

    def _create_agent(self):
        """Set up the agent with custom tools"""
        query_engine = self.index.as_query_engine(similarity_top_k=5)
        
        search_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="document_search",
                description="Search through the document database",
            ),
        )

        def custom_function(query: str) -> str:
            """Custom functionality to return the current date and time"""
            if "current date" in query.lower():
                return f"The current date and time is: {datetime.datetime.now()}"
        
        custom_tool = FunctionTool.from_defaults(
            fn=custom_function,
            name="custom_tool",
            description="Returns the current date and time if asked"
        )

        self.agent = ReActAgent.from_tools(
            [search_tool, custom_tool],
            verbose=True,
            system_prompt=self.system_prompt,
            memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
        )

    def query(self, query: str) -> QueryResult:
        """
        Process a query and return results.
        :param query: User's question or request.
        :return: QueryResult with answer and sources.
        """
        if not self.agent:
            raise ValueError("Agent not initialized")
        
        response = self.agent.chat(query)
        
        return QueryResult(
            answer=response.response,
            source_nodes=[],  # Add sources if needed
        )

    def save_index(self):
        """Save the vector index to disk"""
        os.makedirs(self.index_path, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.index_path)

if __name__ == "__main__":
    assistant = CustomAIAssistant(
        data_path="./your_data_directory",  # Path to your documents
        index_path="./your_index_directory"  # Path to store index
    )
    result = assistant.query("What are the best ways to prepare for a job interview?")
    print(result.answer)

