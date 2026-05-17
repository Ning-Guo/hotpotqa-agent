A Hybrid Retriever that intelligently combines BM25 keyword search and vector embeddings to retrieve the most relevant passages
A Research Agent that analyzes the retrieved content and generates an initial response
A Verification Agent that cross-checks the response against the original document to detect hallucinations and flag unsupported claims
A Self-Correction Mechanism that re-runs the research step if any contradictions or unsupported statements are found

This multi-step, verification-driven approach ensures that DocChat provides precise, document-grounded answers, even for complex and long-form documents that general-purpose chatbots struggle with. Whether you need to extract specific data points, summarize sections, compare multiple reports, or analyze tables, DocChat is built to help you navigate your documents with confidence.







Understand why multi-agent RAG is used
A Naïve RAG (Retrieval-Augmented Generation) pipeline is often insufficient for handling long, structured documents due to several limitations:

Limited query understanding: Naïve RAG processes queries at a single level, failing to break down complex questions into multiple reasoning steps. This results in shallow or incomplete answers when dealing with multi-faceted queries.

No hallucination detection or error handling: Traditional RAG pipelines lack a verification step. This means that if a response contains hallucinated or incorrect information, there's no mechanism to detect, correct, or refine the output.

Inability to handle out-of-scope queries: Without a proper scope-checking mechanism, Naïve RAG may attempt to generate answers even when no relevant information exists, leading to misleading or fabricated responses.

Inefficient multi-document retrieval: When multiple documents are uploaded, a Naïve RAG system might retrieve irrelevant or suboptimal passages, failing to select the most relevant content dynamically.

To overcome these challenges, DocChat implements a multi-agent RAG research system, which introduces intelligent agents to enhance retrieval, reasoning, and verification.

How multi-agent RAG solves these issues
Scope checking & routing
A Scope-Checking Agent first determines whether the user's question is relevant to the uploaded documents. If the query is out of scope, DocChat explicitly informs the user instead of generating hallucinated responses.
Dynamic multi-step query processing
For complex queries, an Agent Workflow ensures that the question is broken into smaller sub-steps, retrieving the necessary information before synthesizing a complete response.
For example, if a question requires comparing two sections of a document, an agent-based approach recognizes this need, retrieves both parts separately, and constructs a comparative analysis in the final answer.
Hybrid retrieval for multi-document contexts
When multiple documents are uploaded, the Hybrid Retriever (BM25 + Vector Search) ensures that the most relevant document(s) are selected dynamically, improving accuracy over traditional retrieval pipelines.

Fact verification & self-correction
After an initial response is generated, a Verification Agent cross-checks the output against the retrieved documents.
If any contradictions or unsupported claims are found, the Self-Correction Mechanism refines the answer before presenting it to the user.
Shared global state for context awareness
The Agent Workflow maintains a shared state, allowing each step (retrieval, reasoning, verification) to reference previous interactions and refine responses dynamically.
This enables context-aware follow-up questions, ensuring that users can refine their queries without losing track of previous answers.

Setting up your development environment

Project overview
The content of this lab is licensed under Apache 2.0












Project overview
Below is a breakdown of DocChat's workflow
project overview

1 - User query processing & relevance analysis
The system starts when a user submits a question about their uploaded document(s)
Before retrieving any data, DocChat first analyzes query relevance to determine if the question is within the scope of the uploaded content
2 - Routing & query categorization
The query is routed through an intelligent agent that decides whether the system can answer it using the document(s):
In scope: Proceed with document retrieval and response generation.
Not in scope: Inform the user that the question cannot be answered based on the provided documents, preventing hallucinations.
3 - Multi-agent research & document retrieval
If the query is relevant, DocChat retrieves relevant document sections from a hybrid search system:
Docling converts the document into a structured Markdown format for better chunking
LangChain splits the document into logical chunks based on headers and stores them in ChromaDB (a vector store)
The retrieval module searches for the most contextually relevant document chunks using BM25 and vector search
4 - Answer generation & verification loop
Conduct research:

The research agent generates an initial answer based on retrieved content
A sub-process starts where queries are dynamically generated for more precise retrieval
Verification process:

The verification agent cross-checks the generated response against the retrieved content
If the response is fully supported, the system finalizes and returns the answer
If verification fails (e.g., hallucinations, unsupported claims), the system re-runs the research step until a verifiable response is found
5 - Response finalization
After verification is complete, DocChat returns the final response to the user
The workflow ensures that each answer is sourced directly from the provided document(s), preventing fabrication or unreliable outputs
In the next step, start by building the vector database.


Understand why multi-agent RAG is used

Build vector database
The content of this lab is licensed under Apache 2.0