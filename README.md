# DeepWebResearcher

DeepWebResearcher is an advanced research engine that automates comprehensive web research, fact-checking, and content creation using AI agents and LLM models.

## Overview

This tool helps users conduct deep research on any topic by:
1. Optimizing search queries
2. Gathering comprehensive information from reliable sources
3. Extracting and verifying key claims
4. Generating well-structured content in various formats

## Architecture

### Core Components

1. **Research Agent**: Optimizes queries and gathers information from the web
2. **Fact-Checking Agent**: Verifies claims against trusted sources
3. **Content Generation Agent**: Creates polished drafts in various styles
4. **Workflow Orchestration**: Manages the research pipeline using LangGraph

### Technology Stack

- **LLM Models**: Groq's deepseek-r1-distill-llama-70b
- **Search API**: Tavily Search API for web research and fact verification
- **Workflow Management**: LangGraph for agent orchestration
- **Frontend**: Gradio for the user interface
- **Backend**: Python with LangChain for LLM interactions

## Workflow Process

### 1. Query Optimization
The system takes a user query and transforms it into a more specific, detailed search query using the LLM.

```python
def optimize_query_directly(query: str) -> str:
    """Optimize a query directly using the LLM"""
    optimization_prompt = ChatPromptTemplate.from_template("""
    You are a query optimization expert. Your task is to transform natural language queries into 
    detailed, domain-specific optimized queries that can be processed by specialized systems.
    
    Original query: {query}
    
    Please provide an optimized version of this query that:
    1. Is more specific and detailed
    2. Includes relevant domain terminology
    3. Is structured for better processing by downstream systems
    4. Maintains the original intent of the query
    
    Optimized query:
    """)
    
    chain = optimization_prompt | research_llm | StrOutputParser()
    return chain.invoke({"query": query})
```

### 2. Research Gathering
Using the Tavily Search API, the system gathers relevant information from the web and summarizes it into a structured research output.

```python
def conduct_research(state: ResearchState) -> ResearchState:
    """Conduct research based on the optimized query"""
    # Search for information using Tavily
    search_results = tavily_search.invoke(state["optimized_query"])
    
    # Summarize the search results
    research_output = summarize_search_results(state["optimized_query"], search_results)
    
    return {"research_output": research_output}
```

### 3. Claim Extraction
The system identifies 3-5 key factual claims from the research output that need verification.

```python
def extract_claims(research_output):
    """Extract key factual claims from research output"""
    extraction_prompt = ChatPromptTemplate.from_template("""
    You are an expert at identifying factual claims in text. 
    From the following research output, extract the 3-5 most significant factual claims that should be verified.
    
    Research output:
    {research_output}
    
    For each claim, provide:
    1. The claim statement
    2. The importance of verifying this claim (high/medium/low)
    
    Format your response as a JSON array of objects with "claim" and "importance" fields.
    """)
    
    chain = extraction_prompt | fact_checker_llm | JsonOutputParser()
    return chain.invoke({"research_output": research_output})
```

### 4. Fact Checking
Each claim is verified against trusted sources using a separate search and analysis process.

```python
def verify_claim(claim):
    """Verify a single claim using search and LLM analysis"""
    # Search for verification information
    search_results = fact_verification_search.invoke(claim)
    
    # Format verification data
    verification_data = "\n\n".join([
        f"Source: {result.get('url', 'Unknown')}\n"
        f"Title: {result.get('title', 'No title')}\n"
        f"Content: {result.get('content', 'No content')}"
        for result in search_results
    ])
    
    # Run credibility check
    chain = credibility_check_prompt | fact_checker_llm | JsonOutputParser()
    return chain.invoke({"claim": claim, "verification_data": verification_data})
```

### 5. Fact-Check Report Generation
The system compiles a comprehensive fact-check report based on the verification results.

```python
def generate_fact_check_report(state: ResearchState) -> ResearchState:
    """Generate a comprehensive fact-check report"""
    # Generate report using LLM
    chain = overall_report_prompt | fact_checker_llm | StrOutputParser()
    fact_check_report = chain.invoke({
        "research_output": state["research_output"],
        "verification_results": json.dumps(clean_verification_results, indent=2),
        "references": "\n".join(state["references"])
    })
    
    return {"fact_check_report": fact_check_report}
```

### 6. Content Creation
Finally, the system generates a polished content draft in the user's chosen style (blog post, detailed report, or executive summary).

```python
def create_draft_content(state: ResearchState) -> ResearchState:
    """Draft content in the specified style"""
    chain = draft_prompt | research_llm | StrOutputParser()
    draft_content = chain.invoke({
        "optimized_query": state["optimized_query"],
        "research": state["research_output"],
        "fact_check": state["fact_check_report"],
        "style": state["content_style"],
        "references": "\n".join(state["references"])
    })
    
    return {
        "draft_content": draft_content,
        "status": "completed"
    }
```

## User Interface

The system provides a Gradio-based web interface that allows users to:
1. Enter a research query
2. Select a content style (Blog post, Detailed report, or Executive summary)
3. View the research results in multiple tabs:
   - Query Information
   - Fact-Check Report
   - Research work (Final Content)
   - Drafted information (References)

## API Usage

### Groq API
- Used for accessing the deepseek-r1-distill-llama-70b LLM model
- Handles query optimization, research summarization, claim extraction, fact-checking, and content generation

### Tavily Search API
- Provides web search capabilities for research gathering
- Used for fact verification with specialized search parameters
- Configured with different search depths based on the task

## Libraries Used

### LangChain
- `langchain_groq`: Interface to Groq's LLM models
- `langchain_core`: Core components for prompts and output parsing
- `langchain.tools`: Integration with external tools like Tavily Search

### LangGraph
- Used for creating and managing the research workflow
- Handles state management between different research stages
- Provides a directed graph structure for the research pipeline

### Gradio
- Creates the web-based user interface
- Handles user inputs and displays research outputs
- Organizes results in a tabbed interface for better user experience

## Output Formats

### Research Output
Comprehensive information gathered from reliable sources, structured in a logical manner.

### Fact-Check Report
A detailed analysis of key claims, including:
- Overall reliability score (0-10)
- Significant accuracy issues
- Context for misleading information
- Suggested improvements
- References to verification sources

### Content Draft
A polished document in the user's chosen style:
1. **Blog Post**: Conversational tone with clear headings and actionable insights
2. **Detailed Report**: Comprehensive structure with executive summary, methodology, findings, and recommendations
3. **Executive Summary**: Concise overview highlighting key findings and strategic implications

## Setup and Usage

### Environment Variables
The system requires the following API keys:
- `GROQ_API_KEY`: For accessing Groq's LLM models
- `TAVILY_API_KEY`: For web search capabilities

### Running the Application
1. Install the required dependencies
2. Set up the environment variables
3. Run the Gradio interface:
```bash
python gradio_interface.py
```

## Limitations and Considerations
- The quality of research depends on the availability of information through the Tavily Search API
- Fact-checking is limited to publicly available information
- LLM-based analysis may occasionally contain inaccuracies or biases
- Processing time varies based on query complexity and API response times
