# Deep Research with LangGraph and Gemini 2.5

This implementation demonstrates a sophisticated deep research system using LangGraph with Gemini 2.5, featuring integrated thinking summaries within the same trace.

## Features

- **Multi-step Research Workflow**: Automatically generates and executes research plans
- **Thinking Summaries**: Gemini 2.5 provides reasoning traces within each step
- **State Management**: Uses LangGraph's state management with checkpoints
- **Configurable Depth**: Research depth from 1-5 levels
- **Trace Integration**: All thinking summaries are captured within the same run

## Installation

```bash
pip install langgraph langchain-google-genai langchain-core
export GOOGLE_API_KEY="your-gemini-api-key"
```

## Usage

```python
from deep_research_graph import DeepResearchGraph

# Initialize the research graph
researcher = DeepResearchGraph(model_name="gemini-2.5-pro-exp")

# Run deep research
results = researcher.run_research(
    query="What are the latest developments in quantum computing?",
    research_depth=4
)

# Access results
print(results["final_report"])  # Comprehensive research report
print(results["thinking_summaries"])  # All thinking traces
```

## Architecture

### State Structure
```python
class ResearchState(TypedDict):
    query: str                    # Research question
    research_plan: List[str]      # Generated research steps
    current_step: int            # Current step index
    findings: List[Dict]         # Research findings per step
    thinking_summaries: List[ThinkingSummary]  # Reasoning traces
    final_report: str              # Synthesized final report
    research_depth: int           # Configurable depth (1-5)
```

### Graph Flow
1. **Generate Plan**: Creates research steps based on query
2. **Research Steps**: Executes each step with thinking summaries
3. **Conditional Routing**: Continues until all steps complete
4. **Final Synthesis**: Combines all findings into comprehensive report

### Thinking Summary Format
Each step generates:
- **Step name**: Current research phase
- **Reasoning**: Detailed thinking process
- **Key insights**: Extracted important points
- **Confidence**: Numerical confidence score
- **Timestamp**: When generated

## Example Output Structure

```json
{
    "query": "Research question",
    "final_report": "Comprehensive research findings...",
    "research_plan": ["Step 1", "Step 2", "Step 3"],
    "findings": [
        {
            "step": "Step 1",
            "content": "Research findings...",
            "timestamp": "2025-01-01T12:00:00"
        }
    ],
    "thinking_summaries": [
        {
            "step": "research_planning",
            "reasoning": "I'll approach this by...",
            "key_insights": ["Key insight 1", "Key insight 2"],
            "confidence": 0.85,
            "timestamp": "2025-01-01T12:00:00"
        }
    ],
    "total_steps": 3,
    "completed_steps": 3
}
```

## Advanced Usage

### Custom Configuration
```python
researcher = DeepResearchGraph(
    model_name="gemini-2.5-pro-exp",
    temperature=0.3  # Lower temperature for more focused research
)
```

### Thread Management
```python
# Continue research in same conversation
results = researcher.run_research(
    query="Tell me more about the quantum cryptography aspects",
    thread_id="previous_research_session"
)
```

### Access Thinking Summaries
```python
for summary in results["thinking_summaries"]:
    print(f"Step: {summary['step']}")
    print(f"Confidence: {summary['confidence']}")
    print(f"Reasoning: {summary['reasoning']}")
```

## Key Benefits

1. **Transparent Reasoning**: Every step includes thinking summaries
2. **Traceable Research**: Full audit trail of reasoning process
3. **Configurable Depth**: Adapt research thoroughness to needs
4. **Memory Persistence**: Conversations can continue across sessions
5. **Structured Output**: Consistent, parseable results

The thinking summaries are generated within the same LangGraph execution trace, ensuring they're properly associated with their corresponding research steps and available for analysis or debugging.