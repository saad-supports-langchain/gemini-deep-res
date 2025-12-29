"""
Deep Research LangGraph Implementation with Gemini 2.5 and Thinking Summaries
"""

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
import json
import logging
from datetime import datetime
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.types import Command, StreamWriter
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig

try:
    from prompt_manager import PromptManager
    PROMPT_MANAGER_AVAILABLE = True
except ImportError:
    PROMPT_MANAGER_AVAILABLE = False
    PromptManager = None


@dataclass
class ThinkingSummary:
    """Container for thinking summary data"""
    step: str
    reasoning: str
    key_insights: List[str]
    confidence: float
    timestamp: str


class ResearchState(TypedDict):
    """State for the research graph"""
    query: str
    research_plan: List[str]
    current_step: int
    findings: List[Dict[str, Any]]
    thinking_summaries: List[ThinkingSummary]
    final_report: str
    research_depth: int  # 1-5 scale
    messages: List[Any]  # Store all messages with metadata


class DeepResearchGraph:
    """
    LangGraph implementation for deep research tasks using Gemini 2.5
    with integrated thinking summaries within the same trace.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        use_langsmith_prompts: bool = False,
        prompt_manager: Optional["PromptManager"] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.use_langsmith_prompts = use_langsmith_prompts
        self._llm = None
        self.memory = MemorySaver()
        
        if use_langsmith_prompts and PROMPT_MANAGER_AVAILABLE:
            self.prompt_manager = prompt_manager or PromptManager()
            logging.info("LangSmith prompt manager enabled")
        else:
            self.prompt_manager = None
            if use_langsmith_prompts:
                logging.warning(
                    "LangSmith prompts requested but prompt_manager not available"
                )
        
        self.graph = self._build_graph()

    async def _ensure_llm_initialized(self):
        """Ensure LLM is initialized in a non-blocking way."""
        if self._llm is None:
            # Run the blocking initialization in a thread pool
            self._llm = await asyncio.to_thread(
                ChatGoogleGenerativeAI,
                model=self.model_name,
                temperature=self.temperature
            )
        return self._llm

    @property
    def llm(self):
        """Get the LLM instance (may block on first access - prefer _ensure_llm_initialized)."""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature
            )
        return self._llm
        
    def _create_thinking_prompt(self, context: str, task: str) -> str:
        """Create a prompt that encourages thinking and summary generation"""
        
        if self.prompt_manager:
            prompt_template = self.prompt_manager.get_prompt(
                "thinking-prompt",
                fallback=self._get_default_thinking_prompt()
            )
            
            try:
                return prompt_template.format(context=context, task=task)
            except Exception as e:
                logging.warning(f"Failed to format LangSmith prompt: {e}")
                return self._get_default_thinking_prompt_formatted(context, task)
        
        return self._get_default_thinking_prompt_formatted(context, task)
    
    def _get_default_thinking_prompt(self) -> str:
        """Get the default thinking prompt template"""
        return """You are a research assistant with advanced reasoning capabilities.

Context: {context}

Task: {task}

Please provide your response in the following format:

<THINKING>
Briefly outline your thinking process, key considerations, and reasoning steps.
Include:
1. What approach you're taking
2. Key insights or patterns you notice
3. Any uncertainties or areas needing clarification
4. Your confidence level (0-1)
</THINKING>

<RESPONSE>
Your actual response to the task
</RESPONSE>

Ensure your thinking is thorough but concise, and your response is well-structured."""
    
    def _get_default_thinking_prompt_formatted(self, context: str, task: str) -> str:
        """Get the default thinking prompt with context and task filled in"""
        return f"""You are a research assistant with advanced reasoning capabilities.

Context: {context}

Task: {task}

Please provide your response in the following format:

<THINKING>
Briefly outline your thinking process, key considerations, and reasoning steps.
Include:
1. What approach you're taking
2. Key insights or patterns you notice
3. Any uncertainties or areas needing clarification
4. Your confidence level (0-1)
</THINKING>

<RESPONSE>
Your actual response to the task
</RESPONSE>

Ensure your thinking is thorough but concise, and your response is well-structured."""

    def _create_content_block(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a content block with text and metadata"""
        return {
            "type": "text",
            "text": content,
            "metadata": metadata
        }
    
    def _parse_thinking_response(self, response: str, step_name: str) -> tuple[str, ThinkingSummary]:
        """Parse the thinking and response sections from LLM output"""
        thinking_match = response.find("<THINKING>")
        response_match = response.find("<RESPONSE>")
        
        if thinking_match == -1 or response_match == -1:
            # Fallback if parsing fails
            return response, ThinkingSummary(
                step=step_name,
                reasoning="Failed to parse thinking section",
                key_insights=[],
                confidence=0.5,
                timestamp=datetime.now().isoformat()
            )
        
        thinking_start = thinking_match + len("<THINKING>")
        thinking_end = response.find("</THINKING>")
        response_start = response_match + len("<RESPONSE>")
        response_end = response.find("</RESPONSE>")
        
        thinking_content = response[thinking_start:thinking_end].strip()
        response_content = response[response_start:response_end].strip()
        
        # Extract key insights and confidence from thinking
        insights = []
        confidence = 0.7
        
        # Simple extraction logic
        lines = thinking_content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['insight:', 'key:', 'important:', 'finding:']):
                insights.append(line.strip())
            if 'confidence' in line.lower():
                try:
                    # Look for decimal number in confidence line
                    import re
                    conf_match = re.search(r'[\d.]+', line)
                    if conf_match:
                        confidence = float(conf_match.group())
                except:
                    pass
        
        thinking_summary = ThinkingSummary(
            step=step_name,
            reasoning=thinking_content,
            key_insights=insights or [thinking_content[:200]],  # Fallback to first 200 chars
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
        return response_content, thinking_summary

    async def _generate_research_plan(
        self,
        state: ResearchState,
        *,
        store: Optional[BaseStore] = None,
        config: RunnableConfig,
        writer: Optional[StreamWriter] = None
    ) -> Command:
        """
        Generate a research plan based on the query.
        """
        # Get the stream writer from context
        writer = get_stream_writer()
        
        # Ensure LLM is initialized without blocking
        llm = await self._ensure_llm_initialized()

        context = f"Research query: {state['query']}"
        research_depth = state.get('research_depth', 3)
        task = f"Create a detailed research plan with {research_depth} levels of depth. " \
               f"Break this down into specific research steps that will help gather comprehensive information."

        prompt = self._create_thinking_prompt(context, task)

        accumulated_content = ""
        content_blocks = []

        try:
            # Stream LLM response chunk by chunk with content blocks
            async for chunk in llm.astream([HumanMessage(content=prompt)], config=config):
                chunk_text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                accumulated_content += chunk_text

                # Create content block for this chunk
                content_block = self._create_content_block(
                    chunk_text,
                    {
                        "user_facing": True,
                        "fetch": False,
                        "is_last": False,
                        "node_name": "generate_plan",
                    }
                )
                content_blocks.append(content_block)
                
                # Stream the message chunk with content block
                writer(AIMessageChunk(content=[content_block]))

            # Parse the complete response
            response_content, thinking_summary = self._parse_thinking_response(
                accumulated_content, "research_planning"
            )

            # Parse research steps from response
            steps = []
            lines = response_content.split('\n')
            for line in lines:
                if any(line.strip().startswith(f"{i}.") for i in range(1, 20)):
                    steps.append(line.strip())

            if not steps:  # Fallback
                steps = ["Initial research", "Analysis", "Synthesis", "Final report"]

            # Store the research plan (if store is available)
            if store is not None:
                thread_id = config.get("configurable", {}).get("thread_id", "unknown")
                await store.aput(
                    namespace=(thread_id, "research"),
                    key="plan",
                    value={
                        "query": state['query'],
                        "steps": steps,
                        "research_depth": research_depth,
                        "created_at": datetime.now().isoformat()
                    }
                )

            # Emit final completion message with content block
            completion_text = f"Research plan created with {len(steps)} steps."
            completion_block = self._create_content_block(
                completion_text,
                {
                    "user_facing": True,
                    "fetch": True,
                    "is_last": True,
                    "node_name": "generate_plan",
                }
            )
            writer(AIMessage(content=[completion_block]))

            return Command(update={
                "research_plan": steps,
                "current_step": 0,
                "thinking_summaries": [thinking_summary],
                "messages": [
                    AIMessage(content=[completion_block])
                ]
            })

        except Exception as e:
            # Error handling with content block
            error_text = f"Error generating research plan: {str(e)}"
            error_block = self._create_content_block(
                error_text,
                {
                    "user_facing": True,
                    "fetch": False,
                    "is_last": True,
                    "node_name": "generate_plan",
                    "error": True,
                }
            )
            return Command(update={
                "messages": [
                    AIMessage(content=[error_block])
                ]
            })

    async def _execute_research_step(
        self,
        state: ResearchState,
        *,
        store: Optional[BaseStore] = None,
        config: RunnableConfig,
        writer: Optional[StreamWriter] = None
    ) -> Command:
        """
        Execute the current research step.
        """
        # Get the stream writer from context
        writer = get_stream_writer()
        
        current_step = state.get("current_step", 0)
        research_plan = state.get("research_plan", [])

        if current_step >= len(research_plan):
            return Command(update={})

        # Ensure LLM is initialized without blocking
        llm = await self._ensure_llm_initialized()

        current_step_name = research_plan[current_step]

        context = f"""
        Research Query: {state.get('query', '')}
        Current Step: {current_step_name}
        Previous Findings: {json.dumps(state.get('findings', []), indent=2)}
        """

        task = f"Execute this research step thoroughly. Focus on: {current_step_name}"
        prompt = self._create_thinking_prompt(context, task)

        accumulated_content = ""
        content_blocks = []

        try:
            # Stream LLM response chunk by chunk with content blocks
            async for chunk in llm.astream([HumanMessage(content=prompt)], config=config):
                chunk_text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                accumulated_content += chunk_text

                # Create content block for this chunk
                content_block = self._create_content_block(
                    chunk_text,
                    {
                        "user_facing": True,
                        "fetch": False,
                        "is_last": False,
                        "node_name": "research_step",
                        "step_number": current_step + 1,
                        "step_name": current_step_name,
                    }
                )
                content_blocks.append(content_block)
                
                # Stream the message chunk with content block
                writer(AIMessageChunk(content=[content_block]))

            # Parse the complete response
            response_content, thinking_summary = self._parse_thinking_response(
                accumulated_content, current_step_name
            )

            new_finding = {
                "step": current_step_name,
                "content": response_content,
                "timestamp": datetime.now().isoformat()
            }

            # Store the finding (if store is available)
            if store is not None:
                thread_id = config.get("configurable", {}).get("thread_id", "unknown")
                await store.aput(
                    namespace=(thread_id, "findings"),
                    key=f"step_{current_step}",
                    value=new_finding
                )

            # Determine if this is the last research step
            is_last_step = (current_step + 1) >= len(research_plan)

            # Emit final completion message with content block
            completion_text = f"Completed research step: {current_step_name}"
            completion_block = self._create_content_block(
                completion_text,
                {
                    "user_facing": True,
                    "fetch": True,
                    "is_last": True,
                    "node_name": "research_step",
                    "step_number": current_step + 1,
                    "step_name": current_step_name,
                    "is_final_step": is_last_step,
                }
            )
            writer(AIMessage(content=[completion_block]))

            return Command(update={
                "findings": state.get("findings", []) + [new_finding],
                "thinking_summaries": state.get("thinking_summaries", []) + [thinking_summary],
                "current_step": current_step + 1,
                "messages": [
                    AIMessage(content=[completion_block])
                ]
            })

        except Exception as e:
            # Error handling with content block
            error_text = f"Error in research step '{current_step_name}': {str(e)}"
            error_block = self._create_content_block(
                error_text,
                {
                    "user_facing": True,
                    "fetch": False,
                    "is_last": True,
                    "node_name": "research_step",
                    "error": True,
                }
            )
            return Command(update={
                "messages": [
                    AIMessage(content=[error_block])
                ]
            })

    async def _synthesize_final_report(
        self,
        state: ResearchState,
        *,
        store: Optional[BaseStore] = None,
        config: RunnableConfig,
        writer: Optional[StreamWriter] = None
    ) -> Command:
        """
        Synthesize all findings into a final research report.
        """
        # Get the stream writer from context
        writer = get_stream_writer()
        
        # Ensure LLM is initialized without blocking
        llm = await self._ensure_llm_initialized()

        context = f"""
        Original Query: {state.get('query', '')}
        Research Plan: {json.dumps(state.get('research_plan', []), indent=2)}
        All Findings: {json.dumps(state.get('findings', []), indent=2)}
        Thinking Summaries: {[s.reasoning for s in state.get('thinking_summaries', [])]}
        """

        task = "Synthesize all the research findings into a comprehensive final report. " \
               "Include key insights, conclusions, and any recommendations. Make it well-structured and detailed."

        prompt = self._create_thinking_prompt(context, task)

        accumulated_content = ""
        content_blocks = []

        try:
            # Stream LLM response chunk by chunk with content blocks
            async for chunk in llm.astream([HumanMessage(content=prompt)], config=config):
                chunk_text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                accumulated_content += chunk_text

                # Create content block for this chunk
                content_block = self._create_content_block(
                    chunk_text,
                    {
                        "user_facing": True,
                        "fetch": False,
                        "is_last": False,
                        "node_name": "synthesize_report",
                    }
                )
                content_blocks.append(content_block)
                
                # Stream the message chunk with content block
                writer(AIMessageChunk(content=[content_block]))

            # Parse the complete response
            response_content, thinking_summary = self._parse_thinking_response(
                accumulated_content, "final_synthesis"
            )

            # Store the final report (if store is available)
            if store is not None:
                thread_id = config.get("configurable", {}).get("thread_id", "unknown")
                await store.aput(
                    namespace=(thread_id, "research"),
                    key="final_report",
                    value={
                        "report": response_content,
                        "query": state.get('query', ''),
                        "total_steps": len(state.get("research_plan", [])),
                        "completed_at": datetime.now().isoformat()
                    }
                )

            # Emit final completion message with content block
            completion_text = "Final research report completed."
            completion_block = self._create_content_block(
                completion_text,
                {
                    "user_facing": True,
                    "fetch": True,
                    "is_last": True,
                    "node_name": "synthesize_report",
                    "total_steps": len(state.get("research_plan", [])),
                }
            )
            writer(AIMessage(content=[completion_block]))

            return Command(update={
                "final_report": response_content,
                "thinking_summaries": state.get("thinking_summaries", []) + [thinking_summary],
                "messages": [
                    AIMessage(content=[completion_block])
                ]
            })

        except Exception as e:
            # Error handling with content block
            error_text = f"Error synthesizing final report: {str(e)}"
            error_block = self._create_content_block(
                error_text,
                {
                    "user_facing": True,
                    "fetch": False,
                    "is_last": True,
                    "node_name": "synthesize_report",
                    "error": True,
                }
            )
            return Command(update={
                "messages": [
                    AIMessage(content=[error_block])
                ]
            })

    def _should_continue_research(self, state: ResearchState) -> str:
        """Determine if research should continue"""
        if state.get("current_step", 0) < len(state.get("research_plan", [])):
            return "continue"
        return "synthesize"

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_plan", self._generate_research_plan)
        workflow.add_node("research_step", self._execute_research_step)
        workflow.add_node("synthesize_report", self._synthesize_final_report)
        
        # Add edges
        workflow.add_edge("generate_plan", "research_step")
        workflow.add_conditional_edges(
            "research_step",
            self._should_continue_research,
            {
                "continue": "research_step",
                "synthesize": "synthesize_report"
            }
        )
        workflow.add_edge("synthesize_report", END)
        
        # Set entry point
        workflow.set_entry_point("generate_plan")
        
        return workflow.compile(checkpointer=self.memory)
    
    def _build_graph_for_langgraph_api(self) -> CompiledStateGraph:
        """Build the LangGraph workflow for API deployment (no custom checkpointer)"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_plan", self._generate_research_plan)
        workflow.add_node("research_step", self._execute_research_step)
        workflow.add_node("synthesize_report", self._synthesize_final_report)
        
        # Add edges
        workflow.add_edge("generate_plan", "research_step")
        workflow.add_conditional_edges(
            "research_step",
            self._should_continue_research,
            {
                "continue": "research_step",
                "synthesize": "synthesize_report"
            }
        )
        workflow.add_edge("synthesize_report", END)
        
        # Set entry point
        workflow.set_entry_point("generate_plan")
        
        # For LangGraph API deployment, don't use custom checkpointer
        return workflow.compile()

    async def run_research(self, query: str, research_depth: int = 3, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete research task with thinking summaries (async).

        Args:
            query: The research question or topic
            research_depth: Depth of research (1-5)
            thread_id: Optional thread ID for conversation memory

        Returns:
            Dictionary containing the research results and thinking summaries
        """
        if thread_id is None:
            thread_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        initial_state = {
            "query": query,
            "research_plan": [],
            "current_step": 0,
            "findings": [],
            "thinking_summaries": [],
            "final_report": "",
            "research_depth": max(1, min(5, research_depth)),
            "messages": []
        }

        config = RunnableConfig(configurable={"thread_id": thread_id})

        # Run the graph asynchronously
        result = await self.graph.ainvoke(initial_state, config)

        return {
            "query": result["query"],
            "final_report": result["final_report"],
            "research_plan": result["research_plan"],
            "findings": result["findings"],
            "thinking_summaries": [
                {
                    "step": ts.step,
                    "reasoning": ts.reasoning,
                    "key_insights": ts.key_insights,
                    "confidence": ts.confidence,
                    "timestamp": ts.timestamp
                }
                for ts in result["thinking_summaries"]
            ],
            "total_steps": len(result["research_plan"]),
            "completed_steps": result["current_step"]
        }


async def main():
    """Main async function for running research."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize the research graph
    research_graph = DeepResearchGraph()

    # Example research query
    query = "What are the latest developments in quantum computing and their potential impact on cryptography?"

    print(f"Starting deep research on: {query}")
    print("=" * 80)

    # Run the research asynchronously
    results = await research_graph.run_research(query, research_depth=4)

    # Display results
    print(f"\nFinal Research Report:")
    print("=" * 80)
    print(results["final_report"])

    print(f"\nThinking Summaries ({len(results['thinking_summaries'])} total):")
    print("=" * 80)
    for i, ts in enumerate(results["thinking_summaries"], 1):
        print(f"\n{i}. Step: {ts['step']}")
        print(f"   Confidence: {ts['confidence']:.2f}")
        print(f"   Reasoning: {ts['reasoning'][:200]}...")
        if ts['key_insights']:
            print(f"   Key Insights: {', '.join(ts['key_insights'][:3])}")

    print(f"\nResearch completed in {results['completed_steps']} steps.")


if __name__ == "__main__":
    # Example usage
    import os

    asyncio.run(main())


def create_deep_research_graph():
    """Factory function to create a deep research graph instance"""
    return DeepResearchGraph().graph

def create_deep_research_graph_for_api():
    """Factory function to create a deep research graph instance for LangGraph API"""
    instance = DeepResearchGraph()
    return instance._build_graph_for_langgraph_api()

def create_initial_state_for_api(query: str, research_depth: int = 3) -> dict:
    """Create a properly formatted initial state for API calls"""
    return {
        "query": query,
        "research_plan": [],
        "current_step": 0,
        "findings": [],
        "thinking_summaries": [],
        "final_report": "",
        "research_depth": max(1, min(5, research_depth)),
        "messages": []
    }

# Create default graph instances for LangGraph configuration
try:
    deep_research_graph = create_deep_research_graph()
    deep_research_graph_for_api = create_deep_research_graph_for_api()
except Exception as e:
    # If initialization fails (e.g., no API key), create placeholders
    import warnings
    warnings.warn(f"Could not initialize default graphs: {e}. Graphs will be initialized when needed.")
    deep_research_graph = None
    deep_research_graph_for_api = None
