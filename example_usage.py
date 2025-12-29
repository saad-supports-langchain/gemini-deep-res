#!/usr/bin/env python3
"""
Example usage of the Deep Research Graph with Gemini 2.5 and thinking summaries
"""

import os
import json
import asyncio
from datetime import datetime
from deep_research_graph import DeepResearchGraph


async def main():
    # Set up API key (ensure GOOGLE_API_KEY is set in environment)
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not found in environment variables")
        print("Please set your Gemini API key: export GOOGLE_API_KEY='your-key'")
        return
    
    # Initialize the research graph
    print("🚀 Initializing Deep Research Graph with Gemini 2.5...")
    researcher = DeepResearchGraph(
        model_name="gemini-2.5-pro-exp",
        temperature=0.7
    )
    
    # Example research queries
    research_queries = [
        "What are the latest developments in quantum computing and their potential impact on cryptography?",
        "How is AI transforming healthcare diagnostics and what are the regulatory challenges?",
        "What are the environmental impacts of cryptocurrency mining and potential solutions?"
    ]
    
    # Run research for each query
    for i, query in enumerate(research_queries, 1):
        print(f"\n{'='*80}")
        print(f"🔍 Research Task {i}: {query}")
        print(f"{'='*80}")
        
        # Run the research with thinking summaries
        results = await researcher.run_research(
            query=query,
            research_depth=4,
            thread_id=f"research_session_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Display results
        print(f"\n📊 Research Summary:")
        print(f"   • Total Steps: {results['total_steps']}")
        print(f"   • Completed Steps: {results['completed_steps']}")
        print(f"   • Thinking Summaries: {len(results['thinking_summaries'])}")
        
        print(f"\n🧠 Key Thinking Insights:")
        for j, ts in enumerate(results['thinking_summaries'][:3], 1):  # Show top 3
            print(f"   {j}. {ts['step']} (confidence: {ts['confidence']:.2f})")
            if ts['key_insights']:
                print(f"      💡 {ts['key_insights'][0]}")
        
        print(f"\n📄 Final Report Preview:")
        preview = results['final_report'][:500] + "..." if len(results['final_report']) > 500 else results['final_report']
        print(f"   {preview}")
        
        # Save detailed results to file
        filename = f"research_results_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {filename}")


async def interactive_mode():
    """Interactive mode for custom research queries"""
    print("🔬 Deep Research Interactive Mode")
    print("Enter your research question (or 'quit' to exit):")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not found. Please set it first.")
        return
    
    researcher = DeepResearchGraph()
    
    while True:
        query = input("\nResearch Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
        
        # Ask for research depth
        try:
            depth = int(input("Research depth (1-5, default 3): ") or "3")
            depth = max(1, min(5, depth))
        except ValueError:
            depth = 3
        
        print(f"\n🚀 Starting research with depth {depth}...")
        
        # Run research
        results = await researcher.run_research(query, research_depth=depth)
        
        # Display results
        print(f"\n{'='*80}")
        print("📄 FINAL RESEARCH REPORT")
        print(f"{'='*80}")
        print(results['final_report'])
        
        print(f"\n{'='*80}")
        print("🧠 THINKING SUMMARIES")
        print(f"{'='*80}")
        for i, ts in enumerate(results['thinking_summaries'], 1):
            print(f"\n{i}. Step: {ts['step']}")
            print(f"   Confidence: {ts['confidence']:.2f}")
            print(f"   Reasoning: {ts['reasoning'][:200]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(main())