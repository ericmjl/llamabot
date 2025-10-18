# LlamaBot Notebooks

This directory contains Marimo notebooks that demonstrate various LlamaBot features and patterns.

## Available Notebooks

### `react-agentbot-demo.py`
**ReAct Pattern AgentBot Demo**

This notebook showcases the new ReAct (Reasoning and Acting) pattern in AgentBot. It demonstrates:

- **Explicit Reasoning**: How the agent shows its thinking process through "Thought:" messages
- **Tool Integration**: Using ToolBot for intelligent tool selection
- **Structured Observations**: Clear "Observation:" labels for tool results
- **Transparent Process**: Easy-to-follow decision-making flow
- **Performance Analytics**: Rich metadata about tool usage and execution

**Key Features Demonstrated:**
- Weather information queries
- Bill calculation with tips
- Multi-step complex tasks
- Error handling and recovery
- Tool creation best practices
- Performance monitoring

**How to Run:**
```bash
# Install marimo if you haven't already
pip install marimo

# Run the notebook
marimo run notebooks/react-agentbot-demo.py
```

**What You'll See:**
- Step-by-step ReAct cycles with visible reasoning
- Tool selection and execution process
- Structured message flow (Thought → Action → Observation → Answer)
- Execution analytics and performance metrics
- Best practices for building ReAct agents

This notebook is perfect for understanding how the new ReAct pattern works and how to build transparent, reasoning AI agents with LlamaBot.
