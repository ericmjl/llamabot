# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckduckgo-search==7.3.0",
#     "litellm",
#     "llamabot==0.10.12",
#     "marimo",
#     "pydantic==2.10.6",
# ]
# ///

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # What exactly makes an agent?

    Eric J. Ma

    PyData Boston/Cambridge, 29 Jan 2025
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This talk is going to leave us with more questions than answers about "what makes an agent".

    But I hope it also leaves us with more focused questions and frameworks for thinking about "what makes an agent".

    It is based loosely on [my blog post](https://ericmjl.github.io/blog/2025/1/4/what-makes-an-agent/). After this talk, I will turn this into a new blog post.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import llamabot as lmb

    mo.show_code()
    return lmb, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Let's start with LLM bots

    This is a helpful framing to anchor our discussion of agents.

    [LlamaBot](http://github.com/ericmjl/llamabot) implements a variety of LLM bots.

    LlamaBot is a Python package that I made to pedagogically explore the landscape of LLMs as the field evolves. Intentionally designed to lag 1-2 steps behind innovations and incorporate only what turns out to be robust.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The basic anatomy of an LLM API call:

    **1️⃣ A system prompt:** maintained over all calls.

    **2️⃣ A model name:** specifying which model to chat with.

    **3️⃣ A user's prompt:** your input into the LLM's internals.

    **4️⃣ The response:** returned by autoregressive next-token prediction.

    These are incorporated in `SimpleBot`:
    """
    )
    return


@app.cell(hide_code=True)
def _(lmb, mo):
    flight_bot = lmb.SimpleBot(
        system_prompt="""You are a helpful travel assistant.
        You provide information about flights between cities.""",
        model_name="ollama_chat/gemma2:2b",
    )

    # Simple interaction
    _ = flight_bot("Find me the prices for flights from San Francisco to Tokyo.")
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Let's evaluate that response together. Is this an agent?

    **Audience commentary:**

    - Not an agent: didn't take action apart from introspection of weights.
    - Not an agent: no external APIs called.
    - Yes: agent doesn't mean it has to have a tool, anything that gives usable information is an agent.
    - Yes (to an extent): Follows up with a question/prompt back to the human.
    - No: agent should make decisions. This LLM call did not make any decisions. Decision-making outsourced to user.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""`SimpleBot`s are heavily reliant on what's in the training data; they are just calling the LLM directly, with no memory of historical chat information. Given the range of regular human experiences in North America, I would venture to guess that it is rare for us to go outside of any base LLM's training data, unless we hit very specialized and niche topics not covered on the internet. (Try thinking of some!)"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Agents that give us structure?

    Let's try a different view of agents. What if we said, "Agents give you information structured the way you wanted it?" Let's start with that as a working definition.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""LlamaBot implements structured generation. To use it, you need to provide a Pydantic model, effectively a form that you want the bot to fill, and pair it with a `StructuredBot`. `StructuredBot`s have the same basic requirements as `SimpleBot`s, which are a system prompt and model name, but also need to have a Pydantic model passed in. When called, it will return an instance of that Pydantic model."""
    )
    return


@app.cell
def _(lmb, mo):
    from pydantic import BaseModel, Field

    class FlightInfo(BaseModel):
        origin: str
        destination: str
        duration: float = Field(..., description="Duration in hours")
        price: float = Field(..., description="Price in USD")
        recommended_arrival: float = Field(
            ...,
            description="Recommended time to arrive at airport before flight, in hours",
        )

    flight_bot_structured = lmb.StructuredBot(
        system_prompt="""You are a helpful travel assistant that provides
        structured information about flights.""",
        model_name="ollama_chat/llama3.2:3b",
        pydantic_model=FlightInfo,
    )

    # Now we get structured data
    result = flight_bot_structured(
        "Find me the prices for flights from San Francisco to Tokyo."
    )
    print()
    print("##########")
    print(
        f"The price of the flight is ${result.price}. We recommend that you arrive at the airport {result.recommended_arrival} hours before the flight."
    )
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Audience discussion: Is this an agent?

    Where does this fall short of an agent's definition?

    - No: still like RAG. Doesn't read the situation, is this a good time to buy? Still a simple answer.
    - No: it would independent, independently decide what to look at, where to look.
    - Yes: Based on the working definition, yes, and the nature of decision-making means the LLM is doing rapid-fire retrieval from memorized data. Recommended arrival time **is** an added recommendation that was not part of the original query intent.
    - No: Is the definition even correct? Why isn't a SQL query an agent, under that definition?
    - No: within this context, the **latent** query intent might be to buy tix, but this bot doesn't help me do it.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Structured generation gets us midway

    It affords us more **control** over what we are asking an LLM to generate. The response is constrained.

    But it seems like it may still lack a crucial element of an "agent". I'm glad the audience shred up this definition! 🤗
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## What if we said, "An agent needed to interact externally"?

    Here is my attempt, within LlamaBot, to try to nail down what an agent actually is. This flight agent-like bot helps me plan out trips, using one tool, the `search_internet` tool.
    """
    )
    return


@app.cell
def _(lmb, mo):
    from llamabot.bot.agentbot import search_internet

    travel_agent = lmb.AgentBot(
        system_prompt="""You are a travel planning assistant. Help users plan their trips
        by searching flights, hotels, and checking weather.""",
        functions=[search_internet],
        model_name="gpt-4o",
    )

    # Now we can handle complex queries
    travel_response = travel_agent(
        "What's the flight time from San Francisco to Tokyo?"
    )

    print("\n##########")
    print(travel_response.content)
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    If we got a straight answer, great!

    But if we didn't get a straight answer, that hints at some problems with agents, if they aren't designed properly.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Audience discussion: is this really an agent?

    I asked the audience to disregard the fact that I called this an `AgentBot`. Just because I said so doesn't make it so!

    Where are the holes in this definition of an agent?

    - Getting closer: agent will do research, in this example, it is the use of a search tool, and processing the text to give a response; the task could have been clearer for end user.
    - Yes: it has introduced a new error mode. A new way to be wrong, need someone to blame for this.
    - New component here: this bot has agency, decides "do I call this function or not".
    - Principal-Agent problem: agent does something that may not be what the Principal wanted it to do.

    More questions:

    - Q: Is part of the definition of an agent the fact that it is going to interact with a human?
    - Q: In this implementaiton of search, is the LLM always going to do the function call?
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Design patterns: should this be an agent?

    We discussed what an agent is. Now, let's assume that we know what an agent is. (This is an assumption!) If so, how should we design agents, and even then, should it be an agent?

    Our minimally complex example is a restaurant bill calculator. It's got a few key characteristics:

    1. Bill calculation is computable (and hence easily verifiable).
    2. There is sufficient variation in the kinds of questions we can ask.
    3. We can easily implement multiple designs.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### AgentBot implementation

    Let's start with an implementation based on AgentBot.

    We have two tools, which are nothing more than Python functions that are decorated with a `@tool` decorator. They are, namely, `calculate_total_with_tip` and `split_bill`.
    """
    )
    return


@app.cell
def _(lmb, mo):
    # Define the tools
    @lmb.tool
    def calculate_total_with_tip(bill_amount: float, tip_rate: float) -> float:
        if tip_rate < 0 or tip_rate > 1.0:
            raise ValueError("Tip rate must be between 0 and 1.0")
        return bill_amount * (1 + tip_rate)

    @lmb.tool
    def split_bill(total_amount: float, num_people: int) -> float:
        return total_amount / num_people

    mo.show_code()
    return calculate_total_with_tip, split_bill


@app.cell
def _(mo):
    mo.md(r"""Then, we create the AgentBot:""")
    return


@app.cell
def _(calculate_total_with_tip, lmb, mo, split_bill):
    # Create the bot
    bot = lmb.AgentBot(
        system_prompt=lmb.system(
            "You are my assistant with respect to restaurant bills."
        ),
        functions=[calculate_total_with_tip, split_bill],
        model_name="gpt-4o",
    )
    mo.show_code()
    return (bot,)


@app.cell
def _(mo):
    mo.md(r"""Now, let's try a few calculations:""")
    return


@app.cell
def _(bot, mo):
    # Calculate total with tip
    calculate_total_only_prompt = (
        "My dinner was $2300 without tips. Calculate my total with an 18% tip."
    )
    resp = bot(calculate_total_only_prompt)
    print(resp.content)
    mo.show_code()
    return


@app.cell
def _(bot, mo):
    # Split the bill
    split_bill_only_prompt = "My dinner was $2300 in total, I added an 18% gratuity, split the bill between 20 people."
    resp2 = bot(split_bill_only_prompt)
    print(resp2.content)
    mo.show_code()
    return (split_bill_only_prompt,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Couldn't it have been a Python function?

    Should this have been an agent?

    We very well could have done this instead:
    """
    )
    return


@app.cell
def _(mo):
    def calculate_bill(
        total_amount: float, tip_percentage: int, num_people: int
    ) -> float:
        return (total_amount * (100 + tip_percentage) / 100) / num_people

    calculate_bill(2300, 18, 20)

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Is there a way to make that Python function more flexible?

    But this is a very restrictive implementation. What if we didn't have any division to do?
    """
    )
    return


@app.cell
def _(mo):
    def calculate_bill_v2(
        total_amount: float, tip_percentage: int = 0, num_people: int = 1
    ) -> float:
        if tip_percentage:
            total_amount = total_amount * (100 + tip_percentage) / 100
        if num_people:
            return total_amount / num_people
        return total_amount

    calculate_bill_v2(2714, 0, 20)

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""Same functionality, but with more flexibility. The scope of inputs is more variable, in the form of almost anything you want with natural language. And this point, by the way, seems to distinguish between an LLM agent and a Python program."""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### But couldn't reasoning models do this?

    What if we did this instead?
    """
    )
    return


@app.cell
def _(lmb, mo):
    r1_bot = lmb.SimpleBot(
        "You are a smart bill calculator.", model_name="ollama_chat/deepseek-r1:latest"
    )
    r1_bot(
        "My dinner was $2300 in total, with 18% tip. Split the bill between 20 people. Respond with only the number."
    )

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What if we didn't have to provide any tools instead?

    Or what if, we asked the agent to write and execute its own code? (This follows HuggingFace's definition of an agent.)
    """
    )
    return


@app.cell
def _(lmb, mo, split_bill_only_prompt):
    from llamabot.bot.agentbot import write_and_execute_script

    # Create the bot
    autonomous_code_bot = lmb.AgentBot(
        system_prompt=lmb.system(
            "You are my assistant with respect to restaurant bills."
        ),
        functions=[write_and_execute_script],
        model_name="gpt-4o",
    )

    # Split the bill
    autonomous_code_bot(split_bill_only_prompt)
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Dissection

    If we think carefully about the distinction between a Python function, its stochastic variant, and an LLM agent, we might make the following observations:

    #### Functions

    1. Functions are written to accomplish a goal.
    1. Functions have an input signature, a body, and a return.
    2. Function inputs are constrained to the types that are accepted; they cannot be natural language.
    3. Function program flow is deterministic.

    #### Stochastic Functions

    1. Stochastic functions have non-deterministic flow control, resulting in a distribution of possible outputs.

    #### LLM Agents

    1. Are non-deterministic in flow control.
    2. Rely on structured outputs internally.
    3. Allow for natural language inputs.
    4. Nonetheless accomplish a goal.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What should be an agent?

    Anthropic has [guidance](https://www.anthropic.com/research/building-effective-agents).

    > When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. **This might mean not building agentic systems at all.** Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.
    >
    > When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale. For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough.

    In other words, can you build it with a regular Python program first? If so, maybe just start there.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    And from my own blog:

    > ## A roadmap for designing agent applications
    >
    > I've found that the most effective way to design agent applications is to progressively relax constraints on inputs and execution order.
    >
    > ### Start with a deterministic program
    >
    > - Design your application as you would with regular API calls
    > - Define clear input/output specifications
    > - Implement core functionality with standard programming patterns
    >
    > ### Relax input constraints
    >
    > - Accept natural language input and convert it to structured parameters for function calls
    > - Enable autonomous function calling based on natural language understanding
    >
    > ### Relax execution order constraints
    >
    > - Only necessary when natural language inputs are varied enough to require different execution paths
    > - Allow flexible ordering of operations when needed
    > - Enable dynamic selection of which functions to call
    > - Maintain boundaries around available actions while allowing flexibility in their use
    >
    > This progressive relaxation approach helps us transition from traditional deterministic programming to an agent-driven paradigm where execution order is non-deterministic and inputs are natural language.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## More perspectives

    - [Function calling is not solved](https://www.linkedin.com/posts/philipp-schmid-a6a2bb196_function-calling-is-not-solved-yet-a-new-activity-7288821311613591552-i5dH?utm_source=share&utm_medium=member_desktop)
    - [Google's definition of agents](https://drive.google.com/file/d/1oEjiRCTbd54aSdB_eEe3UShxLBWK9xkt/view)
    - [HuggingFace's definition of agents](https://x.com/AymericRoucher/status/1874116324898598934)
    - [My blog post](https://ericmjl.github.io/blog/2025/1/4/what-makes-an-agent/)
    """
    )
    return


if __name__ == "__main__":
    app.run()
