# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "marimo",
#     "llamabot",
#     "matplotlib",
#     "networkx",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../../", editable = true }
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Extract knowledge graph triples from text

    In this notebook, we will show how to use LlamaBot's StructuredBot
    to extract knowledge graph triples from text.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Firstly, let's start by defining a class `KnowledgeGraphTriple` to represent a triple from a knowledge graph.
    """
    )
    return


@app.cell
def _():
    from llamabot import StructuredBot
    from pydantic import BaseModel, Field, model_validator

    class KnowledgeGraphTriple(BaseModel):
        """A triple from a knowledge graph."""

        sub: str = Field(description="The subject of the triple.")
        pred: str = Field(description="The predicate of the triple.")
        obj: str = Field(description="The object of the triple.")
        quote: str = Field(
            description="Quote from the provided text that indicates the triple relationship."
        )

        @model_validator(mode="after")
        def validate_lengths(self):
            """Validate the lengths of subjects, predicates, and objects.

            They should not be >5 words each.
            """
            if len(self.sub.split()) > 5:
                raise ValueError(
                    f"Subject '{self.sub}' is too long. It needs to be <=5 words."
                )

            if len(self.pred.split()) > 5:
                raise ValueError(
                    f"Predicate '{self.pred}' is too long. It needs to be <=5 words."
                )

            if len(self.obj.split()) > 5:
                raise ValueError(
                    f"Object '{self.obj}' is too long. It needs to be <=5 words."
                )

            return self

    return BaseModel, Field, KnowledgeGraphTriple, StructuredBot


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Within the KnowledgeGraphTriple class,
    we've added a `validate_lengths` method
    to ensure that the lengths of the subject, predicate, and object
    are not greater than 5 words.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Next, I am going to define a `KnowledgeGraphTriplets` class
    that houses a collection of KnowledgeGraphTriple objects.
    This class will allow me to instruct the StructuredBot
    to extract _multiple_ triples from a given text.
    """
    )
    return


@app.cell
def _(BaseModel, Field, KnowledgeGraphTriple):
    class KnowledgeGraphTriplets(BaseModel):
        """A list of knowledge graph triples."""

        triples: list[KnowledgeGraphTriple] = Field(description="A list of triples.")

        def draw(self):
            import matplotlib.pyplot as plt
            import networkx as nx

            G = nx.DiGraph()
            for triple in self.triples:
                G.add_edge(triple.sub, triple.obj, label=triple.pred)

            pos = nx.spring_layout(G, seed=42)
            fig, ax = plt.subplots(figsize=(10, 8))
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_color="lightblue", node_size=1500
            )
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            nx.draw_networkx_edges(
                G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=20
            )
            elabels = nx.get_edge_attributes(G, "label")
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=elabels, ax=ax, font_size=6
            )
            ax.axis("off")
            fig.tight_layout()
            return fig

    return (KnowledgeGraphTriplets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, let's set up the StructuredBot, which we will call `kgbot`,
    which stands for "Knowledge Graph Bot".
    We will use the Groq-hosted `llama-3.1-70b-versatile`,
    which is a powerful open-source language model developed by Meta AI
    and hosted on Groq for lightning fast text generation.
    """
    )
    return


@app.cell
def _(KnowledgeGraphTriplets, StructuredBot):
    from llamabot import prompt

    @prompt
    def kgbot_sysprompt() -> str:
        """You are an expert at knowledge graph triples.

        You will be given a paragraph of text and you need to extract triples from it.
        Each triple should be presented as a structured object.
        """

    kgbot = StructuredBot(
        system_prompt=kgbot_sysprompt(),
        pydantic_model=KnowledgeGraphTriplets,
        stream_target="none",
        model_name="groq/llama-3.1-70b-versatile",
    )
    return kgbot, prompt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The LLM of choice here is GPT-4.
    We could choose to use `ollama/llama3`,
    which is a powerful open source language model developed by Meta AI.
    It needs just over 16GB of RAM to run.
    It's going to be much slower than calling on the GPT-4 API.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    With the bot instantiated, let's now analyze a paragraph of text about the citric acid cycle.
    """
    )
    return


@app.cell
def _(kgbot):
    txt = """
    The citric acid cycle—also known as the Krebs cycle, Szent–Györgyi–Krebs cycle or the TCA cycle (tricarboxylic acid cycle)[1][2]—is a series of biochemical reactions to release the energy stored in nutrients through the oxidation of acetyl-CoA derived from carbohydrates, fats, and proteins. The chemical energy released is available under the form of ATP. The Krebs cycle is used by organisms that respire (as opposed to organisms that ferment) to generate energy, either by anaerobic respiration or aerobic respiration. In addition, the cycle provides precursors of certain amino acids, as well as the reducing agent NADH, that are used in numerous other reactions. Its central importance to many biochemical pathways suggests that it was one of the earliest components of metabolism.[3][4] Even though it is branded as a "cycle", it is not necessary for metabolites to follow only one specific route; at least three alternative segments of the citric acid cycle have been recognized.[5]
    """

    triples_set = kgbot(txt, verbose=True)
    return (triples_set,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's visualize the triples extracted from the given text.
    """
    )
    return


@app.cell
def _(triples_set):
    triples_set.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Effectively we have gained an auto-generated mind map for ourselves!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Let's try another text, one that is on pair programming, from Martin Fowler's website.
    """
    )
    return


@app.cell
def _(kgbot):
    _text = "Pair programming essentially means that two people write code together on one machine. It is a very collaborative way of working and involves a lot of communication. While a pair of developers work on a task together, they do not only write code, they also plan and discuss their work. They clarify ideas on the way, discuss approaches and come to better solutions."
    pair_programming_knowledge = kgbot(_text)
    return (pair_programming_knowledge,)


@app.cell
def _(pair_programming_knowledge):
    pair_programming_knowledge.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Applying the bot to more text:
    """
    )
    return


@app.cell
def _(kgbot):
    _text = "\nRenewable energy sources are becoming increasingly vital as the world grapples with the adverse effects of climate change and the depletion of fossil fuels. Unlike traditional energy sources such as coal, oil, and natural gas, renewable energy comes from resources that are naturally replenished on a human timescale, including sunlight, wind, rain, tides, waves, and geothermal heat.\n\nOne of the most prominent forms of renewable energy is solar power. Solar panels convert sunlight directly into electricity through photovoltaic cells. This technology has seen significant advancements over the past decade, leading to increased efficiency and reduced costs. Solar energy is abundant and can be harnessed in virtually every part of the world, making it a cornerstone of future energy strategies.\n\nWind energy is another major player in the renewable energy sector. Wind turbines capture kinetic energy from wind and convert it into electrical power. Wind farms, both onshore and offshore, have been developed across the globe. The scalability of wind power, from small residential turbines to large commercial farms, provides versatility in its applications.\n\nHydropower, the largest source of renewable electricity globally, generates power by using the energy of moving water. Dams built on large rivers create reservoirs that can be used to generate electricity on demand. Small-scale hydro projects also contribute significantly to local energy needs, especially in remote areas.\n\nGeothermal energy harnesses heat from within the Earth. This heat can be used directly for heating or to generate electricity. Regions with high geothermal activity, such as Iceland and parts of the United States, have successfully integrated geothermal energy into their power grids.\n\nBioenergy, derived from organic materials such as plant and animal waste, is a versatile renewable energy source. It can be used for electricity generation, heating, and as a biofuel for transportation. The use of bioenergy can also help manage waste and reduce greenhouse gas emissions.\n\nThe transition to renewable energy sources is not without challenges. The intermittency of solar and wind power requires advancements in energy storage technologies to ensure a stable supply. Moreover, the initial investment costs for renewable energy infrastructure can be high, although these are often offset by long-term savings and environmental benefits.\n\nGovernment policies and international cooperation play crucial roles in promoting renewable energy. Subsidies, tax incentives, and research funding are essential to accelerate the development and adoption of these technologies. Public awareness and community involvement also contribute to the successful implementation of renewable energy projects.\n\nIn conclusion, renewable energy represents a sustainable and essential path forward for global energy needs. By reducing reliance on fossil fuels, we can mitigate the impacts of climate change, enhance energy security, and create a cleaner, healthier environment for future generations.\n"
    energy_kg = kgbot(_text)
    return (energy_kg,)


@app.cell
def _(energy_kg):
    energy_kg.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Finally, let's look at inductive Bible study, a technique for studying Biblical texts
    using inductive reasoning and analytic thinking.

    We will create a different bot (`ibsbot`, which stands for "Inductive Bible Study Bot"),
    to help us parse and extract structured information from the text.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Firstly, let's define the pydantic models for what we want to extract from scripture.
    """
    )
    return


@app.cell
def _(BaseModel, Field):
    class IBSLogicalRelationship(BaseModel):
        statement1: str = Field(description="First statement made by the author.")
        joiner: str = Field(
            description="Joiner between statement1 and statement2. Should be a logical conjunction, such as 'but', 'so that', 'in order that', 'for', 'because', 'and', 'if'."
        )
        statement2: str = Field(
            description="Second statement logically related to statement1."
        )

    class IBSRelationships(BaseModel):
        """A list of inductive Bible study relationships."""

        relationships: list[IBSLogicalRelationship] = Field(
            description="A list of relationships."
        )

        def draw(self):
            """Draw a diagram of the relationships."""
            import matplotlib.pyplot as plt
            import networkx as nx

            G = nx.DiGraph()
            for triple in self.relationships:
                G.add_edge(triple.statement1, triple.statement2, label=triple.joiner)

            pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
            labels = {n: (n[:30] + "…") if len(n) > 30 else n for n in G.nodes()}
            fig, ax = plt.subplots(figsize=(14, 10))
            nx.draw_networkx_nodes(
                G, pos, ax=ax, node_color="lightyellow", node_size=2000
            )
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
            nx.draw_networkx_edges(
                G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=15
            )
            elabels = nx.get_edge_attributes(G, "label")
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=elabels, ax=ax, font_size=6
            )
            ax.axis("off")
            fig.tight_layout()
            return fig

    return (IBSRelationships,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, let's create the system prompt for the IBSBot.
    """
    )
    return


@app.cell
def _(prompt):
    @prompt
    def ibsbot_sysprompt() -> str:
        """You are an expert at inductive Bible study.

        Your goal is to extract, from a given text,
        logical relationships between the author's statements.

        Segment the text into atomic segments that each contain a single idea
        that can be related to each other through a conjunction.

        Then, populate the provided pydantic model with the extracted relationships.
        Reuse as many atomic segments as possible.
        """

    return (ibsbot_sysprompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Finally, we create the `ibsbot`.
    """
    )
    return


@app.cell
def _(IBSRelationships, StructuredBot, ibsbot_sysprompt):
    ibsbot = StructuredBot(
        system_prompt=ibsbot_sysprompt(),
        pydantic_model=IBSRelationships,
        stream_target="none",
        model_name="groq/llama-3.1-70b-versatile",
    )
    return (ibsbot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, let's extract and visualize relationships from the text, Ephesians 2:1-10.
    """
    )
    return


@app.cell
def _(ibsbot):
    ephesians_text = """2 As for you, you were dead in your transgressions and sins, 2 in which you used to live when you followed the ways of this world and of the ruler of the kingdom of the air, the spirit who is now at work in those who are disobedient. 3 All of us also lived among them at one time, gratifying the cravings of our flesh[a] and following its desires and thoughts. Like the rest, we were by nature deserving of wrath. 4 But because of his great love for us, God, who is rich in mercy, 5 made us alive with Christ even when we were dead in transgressions—it is by grace you have been saved. 6 And God raised us up with Christ and seated us with him in the heavenly realms in Christ Jesus, 7 in order that in the coming ages he might show the incomparable riches of his grace, expressed in his kindness to us in Christ Jesus. 8 For it is by grace you have been saved, through faith—and this is not from yourselves, it is the gift of God— 9 not by works, so that no one can boast. 10 For we are God’s handiwork, created in Christ Jesus to do good works, which God prepared in advance for us to do."""
    ephesians_kg = ibsbot(ephesians_text)
    return (ephesians_kg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    What have we extracted? Let's visualize that.
    """
    )
    return


@app.cell
def _(ephesians_kg):
    ephesians_kg.draw()
    return


if __name__ == "__main__":
    app.run()
