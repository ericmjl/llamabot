"""CLI for creating and keeping Markdown documentation up-to-date."""

try:
    import frontmatter
except ImportError:
    raise ImportError(
        "frontmatter is not installed. Please install it with `pip install llamabot[cli]`."
    )

from enum import Enum
from pathlib import Path

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyprojroot import here
from typer import Typer

import llamabot as lmb
from llamabot.prompt_manager import prompt
from llamabot.bot.simplebot import SimpleBot
from llamabot.components.messages import AIMessage

app = Typer()


class DiataxisType(Enum):
    """The type of diataxis to use for the documentation.

    Based on the Diátaxis framework (https://diataxis.fr/), which defines four modes of documentation:
    - Tutorials: Learning-oriented, practical steps for newcomers
    - How-to guides: Problem-oriented, practical steps for specific goals
    - Reference: Information-oriented, theoretical knowledge
    - Explanation: Understanding-oriented, background and context
    """

    TUTORIAL = "tutorial"  # Learning-oriented guides for beginners
    HOWTO = "howto"  # Task-oriented guides for specific problems
    REFERENCE = "reference"  # Technical reference documentation
    EXPLANATION = "explanation"  # Background and conceptual guides


diataxis_sources = {
    DiataxisType.HOWTO: "https://raw.githubusercontent.com/evildmp/diataxis-documentation-framework/refs/heads/main/how-to-guides.rst",
    DiataxisType.REFERENCE: "https://raw.githubusercontent.com/evildmp/diataxis-documentation-framework/refs/heads/main/reference.rst",
    DiataxisType.TUTORIAL: "https://raw.githubusercontent.com/evildmp/diataxis-documentation-framework/refs/heads/main/tutorials.rst",
    DiataxisType.EXPLANATION: "https://raw.githubusercontent.com/evildmp/diataxis-documentation-framework/refs/heads/main/explanation.rst",
}


class MarkdownSourceFile(BaseModel):
    """Content related to a Markdown source file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    file_path: Path = Field(..., description="Path to the Markdown source file.")
    post: frontmatter.Post = Field(..., description="The Markdown content.")
    diataxis_type: DiataxisType | None = Field(
        None, description="The type of diataxis to use for the documentation."
    )
    diataxis_source: str | None = Field(
        None, description="The source text for the diataxis documentation type."
    )

    linked_files: dict[str, str] = Field(
        {},
        description="Dictionary mapping linked files to their source texts. These must be relative to the repository root.",
    )
    raw_content: str = Field(
        "", description="The raw content of the Markdown source file."
    )

    def __init__(self, file_path: Path):
        # Initialize with basic file info first
        file_post = frontmatter.load(str(file_path))  # Convert Path to str
        super().__init__(file_path=file_path, post=file_post)

        # Handle linked files
        for fpath in self.post.get("linked_files", []):
            with open(here() / fpath, "r+") as f:
                self.linked_files[fpath] = f.read()

        # Handle diataxis type and source
        if self.post.get("diataxis_type"):
            self.diataxis_type = DiataxisType(self.post.get("diataxis_type"))
            if self.diataxis_type in diataxis_sources:
                response = requests.get(diataxis_sources[self.diataxis_type])
                if response.status_code == 200:
                    self.diataxis_source = response.text

        # Handle raw content
        with open(file_path, "r+") as f:
            self.raw_content = "".join(
                f"{i + 1}: {line}" for i, line in enumerate(f.readlines())
            )

    def save(self):
        """Save the Markdown source file with the updated content."""

        with open(self.file_path, "w") as f:
            f.write(frontmatter.dumps(self.post))


class DocumentationContent(BaseModel):
    """Content related to documentation."""

    content: str = Field(..., description="The documentation content.")

    @model_validator(mode="after")
    def check_content(self):
        """Validate the content field."""
        if self.content.startswith("```") or self.content.endswith("```"):
            raise ValueError(
                "Documentation content should not start or end with triple backticks."
            )
        return self


class ModelValidatorWrapper(BaseModel):
    """Base class for model validators that wrap the model with additional fields."""

    status: bool = Field(..., description="Status indicator.")
    reasons: list[str] = Field(
        default_factory=list, description="Reasons for the status."
    )

    @model_validator(mode="after")
    def validate_status_and_reasons(self):
        """Validate the status and reasons fields.

        This function checks if the status and reasons fields are in a valid state.
        Validity is defined by:

        - status == True and reasons are not empty
        - status == False and reasons are empty
        """
        # The only valid states are status == True and reasons are not empty,
        # or that status == False and reasons are empty.
        if self.status and not self.reasons:
            raise ValueError("If status is True, reasons must be provided.")
        elif not self.status and self.reasons:
            raise ValueError("If status is False, reasons be empty.")
        return self


class DocumentationOutOfSyncWithSource(ModelValidatorWrapper):
    """Status indicating whether the documentation is out of sync with the source code."""

    status: bool = Field(
        ...,
        description="Whether the documentation is out of sync with the source code.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Reasons why the documentation is out of sync with the source code. Be specific.",
    )


class SourceContainsContentNotCoveredInDocs(ModelValidatorWrapper):
    """Status indicating whether the source contains content not covered in the documentation."""

    status: bool = Field(
        ...,
        description="Whether the source contains content not covered in the documentation.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Reasons why the source contains content not covered in the documentation. Be specific.",
    )


class DocsContainFactuallyIncorrectMaterial(ModelValidatorWrapper):
    """Status indicating whether docs contain factually incorrect material that contradicts the source code."""

    status: bool = Field(
        ...,
        description="Whether the documentation contains factually incorrect material that contradicts the source code.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="The factually incorrect material that contradicts the source code. Specify the contradiction.",
    )


class DocsDoNotCoverIntendedMaterial(ModelValidatorWrapper):
    """Status indicating whether or not the documentation does not cover the intended material."""

    status: bool = Field(
        ...,
        description="Whether the intents of the documentation are not covered by the source. Return False if intents are adequately covered. Return True if some intents are not covered.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Reasons why the intents of the documentation are not covered by the source. Be specific, citing the intent that isn't covered.",
    )


class DocsOutOfDate(BaseModel):
    """Content related to documentation out of date."""

    source_not_covered: SourceContainsContentNotCoveredInDocs
    intents_not_covered: DocsDoNotCoverIntendedMaterial
    factually_inaccurate: DocsContainFactuallyIncorrectMaterial

    def __bool__(self):
        """Return True if any of the sub-models are True."""
        return (
            bool(self.source_not_covered)
            or bool(self.intents_not_covered)
            or bool(self.factually_inaccurate)
        )


@prompt(role="system")
def ood_checker_sysprompt() -> str:
    """You are an expert in documentation management.
    You will be provided information about a written documentation file,
    what the documentation is intended to convey,
    a list of source files that are related to the documentation,
    and their contents.
    """


@prompt(role="user")
def documentation_information(source_file: MarkdownSourceFile) -> str:
    """Here, I will provide you with contextual information to do your work.

    ## Referenced source files

    These are the source files to reference:

    {% for filename, content in source_file.linked_files.items() %}
    [[ {{ filename }} BEGINS ]]
    {{ content }}
    [[ {{ filename }} ENDS ]]
    {% endfor %}

    ## Documentation source file

    Here is the documentation in its current state, {{ source_file.file_path }}:

    [[ {{ source_file.file_path }} BEGINS ]]
    {% if source_file.post.content %}
    {{ source_file.post.content | safe }}
    {% else %}
    <The documentation is empty.>
    {% endif %}
    [[ {{ source_file.file_path }} ENDS ]]

    ## Intents about the documentation

    Here is the intent about the documentation:
    [[ INTENTS BEGINS ]]
    {% for intent in source_file.post.get("intents", []) %}- {{ intent }}
    {% endfor %}
    [[ INTENTS ENDS ]]

    {% if source_file.diataxis_type and source_file.diataxis_source %}
    ## Diataxis Framework Guide

    This documentation should follow the {{ source_file.diataxis_type.value }} style from the Diataxis framework.
    Here is the relevant guide:

    [[ DIATAXIS GUIDE BEGINS ]]
    {{ source_file.diataxis_source }}
    [[ DIATAXIS GUIDE ENDS ]]
    {% endif %}

    ## Instructions

    Now please write the docs as requested according to the intents and diataxis guide.
    Where relevant, leverage any commentary present in the source files.
    """


@prompt(role="user")
def ood_checker_information(ood_result: DocsOutOfDate) -> str:
    """Here is the out-of-date check result:

    [[ OOD CHECK RESULT BEGINS ]]
    {{ ood_result.model_dump_json() }}
    [[ OOD CHECK RESULT ENDS ]]
    """


@prompt(role="system")
def docwriter_sysprompt():
    """
    [[ Persona ]]
    You are an expert in documentation writing.

    [[ Context ]]
    You will be provided a documentation source,
    a list of what the documentation is intended to cover,
    and a list of linked source files.

    [[ Instructions ]]
    Based on the intended information that the documentation is supposed to cover
    and the content of linked source files,
    please edit (or create) the documentation,
    ensuring that the documentation matches the intents
    and has the correct content from the linked source files.
    """


def ood_checker_bot(model_name: str = "gpt-4o") -> lmb.StructuredBot:
    """Return a StructuredBot for the out-of-date checker."""
    return lmb.StructuredBot(
        system_prompt=ood_checker_sysprompt(),
        pydantic_model=DocsOutOfDate,
        model_name=model_name,
    )


def docwriter_bot(model_name: str = "gpt-4o") -> lmb.StructuredBot:
    """Return a StructuredBot for the documentation writer."""
    return lmb.StructuredBot(
        system_prompt=docwriter_sysprompt(),
        pydantic_model=DocumentationContent,
        model_name=model_name,
    )


@prompt(role="system")
def refine_bot_sysprompt():
    """
    You are an expert in documentation writing.
    You will be provided a documentation source that has been written
    using other sources as context.
    Those other sources are not going to be provided to you,
    only the documentation source that you are going to refine will be provided.
    Refine the documentation for clarity and logical flow.
    """


def refine_bot(model_name: str = "o1-preview") -> SimpleBot:
    """Return a SimpleBot for the documentation writer."""
    return lmb.SimpleBot(
        system_prompt=refine_bot_sysprompt(),
        model_name=model_name,
    )


@app.command()
def write(
    file_path: Path,
    from_scratch: bool = False,
    refine: bool = False,
    verbose: bool = False,
    ood_checker_model_name: str = "gpt-4o",
    docwriter_model_name: str = "gpt-4o",
    refiner_model_name: str = "o1-preview",
):
    """Write the documentation based on the given source file.

    The Markdown file should have frontmatter that looks like this:

    ```markdown
    ---
    intents:
    - Point 1 that the documentation should cover.
    - Point 2 that the documentation should cover.
    - ...
    linked_files:
    - path/to/relevant_file1.py
    - path/to/relevant_file2.toml
    - ...
    ---
    ```

    :param file_path: Path to the Markdown source file.
    :param from_scratch: Whether to start with a blank documentation.
    :param refine: Whether to refine the documentation.
    :param verbose: Whether to print the verbose output.
    :param ood_checker_model_name: The model name for the out-of-date checker.
    :param docwriter_model_name: The model name for the docwriter.
    :param refiner_model_name: The model name for the refiner.
    """
    src_file = MarkdownSourceFile(file_path)

    if from_scratch:
        src_file.post.content = ""

    ood_checker = ood_checker_bot(model_name=ood_checker_model_name)
    result: DocsOutOfDate = ood_checker(
        documentation_information(src_file), verbose=verbose
    )

    if (not src_file.post.content) or result:
        docwriter = docwriter_bot(model_name=docwriter_model_name)
        # print("-------")
        # print("OOD CHECKER INFORMATION:")
        # print(ood_checker_information(result))
        # print("-------")
        response: DocumentationContent = docwriter(
            lmb.user(
                documentation_information(src_file), ood_checker_information(result)
            ),
            verbose=verbose,
        )
        src_file.post.content = response.content

    if refine:
        refiner = refine_bot(model_name=refiner_model_name)
        response: AIMessage = refiner(src_file.post.content)
        src_file.post.content = response.content

    src_file.save()
