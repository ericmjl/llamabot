"""CLI for creating and keeping Markdown documentation up-to-date."""

from typer import Typer
from pathlib import Path
import frontmatter
from pydantic import BaseModel, Field, model_validator
from llamabot import prompt, StructuredBot
from pyprojroot import here

from pydantic import ConfigDict

from llamabot.bot.simplebot import SimpleBot


app = Typer()


class MarkdownSourceFile(BaseModel):
    """Content related to a Markdown source file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    file_path: Path = Field(..., description="Path to the Markdown source file.")
    post: frontmatter.Post = Field(..., description="The Markdown content.")

    linked_files: dict[str, str] = Field(
        {},
        description="Dictionary mapping linked files to their source texts. These must be relative to the repository root.",
    )
    raw_content: str = Field(
        "", description="The raw content of the Markdown source file."
    )

    def __init__(self, file_path: Path):
        super().__init__(file_path=file_path, post=frontmatter.load(file_path))

        for fpath in self.post.get("linked_files", []):
            with open(here() / fpath, "r+") as f:
                self.linked_files[fpath] = f.read()

        # self.post.content = "\n".join(f"{i+1}: {line}" for i, line in enumerate(self.post.content.splitlines()))
        with open(file_path, "r+") as f:
            # Read in the file contents with line numbers added in.
            self.raw_content = "".join(
                f"{i+1}: {line}" for i, line in enumerate(f.readlines())
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


@prompt
def ood_checker_sysprompt() -> str:
    """You are an expert in documentation management.
    You will be provided information about a written documentation file,
    what the documentation is intended to convey,
    a list of source files that are related to the documentation,
    and their contents.
    """


@prompt
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
    """


@prompt
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


def ood_checker_bot(model_name: str = "gpt-4o") -> StructuredBot:
    """Return a StructuredBot for the out-of-date checker."""
    return StructuredBot(
        system_prompt=ood_checker_sysprompt(),
        pydantic_model=DocsOutOfDate,
        model_name=model_name,
    )


def docwriter_bot(model_name: str = "gpt-4o") -> StructuredBot:
    """Return a StructuredBot for the documentation writer."""
    return StructuredBot(
        system_prompt=docwriter_sysprompt(),
        pydantic_model=DocumentationContent,
        model_name=model_name,
    )


@prompt
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
    return SimpleBot(
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

    if not src_file.post.content or result:
        docwriter = docwriter_bot(model_name=docwriter_model_name)
        response: DocumentationContent = docwriter(
            documentation_information(src_file) + "\nNow please write the docs.",
            verbose=verbose,
        )
        src_file.post.content = response.content

    if refine:
        refiner = refine_bot(model_name=refiner_model_name)
        response: str = refiner(src_file.post.content, verbose=verbose)
        src_file.post.content = response

    src_file.save()
