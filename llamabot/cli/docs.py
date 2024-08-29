"""CLI for creating and keeping Markdown documentation up-to-date."""

from typer import Typer
from pathlib import Path
import frontmatter
from pydantic import BaseModel, Field
from llamabot import prompt, StructuredBot, SimpleBot
from pyprojroot import here

from pydantic import ConfigDict

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


class DocumentationOutOfDate(BaseModel):
    """Status indicating whether a documentation is out of date."""

    is_out_of_date: bool = Field(
        ..., description="Whether the documentation is out of date."
    )


@prompt
def ood_checker_sysprompt():
    """You are an expert in documentation management.
    You will be provided information about a written documentation file,
    what the documentation is intended to convey,
    a list of source files that are related to the documentation,
    and their contents.
    """


@prompt
def documentation_information(source_file: MarkdownSourceFile) -> str:
    """## Referenced source files

    These are the source files to reference:

    {% for filename, content in source_file.linked_files.items() %}
    -----
    [ {{ filename }} ]

    {{ content }}
    -----
    {% endfor %}

    ## Documentation source file

    Here is the documentation source file, {{ source_file.file_path }}:

    -----
    {% if source_file.post.content %}
    {{ source_file.post.content | safe }}
    {% else %}
    <The documentation is empty.>
    {% endif %}
    -----

    ## Intents about the documentation

    Here is the intent about the documentation:

    {% for intent in source_file.post.get("intents", []) %}- {{ intent }}{% endfor %}
    """


@prompt
def docwriter_sysprompt():
    """You are an expert in documentation writing.

    You will be provided a documentation source,
    a list of what the documentation is intended to cover,
    and a list of linked source files.

    Based on the intended messages information that the documentation is supposed to cover
    and the content of linked source files,
    please edit or create the documentation,
    ensuring that the documentation matches the intents
    and has the correct content from the linked source files.
    """


@app.command()
def write(file_path: Path, force: bool = False):
    """Write the documentation based on the given source file.

    :param file_path: Path to the Markdown source file.
    :param force: Force writing the documentation even if it is not out of date.
    """
    src_file = MarkdownSourceFile(file_path)

    if not src_file.post.content:  # i.e. the documentation is empty
        docwriter = SimpleBot(system_prompt=docwriter_sysprompt())
        response = docwriter(documentation_information(src_file))
        src_file.post.content = response.content

    else:
        ood_checker = StructuredBot(
            system_prompt=ood_checker_sysprompt(), pydantic_model=DocumentationOutOfDate
        )
        result = ood_checker(documentation_information(src_file))
        if result.is_out_of_date or force:
            docwriter = SimpleBot(system_prompt=docwriter_sysprompt())
            response = docwriter(documentation_information(src_file))
            src_file.post.content = response.content
    src_file.save()
