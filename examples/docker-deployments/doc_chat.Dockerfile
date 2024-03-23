# NOTE (Saturday, 23 March 2024)
# This doesn't work just yet. However, I'm still committing it just so I don't lose it,
# as I want to use this as an example of we can deploy something using the "baked data" pattern.
FROM condaforge/mambaforge:23.11.0-0

ARG SOURCEDIR=/tmp/llamabot
COPY llamabot ${SOURCEDIR}/llamabot
COPY pyproject.toml ${SOURCEDIR}
COPY data/dshiring.pdf /tmp/dshiring.pdf

ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh
ENV CONDA_PREFIX=/opt/conda/

COPY environment.yml /tmp/environment.yml
RUN mamba env update -f /tmp/environment.yml -n base
RUN pip install -e ${SOURCEDIR}
RUN apt-get update
RUN apt-get install --only-upgrade libstdc++6
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
RUN llamabot doc chat --help
EXPOSE 5006
ENTRYPOINT ["llamabot", "doc", "chat", "--model-name", "gpt-4-0125-preview", "--panel",  "--initial-message", "Hello there, I am a bot to help you with answering questions about this book, 'Hiring Data Scientists and Machine Learning Engineers'.", "/tmp/dshiring.pdf"]
