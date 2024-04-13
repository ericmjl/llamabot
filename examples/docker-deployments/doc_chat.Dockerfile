# NOTE (Saturday, 23 March 2024)
# This doesn't work just yet. However, I'm still committing it just so I don't lose it,
# as I want to use this as an example of we can deploy something using the "baked data" pattern.
FROM python:3.10-slim

ARG OPENAI_API_KEY
ARG SOURCEDIR=/tmp/llamabot
RUN pip install panel
ENV OPENAI_API_KEY=OPENAI_API_KEY
ENV BOKEH_ALLOW_WS_ORIGIN="*"
ENV GIT_PYTHON_REFRESH=quiet
# COPY examples/docker-deployments/test_panel_app.py ${SOURCEDIR}/test_panel_app.py

# EXPOSE 5006
# CMD ["panel", "serve", "/tmp/llamabot/test_panel_app.py", "--address", "0.0.0.0", "--port", "5006"]
# CMD ["python", "/tmp/llamabot/test_panel_app.py"]


COPY llamabot ${SOURCEDIR}/llamabot
COPY pyproject.toml ${SOURCEDIR}
COPY data/dshiring.pdf /tmp/dshiring.pdf

ENV CONDA_PREFIX=/opt/conda/

# COPY environment.yml /tmp/environment.yml
# RUN mamba env update -f /tmp/environment.yml -n base
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN pip install -e ${SOURCEDIR}
RUN apt-get update
RUN apt-get install --only-upgrade libstdc++6 git
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
RUN llamabot doc chat --help
EXPOSE 6363
ENTRYPOINT ["llamabot", "doc", "chat", "--model-name", "gpt-4-0125-preview", "--panel",  "--initial-message", "Hello there, I am a bot to help you with answering questions about this book, 'Hiring Data Scientists and Machine Learning Engineers'.", "/tmp/dshiring.pdf"]
