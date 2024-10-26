# PromptRecorder

!!! note
    This tutorial was written by GPT4 and edited by a human.

The prompt recorder is a class named `PromptRecorder` that helps in recording prompts and responses. It works as a context manager, allowing you to record prompts and responses within a specific context. Here's a brief overview of how it works:

1. The `PromptRecorder` class is initialized with an empty list called `prompts` to store the prompts and responses.

2. When entering the context manager using the `with` statement, the `__enter__()` method is called, which sets the current instance of the `PromptRecorder` as the active recorder in the `prompt_recorder_var` context variable.

3. To log a prompt and response, the `log()` method is called with the prompt and response as arguments. This method appends the prompt and response as a dictionary to the `prompts` list.

4. The `autorecord()` function is provided to be called within every bot. It checks if there is an active `PromptRecorder` instance in the context and logs the prompt and response using the `log()` method.

5. When exiting the context manager, the `__exit__()` method is called, which resets the `prompt_recorder_var` context variable to `None` and prints a message indicating that the recording is complete.

6. The `PromptRecorder` class also provides methods to represent the recorded data in different formats, such as a string representation (`__repr__()`), an HTML representation (`_repr_html_()`), a pandas DataFrame representation (`dataframe()`), and a panel representation (`panel()`).

By using the `PromptRecorder` class as a context manager, you can easily record prompts and responses within a specific context and then analyze or display the recorded data in various formats.
