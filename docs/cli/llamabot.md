# LlamaBot Configuration Tutorial

In this tutorial, we will walk through the configuration process for LlamaBot,
a Python-based bot that uses the OpenAI API.
The configuration process involves setting up the API key
and selecting the default model for the bot.

## Setting up the API Key

The first step in configuring LlamaBot is to set up the API key.
This is done by invoking:

```bash
llamabot configure api-key
```

The user will be prompted to enter their OpenAI API key.
The key will be hidden as you type it, and you will be asked to confirm it.
Once confirmed, the key will be stored as an environment variable, `OPENAI_API_KEY`.

## Configuring the Default Model

The next step in the configuration process is to select the default model for LlamaBot.
This is done by invoking:

```bash
llamabot configure default-model
```

LlamaBot will first load the environment variables
from the `.env` file located at `llamabotrc_path`.
It then retrieves a list of available models from the OpenAI API,
filtering for those that include 'gpt' in their ID.
**For this reason, it is important to set your OpenAI API key before configuring the default model.**

The function then displays the list of available models and prompts you to select one.
As you type, the function will suggest completions based on the available models.
The last model in the list is provided as the default option.

Once you have entered a valid model ID,
the function stores it as an environment variable,
`DEFAULT_LANGUAGE_MODEL`.

## Conclusion

By following these steps,
you can easily configure LlamaBot to use your OpenAI API key
and your chosen default model.
Remember to keep your API key secure,
and to choose a model that best suits your needs.
Happy coding!
