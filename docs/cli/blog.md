# Blog Assistant CLI Tutorial

The Blog Assistant CLI is a powerful tool
that helps you streamline your blogging workflow.
It can generate blog summaries,
apply semantic line breaks (SEMBR),
and even create social media posts for LinkedIn, Patreon, and Twitter.
This tutorial will guide you through the usage of this tool.

## Summarize Command

The `summarize` command is used to generate a blog summary, title, and tags.
Here's how to use it:

1. Run the command `summarize` in your terminal: `llamabot blog summarize`
2. You will be prompted to paste your blog post.
3. The tool will then generate a blog title,
apply SEMBR to your summary,
and provide you with relevant tags.

The output will look something like this:

```text
Here is your blog title:
[Generated Blog Title]

Applying SEMBR to your summary...

Here is your blog summary:
[Generated Blog Summary with SEMBR]

Here are your blog tags:
[Generated Blog Tags]
```

## Social Media Command

The `social_media` command is used to generate social media posts.
Here's how to use it:

1. Run the command `social_media [platform]` in your terminal,
where `[platform]` is either `linkedin`, `patreon`, or `twitter`: `llamabot blog social-media linkedin`.
2. You will be prompted to paste your blog post.
3. The tool will then generate a social media post for the specified platform.

For LinkedIn and Twitter,
the generated post will be copied to your clipboard.
For Patreon,
the tool will display the post in the terminal.

## SEMBR Command

The `sembr` command is used to apply semantic line breaks to a blog post.
Here's how to use it:

1. Run the command `sembr` in your terminal: `llamabot blog sembr`
2. You will be prompted to paste your blog post.
3. The tool will then apply semantic line breaks to your post
and copy the result to your clipboard.

With these commands,
you can streamline your blogging workflow
and ensure your content is optimized for readability and engagement.
Happy blogging!
