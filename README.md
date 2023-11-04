# 🦙 Chat with Llama models on Discord

## Description

Discord bot that lets you chat with Llama models. Supports running the model both locally and through Replicate.

## Requirements

Python 3.11 is recommended. <br>
All library requirements are listed in [requirements.txt](requirements.txt).

## Setup and Installation

Clone the repository:

```bash
git clone https://github.com/davidgonmar/llama-discord-bot.git
cd llama-discord-bot
```

Create and activate a virtual environment (recommended):

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS and Linux
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download a compatible model binary or use one provided via Replicate API (see [Running the bot](#running-the-bot) section below).

Run the bot:

```bash
python main.py
```

## Running the bot

You will need a Discord developer account, and you will need to create a bot application. More information can be found at https://discord.com/developers/docs/intro.

Invite the bot to your Discord server using the OAuth2 URL with appropriate permissions. The bot will automatically respond to messages in any channel it can see. Set the `DISCORD_API_TOKEN` environment variable to the bot token.

For the LLM, you can run it locally or use Replicate to run it in the cloud:

- To run it locally, download a compatible model binary, and place it under a `models` folder in the root of the project. Make sure to set the `MODE` environment variable to `LOCAL`, and the `MODEL_PATH` environment variable to the path of the model binary relative to the root of the project.
- To run it in the cloud, you will need a Replicate API key. Set the `MODE` environment variable to `REPLICATE`, and the `REPLICATE_API_KEY` environment variable to your Replicate API key. You must also set the `REPLICATE_MODEL` environment variable to the model name and version you want to use. For example, you can use a 13B model with `a16z-infra/llama13b-v2-chat:6b4da803a2382c08868c5af10a523892f38e2de1aafb2ee55b020d9efef2fdb8`.

## Interacting with the bot

The bot will respond to any messages it can see. You'll be able to rewrite the responses by clicking 'Rewrite response' on the message. You can also continue a response by clicking 'Continue response' on the message.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full license text.
