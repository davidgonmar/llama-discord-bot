# Chat with Llama 2 on Discord

## Description

Discord bot using the new Meta LLama 2 model. Uses Replicate to run the model.

## Requirements

Python 3.x
discord.py (discord.py library for Discord bot functionality)
dotenv (for loading environment variables)
replicate (to run the model in the cloud)

## Setup and Installation

Clone the repository:

```bash
git clone https://github.com/davidgonmar/llama2-discord-bot.git
cd llama2-discord-bot
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

Create a .env file in the root directory with the following contents:

```bash
DISCORD_API_TOKEN=YOUR_DISCORD_BOT_TOKEN
REPLICATE_MODEL=YOUR_REPLICATE_MODEL_NAME
REPLICATE_API_TOKEN=REPLICATE_API_TOKEN
```

Run the bot:

```bash
python main.py
```

## Usage

You will need a Discord bot account and a Replicate account.
Invite the bot to your Discord server using the OAuth2 URL with appropriate permissions.
The bot will automatically respond to messages in any channel it can see.

The AI model is easily swapable by changing the REPLICATE_MODEL environment variable.
As of now, you can use the 13B model with `a16z-infra/llama13b-v2-chat:6b4da803a2382c08868c5af10a523892f38e2de1aafb2ee55b020d9efef2fdb8`
and the 70B model with `replicate/llama70b-v2-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full license text.
