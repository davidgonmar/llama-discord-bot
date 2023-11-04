import os
from dotenv import load_dotenv
from llama_discord_bot.discord_bot import DiscordBot


def bootstrap():
    """Bootstraps the bot."""

    load_dotenv()

    mode = os.environ["MODE"].lower()
    assert mode in {
        "local",
        "replicate",
    }, f"Invalid mode: {mode}. Must be 'LOCAL' or 'REPLICATE'."

    replicate_model = os.environ.get("REPLICATE_MODEL")
    local_model_path = os.environ.get("LOCAL_MODEL_PATH")
    discord_api_token = os.environ.get("DISCORD_API_TOKEN")

    DiscordBot(
        local=mode == "local",
        discord_api_token=discord_api_token,
        replicate_model=replicate_model,
        local_model_path=local_model_path,
    )


if __name__ == "__main__":
    bootstrap()
