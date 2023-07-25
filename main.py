import asyncio
import os
from dotenv import load_dotenv
import discord
import replicate

load_dotenv()


class DiscordBot(discord.Client):
    def __init__(self, intents):
        super().__init__(intents=intents)
        self.replicate_model = os.environ['REPLICATE_MODEL']
        self.system_prompt = """You are a helpful, respectful and honest assistant.
                                Always answer as helpfully as possible, while being safe. Your answers should not include any harmful,
                                unethical, racist, sexist, toxic, dangerous, or illegal content. 
                                Please ensure that your responses are socially unbiased and positive in nature.
                                If a question does not make any sense, or is not factually coherent, explain why instead of answering
                                something not correct. If you don't know the answer to a question, please don't
                                share false information."""

    async def generate_user_prompt(self, channel, sender, limit=5):
        """Generate a formatted prompt. Only messages from the sender or bot will be included."""
        messages = []
        async for message in channel.history():
            if message.author == self.user:
                # self.user is the bot account, so we do use any specific format for these messages
                messages.append(message.content)
            elif message.author == sender:
                # Indicate the beginning ("[INST]") and end (`"/INST]") of user input
                messages.append(f"""[INST]{message.content}[/INST]""")
            if len(messages) >= limit:
                break

        return "\n".join(reversed(messages))

    async def generate_response(self, user_prompt):
        """Generate a response using the replicate model."""
        loop = asyncio.get_event_loop()

        def run_replicate():
            output = replicate.run(
                self.replicate_model,
                input={"prompt": user_prompt, "system_prompt": self.system_prompt})
            # Response comes chunked
            response_parts = []
            for item in output:
                response_parts.append(item)
            return " ".join(response_parts)

        # Run the model in a separate thread to avoid blocking code
        return await loop.run_in_executor(None, run_replicate)

    async def on_ready(self):
        """Called when the bot is ready and connected to Discord."""
        print(f'We have logged in as {self.user}')

    async def on_message(self, message):
        """Called when a message is sent to any channel the bot can see."""
        try:
            if message.author == self.user:
                return

            async with message.channel.typing():
                user_prompt = await self.generate_user_prompt(
                    channel=message.channel, sender=message.author)
                response = await self.generate_response(user_prompt)
                await message.channel.send(response)

        except Exception as exception:
            print(f"An error occurred: {exception}")


def bootstrap():
    intents = discord.Intents.default()
    intents.message_content = True
    bot = DiscordBot(intents=intents)
    bot.run(os.environ['DISCORD_API_TOKEN'])


bootstrap()
