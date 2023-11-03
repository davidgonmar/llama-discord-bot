import os
from dotenv import load_dotenv
import discord
from llama_discord_bot.llama import LlamaLocal, LlamaReplicate, Message
from llama_discord_bot.view import BotResponseView


class DiscordBot(discord.Client):

    SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant."""
    CONTINUE_RESPONSE_SUFFIX = """This is a conversation you were having. Please continue your response."""
    MESSAGES_AFTER_THIS_ONE = """There has already been messages after this one. You cannot continue the response."""

    def __init__(self, intents, local):
        super().__init__(intents=intents)
        if local:
            print("🖥️  Running model locally")
            self.llama = LlamaLocal(model_path=os.path.abspath(
                os.environ['LOCAL_MODEL_PATH']), system_prompt=self.SYSTEM_PROMPT)
        else:
            print("☁️  Running model through replicate")
            self.llama = LlamaReplicate(
                replicate_model=os.environ['REPLICATE_MODEL'],
                system_prompt=self.SYSTEM_PROMPT)

    async def get_channel_messages(self, channel, limit=5, skip=0, start_with_user=True) -> list[Message]:
        """
        Get the last `limit` messages from the channel, skipping `skip` messages.
        If `start_with_user` is true, it will skip messages until it finds a message from the user (not the bot)
        """
        messages: list[Message] = []
        async for message in channel.history():
            if skip > 0:
                skip -= 1
                continue
            if len(messages) >= limit:
                break
            content = message.content
            if message.author == self.user:
                messages.append(Message(user="bot", content=content))
            else:
                messages.append(Message(user="user", content=content))
        return list(reversed(messages))

    async def on_ready(self):
        print(f'Bot initialized as {self.user}')

    async def on_message(self, message: discord.Message):
        """Called when a message is sent to any channel the bot can see."""
        try:
            # Ignore messages from self
            if message.author == self.user:
                return

            response: discord.Message = None

            async def on_continue_response(interaction: discord.Interaction):
                """Called when the user clicks the 'Continue response' button. It will send a new response continuing the old one"""

                # Since the text generation will probably take longer than 3 seconds, we need to defer
                # the interaction response. If we don't, it'll fail
                await interaction.response.defer()
                messages = await self.get_channel_messages(channel=message.channel)

                # If the last message is not the same as the current message, do not continue response.
                # This might be the case if the user already sent a message after this one
                if (messages[-1].content != response.content):
                    await interaction.followup.send(embed=discord.Embed(title="Error", description=self.MESSAGES_AFTER_THIS_ONE, color=discord.Color.red()), ephemeral=True)
                    return

                llama_response = await self.llama.generate_response(messages=messages, suffix=self.CONTINUE_RESPONSE_SUFFIX)
                await interaction.followup.send(content=llama_response)

            async def on_rewrite_response(interaction: discord.Interaction):
                """Called when the user clicks the 'Rewrite response' button. It will rewrite the response and edit the original message"""

                # Since the text generation will probably take longer than 3 seconds, we need to defer
                # the interaction response. If we don't, it'll fail
                await interaction.response.defer()
                messages = await self.get_channel_messages(channel=message.channel, skip=1)
                llama_response = await self.llama.generate_response(messages=messages)
                await interaction.message.edit(content=llama_response)

            view = BotResponseView(
                on_continue_response=on_continue_response, on_rewrite_response=on_rewrite_response)

            async with message.channel.typing():
                messages = await self.get_channel_messages(channel=message.channel)
                llama_response = await self.llama.generate_response(messages=messages)
                response = await message.channel.send(content=llama_response, view=view)

        except Exception as exception:
            print(
                f"An error occurred: {exception.__class__.__name__}: {exception}")


def bootstrap():
    load_dotenv()

    intents = discord.Intents.default()
    intents.message_content = True

    mode = os.environ['MODE'].lower()
    assert mode in {
        'local', 'replicate'}, f"Invalid mode: {mode}. Must be 'LOCAL' or 'REPLICATE'."

    bot = DiscordBot(intents=intents, local=mode == 'local')
    bot.run(os.environ['DISCORD_API_TOKEN'])


bootstrap()
