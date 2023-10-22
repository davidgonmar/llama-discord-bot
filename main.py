import os
from dotenv import load_dotenv
import discord
from llama import LlamaLocal, LlamaReplicate, Message
from view import BotResponseView


load_dotenv()

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant."""

class DiscordBot(discord.Client):
    def __init__(self, intents, local):
        super().__init__(intents=intents)
        if local:
            print("🖥️  Running model locally")
            self.llama = LlamaLocal(model_path=os.path.abspath(os.environ['LOCAL_MODEL_PATH']), system_prompt=SYSTEM_PROMPT)
        else:
            print("☁️  Running model through replicate")
            self.llama = LlamaReplicate(
                replicate_model=os.environ['REPLICATE_MODEL'],
                system_prompt=SYSTEM_PROMPT)
            
    
    async def get_channel_messages(self, channel, limit=5, skip = 0) -> list[Message]:
        """Get the last `limit` messages from the channel."""
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
        """Called when the bot is ready and connected to Discord."""
        print(f'We have logged in as {self.user}')


    
    async def on_message(self, message: discord.Message):
        """Called when a message is sent to any channel the bot can see."""
        try:

            # If message is from the bot itself, ignore it
            if message.author == self.user:
                return
            
            response = None
            async def on_continue_response(interaction: discord.Interaction):
                # modify interaction button to disable it
                await interaction.response.defer()
                messages = await self.get_channel_messages(channel=message.channel)
                # if the last message is not the same as the current message, do not continue
                if (messages[-1].content != response.content):
                    await interaction.followup.send(embed=discord.Embed(title="Error", description="There has already been messages after this one. You cannot continue the response.", color=discord.Color.red()), ephemeral=True)
                    return
                resp = await self.llama.generate_response(messages=messages, suffix="[INST] Continue response [/INST]")
                await interaction.followup.send(content=resp)

            async def on_rewrite_response(interaction: discord.Interaction):
                await interaction.response.defer()
                messages = await self.get_channel_messages(channel=message.channel, skip=1)
                resp = await self.llama.generate_response(messages=messages)
                await interaction.message.edit(content=resp)

            view = BotResponseView(on_continue_response=on_continue_response, on_rewrite_response=on_rewrite_response)

            async with message.channel.typing():
                messages = await self.get_channel_messages(channel=message.channel)
                resp = await self.llama.generate_response(messages=messages)
                response = await message.channel.send(content=resp, view=view)
            
            
        except Exception as exception:
            print(f"An error occurred: {exception.__class__.__name__}: {exception}")



def bootstrap():
    intents = discord.Intents.default()
    intents.message_content = True
    

    mode = os.environ['MODE'].lower()
    assert mode in {'local', 'replicate'}, f"Invalid mode: {mode}. Must be 'LOCAL' or 'REPLICATE'."

    bot = DiscordBot(intents=intents, local=mode == 'local')
    bot.run(os.environ['DISCORD_API_TOKEN'])

bootstrap()
