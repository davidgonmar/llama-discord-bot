import os
from dotenv import load_dotenv
import discord
from llama import LlamaLocal, LlamaReplicate

load_dotenv()

class DiscordBot(discord.Client):
    def __init__(self, intents, local):
        super().__init__(intents=intents)
        if local:
            self.llama = LlamaLocal(model_path=os.path.abspath(os.environ['LOCAL_MODEL_PATH']))
        else:
            self.llama = LlamaReplicate(
                replicate_model=os.environ['REPLICATE_MODEL'],
                system_prompt="""You are a helpful, respectful and honest assistant.
                                Always answer as helpfully as possible, while being safe. Your answers should not include any harmful,
                                unethical, racist, sexist, toxic, dangerous, or illegal content. 
                                Please ensure that your responses are socially unbiased and positive in nature.
                                If a question does not make any sense, or is not factually coherent, explain why instead of answering
                                something not correct. If you don't know the answer to a question, please don't
                                share false information.""")
            

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
                resp = self.llama.generate_response(user_prompt=user_prompt)
                await message.channel.send(resp)

        except Exception as exception:
            print(f"An error occurred: {exception}")


def bootstrap():
    intents = discord.Intents.default()
    intents.message_content = True
    

    mode = os.environ['MODE'].lower()
    assert mode in {'local', 'replicate'}, f"Invalid mode: {mode}. Must be 'LOCAL' or 'REPLICATE'."

    bot = DiscordBot(intents=intents, local=mode == 'local')
    bot.run(os.environ['DISCORD_API_TOKEN'])


bootstrap()
