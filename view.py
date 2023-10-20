

from typing import Callable, Literal
import discord

TCallback = Callable[[discord.Interaction], None]

class BotResponseView(discord.ui.View):
    # Button labels and styles
    BUTTONS = {
        "continue_response": {
            "label": "Rewrite response",
            "loading_label": "Rewriting response...",
            "style": discord.ButtonStyle.primary,
        },
        "rewrite_response": {
            "label": "Continue response",
            "loading_label": "Continuing response...",
            "style": discord.ButtonStyle.primary,
        }
    }

    def __init__(self, on_continue_response: TCallback, on_rewrite_response: TCallback):
        super().__init__()
        self._on_continue_response = on_continue_response
        self._on_rewrite_response = on_rewrite_response


    async def _transition_button_state(self, interaction: discord.Interaction, button: discord.ui.Button, state: Literal["default", "loading"]) -> None:
        """Transition the button state to either default or loading. Also updates the view so changes are reflected"""
        if state == "default":
            button.disabled = False
            button.label = self.BUTTONS[button.custom_id]["label"]
        elif state == "loading":
            button.disabled = True
            button.label = self.BUTTONS[button.custom_id]["loading_label"]

        # This is necessary to 'refresh' the view so the button state changes are reflected
        await interaction.message.edit(view=self)


    @discord.ui.button(label=BUTTONS["continue_response"]["label"], style=BUTTONS["continue_response"]["style"], custom_id="continue_response")
    async def continue_response(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._transition_button_state(interaction=interaction, button=button, state="loading")
        await self._on_continue_response(interaction)
        await self._transition_button_state(interaction=interaction, button=button, state="default")

    @discord.ui.button(label=BUTTONS["rewrite_response"]["label"], style=BUTTONS["rewrite_response"]["style"], custom_id="rewrite_response")
    async def rewrite_response(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._transition_button_state(interaction=interaction, button=button, state="loading")
        await self._on_rewrite_response(interaction)
        await self._transition_button_state(interaction=interaction, button=button, state="default")


