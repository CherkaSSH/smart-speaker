"""Discord bot: the gateway connection of discord.py, no webhook and no listening port.

It answers known users in direct messages, when mentioned in a server, and in the configured
`channels` (every message there). `!say text` and the `/say` slash command make the speaker say it
aloud. Strangers register or ask for access in a direct message (see chat.py); in a server they are
ignored. Needs the Message Content intent (enable it on the bot's page in the developer portal).
"""

from __future__ import annotations

import asyncio
import logging
import re

from .chat import MAX_VOICE_BYTES, MAX_VOICE_SECONDS, Chat, Service, TranscribeError, split_message

log = logging.getLogger("brain.discord")

MAX_TEXT = 1900  # Discord's limit is 2000 characters
READY_TIMEOUT = 15  # seconds a send waits for the gateway after a reconnect


class DiscordBot(Service):
    name = "discord"
    title = "дискорд"
    prefix = "!"

    def __init__(self, chat: Chat, cfg, scfg: dict):
        super().__init__(chat, cfg, scfg)
        self.channels: set[int] = {int(c) for c in scfg["channels"]}
        self.voice_notes = bool(scfg.get("voice_notes", True))
        self.client = None
        self.ready = asyncio.Event()

    def target(self, uid: int) -> str:
        return f"user:{uid}"

    @property
    def me_id(self) -> int:
        user = getattr(self.client, "user", None)
        return int(user.id) if user is not None else 0

    def make_client(self):
        """The discord.py client with its intents and the /say command (not connected yet)."""
        import discord
        from discord import app_commands
        discord.VoiceClient.warn_nacl = discord.VoiceClient.warn_dave = False  # voice is never used here

        bot = self
        intents = discord.Intents.default()  # direct and server messages
        intents.message_content = True  # privileged: read what is written in channels

        class Client(discord.Client):
            async def setup_hook(self) -> None:
                await self.tree.sync()

            async def on_ready(self) -> None:
                log.info("connected as %s", self.user)
                bot.ready.set()

            async def on_resumed(self) -> None:
                bot.ready.set()

            async def on_disconnect(self) -> None:
                bot.ready.clear()

            async def on_message(self, message) -> None:
                try:
                    await bot.on_message(message)
                except Exception:
                    log.exception("handling a message failed")

        client = Client(intents=intents)
        client.tree = app_commands.CommandTree(client)

        @client.tree.command(name="say", description="Make the speaker say something aloud")
        @app_commands.describe(text="What the speaker should say")
        async def say(interaction: discord.Interaction, text: str) -> None:
            try:
                await bot.on_say(interaction, text)
            except Exception:
                log.exception("handling /say failed")

        self.client = client
        return client

    # Receiving -----------------------------------------------------------
    async def on_message(self, message) -> None:
        author = message.author
        if getattr(author, "bot", False) or int(author.id) == self.me_id:
            return
        direct = message.guild is None
        mentioned = any(int(u.id) == self.me_id for u in message.mentions)
        in_channel = int(message.channel.id) in self.channels
        if not (direct or mentioned or in_channel):
            return
        text = re.sub(rf"<@!?{self.me_id}>", "", message.content or "").strip()
        uid, display = int(author.id), getattr(author, "display_name", None) or str(author.name)

        async def reply(out: str) -> None:
            await self._reply(message.channel, out)

        member = self.member(uid) is not None
        voice = self._voice_attachment(message)
        if voice is not None and not text:
            if not member or not self.voice_notes:  # never download a stranger's file
                return
            text = await self._transcribe(voice, reply)
        if not text:
            return
        if not member:
            if direct:  # registration and access requests happen in private, never in a server channel
                await self.chat.handle(self, uid, display, text, reply)
            else:
                log.debug("ignoring stranger id %s in channel %s", uid, message.channel.id)
            return
        async with message.channel.typing():
            await self.chat.handle(self, uid, display, text, reply)

    @staticmethod
    def _voice_attachment(message):
        """The audio of a voice message (Discord's mic button), or an audio file sent alone; else None."""
        flags = getattr(message, "flags", None)
        for a in getattr(message, "attachments", []) or []:
            if getattr(flags, "voice", False) or str(getattr(a, "content_type", "") or "").startswith("audio/"):
                return a
        return None

    async def _transcribe(self, attachment, reply) -> str:
        """The words of a voice message, echoed back like in Telegram; "" when there are none."""
        if (getattr(attachment, "duration", None) or 0) > MAX_VOICE_SECONDS or attachment.size > MAX_VOICE_BYTES:
            await reply("Слишком длинное голосовое. / That voice message is too long.")
            return ""
        import discord
        try:
            text = await self.chat.transcribe(await attachment.read())
        except (TranscribeError, discord.HTTPException) as e:
            log.warning("voice message failed: %s", e)
            await reply("Не получилось разобрать голосовое. / I couldn't make out that voice message.")
            return ""
        if not text:
            await reply("Ничего не разобрала. / I couldn't hear any words.")
            return ""
        log.info("voice message: %s", text)
        await reply(f"🎤 «{text}»")
        return text

    async def on_say(self, interaction, text: str) -> None:
        """The /say slash command."""
        uid = int(interaction.user.id)
        display = getattr(interaction.user, "display_name", None) or str(interaction.user.name)

        if self.member(uid) is None:
            await interaction.response.send_message(
                "Я тебя пока не знаю: напиши мне в личные сообщения, чтобы зарегистрироваться. / "
                "I don't know you yet: message me directly to register.", ephemeral=True)
            return
        out = await self.chat.say((self.member(uid)[0] or display), text)
        await interaction.response.send_message(out, allowed_mentions=self._no_mentions())

    # Sending -------------------------------------------------------------
    @staticmethod
    def _no_mentions():
        import discord
        return discord.AllowedMentions.none()  # a message must never ping @everyone

    async def _reply(self, channel, text: str) -> None:
        for piece in split_message(text, MAX_TEXT):
            await channel.send(piece, allowed_mentions=self._no_mentions())

    async def send(self, target: str, text: str) -> None:
        """`target`: «user:ID» (a direct message), «channel:ID», or a bare user id."""
        kind, _, ident = target.rpartition(":")
        if not ident.isdigit():
            raise ValueError(f"bad Discord target {target!r}: use user:ID or channel:ID")
        if self.client is None:
            raise RuntimeError("the bot is not connected")
        await asyncio.wait_for(self.ready.wait(), READY_TIMEOUT)
        if kind == "channel":
            channel = self.client.get_channel(int(ident)) or await self.client.fetch_channel(int(ident))
        else:
            channel = await self.client.fetch_user(int(ident))  # users are written to in their DMs
        await self._reply(channel, text)

    # Lifecycle -----------------------------------------------------------
    async def run(self) -> None:
        import discord
        token = self.token()
        if not token:
            log.error("no bot token: set $%s or put it in %s (see README)", self.scfg["token_env"],
                      self.scfg["token_file"])
            return
        client = self.make_client()
        try:
            await client.start(token)
        except discord.LoginFailure:
            log.error("the token is not accepted, giving up (fix it and restart the brain)")
        except discord.PrivilegedIntentsRequired:
            log.error("enable the Message Content intent on the bot's page in the Discord developer portal, "
                      "then restart the brain")
        finally:
            self.ready.clear()
            await client.close()

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
