"""Telegram bot: long polling of the Bot API, private chats only.

No webhook and no listening port: the brain asks Telegram for updates (`getUpdates`), so nothing on
the internet can reach it. Only known users are answered; strangers register or ask (see chat.py). Voice notes are
decoded with ffmpeg to 16 kHz mono and transcribed by the running ASR server's offline endpoint
(`POST /v1/audio/transcriptions`), then handled like typed text.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from urllib.parse import urlsplit

import aiohttp

from .chat import Chat, Service, split_message

log = logging.getLogger("brain.telegram")

POLL_SECONDS = 30  # long-poll wait of one getUpdates
STALE_SECONDS = 300  # messages older than this (sent while the brain was down) are ignored, not run late
MAX_TEXT = 4000  # Telegram's limit is 4096 characters
MAX_VOICE_SECONDS = 180
MAX_VOICE_BYTES = 5_000_000


class TelegramError(Exception):
    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal  # the token is wrong: retrying is pointless


class TelegramBot(Service):
    name = "telegram"
    title = "телеграм"

    def __init__(self, chat: Chat, cfg, scfg: dict):
        super().__init__(chat, cfg, scfg)
        self.api_url = scfg["api_url"].rstrip("/")
        self.voice_notes = bool(scfg["voice_notes"])
        self._token = ""
        self._send_lock = asyncio.Lock()
        self._http: aiohttp.ClientSession | None = None
        asr_url = urlsplit(cfg["asr"]["url"])  # ws://host:port/v1/... -> http://host:port/v1/audio/transcriptions
        self.asr_url = f"{'https' if asr_url.scheme == 'wss' else 'http'}://{asr_url.netloc}/v1/audio/transcriptions"
        self.asr_language = cfg["asr"]["language"]

    # Bot API -------------------------------------------------------------
    def _clean(self, e: BaseException) -> str:
        """An error for the log, without the token that is part of every URL."""
        return f"{type(e).__name__}: {e}".replace(self._token, "***") if self._token else f"{type(e).__name__}"

    async def _call(self, method: str, params: dict | None = None, *, timeout: float = 20, tries: int = 3):
        """One Bot API call, retried on network errors, 5xx and rate limits; returns `result`."""
        if self._http is None:
            raise TelegramError("the bot is not connected")
        url = f"{self.api_url}/bot{self._token}/{method}"
        delay, last = 1.0, ""
        for attempt in range(tries):
            if attempt:
                await asyncio.sleep(delay)
                delay *= 2
            try:
                async with self._http.post(url, json=params or {}, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                    data = await r.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
                last = self._clean(e)
                continue
            if data.get("ok"):
                return data["result"]
            code = int(data.get("error_code") or 0)
            last = f"{data.get('description')} ({code})"
            if code == 429:
                delay = min(float((data.get("parameters") or {}).get("retry_after") or delay), 60.0)
                continue
            if code >= 500:
                continue
            raise TelegramError(f"{method}: {last}", fatal=code in (401, 404))
        raise TelegramError(f"{method}: {last}" if tries == 1 else f"{method}: gave up after {tries} tries ({last})")

    async def send(self, target: str, text: str) -> None:
        chat_id = int(target) if target.lstrip("-").isdigit() else target  # or "@channelname"
        async with self._send_lock:  # one send at a time keeps the pieces of a long message in order
            for piece in split_message(text, MAX_TEXT):
                await self._call("sendMessage", {"chat_id": chat_id, "text": piece})

    async def ask_admin(self, admin: int, uid: int, display: str) -> None:
        """The access request with two buttons; the answer comes back as a callback query."""
        buttons = [[{"text": "Пустить", "callback_data": f"approve:{uid}"},
                    {"text": "Отказать", "callback_data": f"deny:{uid}"}]]
        await self._call("sendMessage", {"chat_id": admin, "text": f"{display} (id {uid}) просит доступ к боту.",
                                         "reply_markup": {"inline_keyboard": buttons}})

    async def _download(self, file_id: str) -> bytes:
        info = await self._call("getFile", {"file_id": file_id})
        url = f"{self.api_url}/file/bot{self._token}/{info['file_path']}"
        try:
            async with self._http.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status != 200:
                    raise TelegramError(f"download: HTTP {r.status}")
                data = await r.content.read(MAX_VOICE_BYTES + 1)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise TelegramError(f"download failed: {self._clean(e)}") from None
        if len(data) > MAX_VOICE_BYTES:
            raise TelegramError("download: file too large")
        return data

    # Voice notes ---------------------------------------------------------
    async def transcribe(self, ogg: bytes) -> str:
        """OGG/Opus -> 16 kHz mono (ffmpeg) -> the ASR server's offline transcription."""
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            pcm, err = await asyncio.wait_for(proc.communicate(ogg), 30)
        except asyncio.TimeoutError:
            proc.kill()
            raise TelegramError("ffmpeg timed out") from None
        if proc.returncode != 0 or not pcm:
            raise TelegramError(f"ffmpeg failed: {err.decode(errors='replace').strip()[:200]}")
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        form = aiohttp.FormData()
        form.add_field("file", wav.getvalue(), filename="note.wav", content_type="audio/wav")
        form.add_field("response_format", "json")
        form.add_field("language", self.asr_language)
        try:
            async with self._http.post(self.asr_url, data=form, timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status != 200:
                    raise TelegramError(f"ASR: HTTP {r.status}")
                return ((await r.json(content_type=None)).get("text") or "").strip()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            raise TelegramError(f"ASR failed: {type(e).__name__}") from None

    # Receiving -----------------------------------------------------------
    async def run(self) -> None:
        self._token = self.token()
        if not self._token:
            log.error("no bot token: set $%s or put it in %s (see README)", self.scfg["token_env"],
                      self.scfg["token_file"])
            return
        async with aiohttp.ClientSession() as http:
            self._http = http
            try:
                await self._poll()
            finally:
                self._http = None

    async def _poll(self) -> None:
        offset, backoff, announced = None, 1.0, False
        while True:
            try:
                if not announced:
                    me = await self._call("getMe", tries=1)
                    log.info("connected as @%s", me.get("username"))
                    announced = True
                updates = await self._call(
                    "getUpdates", {**({"offset": offset} if offset else {}), "timeout": POLL_SECONDS,
                                   "allowed_updates": ["message", "callback_query"]},
                    timeout=POLL_SECONDS + 15, tries=1)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except TelegramError as e:
                if e.fatal:
                    log.error("%s; the token is not accepted, giving up (fix it and restart the brain)", e)
                    return
                log.warning("%s; retrying in %.0fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            for update in updates:
                offset = update["update_id"] + 1
                self.chat.spawn(self._update(update))

    async def _update(self, update: dict) -> None:
        """One update, from its own task: a failure here is logged and never reaches the polling loop."""
        try:
            if "callback_query" in update:
                await self._button(update["callback_query"])
            else:
                await self._message(update.get("message") or {})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("handling an update failed: %s", self._clean(e))

    async def _button(self, q: dict) -> None:
        """An admin pressed «Пустить» or «Отказать» under an access request."""
        uid = int((q.get("from") or {}).get("id") or 0)
        action, _, target = str(q.get("data") or "").partition(":")
        member = self.member(uid)
        if member is None or member[1] != "admin" or action not in ("approve", "deny") or not target.isdigit():
            log.warning("ignoring a button press from id %s: %r", uid, q.get("data"))
            await self._call("answerCallbackQuery", {"callback_query_id": q["id"]}, tries=1)
            return
        by = self.chat.who(self, uid)
        if action == "approve":
            name = await self.chat.approve(self, int(target), by)
        else:
            name = self.chat.deny(self, int(target), by)
        done = (f"Пустила: {name}." if action == "approve" else f"Отказала: {name}.") if name \
            else "Этот запрос уже решён."
        await self._call("answerCallbackQuery", {"callback_query_id": q["id"], "text": done}, tries=1)
        msg = q.get("message") or {}
        if msg.get("message_id"):  # replace the buttons with the outcome, so nobody presses them again
            await self._call("editMessageText", {"chat_id": msg["chat"]["id"], "message_id": msg["message_id"],
                                                 "text": f"{msg.get('text', '')}\n{done}"}, tries=1)

    async def _message(self, m: dict) -> None:
        chat, sender = m.get("chat") or {}, m.get("from") or {}
        if not m or chat.get("type") != "private" or sender.get("is_bot") or "id" not in sender:
            if m:
                log.debug("ignoring a message from chat %s (%s): the bot answers private chats only",
                          chat.get("id"), chat.get("type"))
            return
        if time.time() - int(m.get("date") or 0) > STALE_SECONDS:
            log.info("ignoring a message from user %s sent %d s ago", sender["id"], time.time() - int(m["date"]))
            return
        uid = int(sender["id"])
        display = sender.get("username") or sender.get("first_name") or str(uid)
        chat_id = str(chat["id"])

        async def reply(text: str) -> None:
            await self.send(chat_id, text)

        async def typing() -> None:
            try:
                await self._call("sendChatAction", {"chat_id": int(chat_id), "action": "typing"}, tries=1, timeout=5)
            except TelegramError:
                pass  # only a nicety

        voice = m.get("voice") or m.get("audio")
        text = m.get("text")
        if voice and not text:
            if not self.voice_notes:
                return
            if not await self.chat.admit(self, uid, display, reply):  # never download a stranger's file
                return
            if int(voice.get("duration") or 0) > MAX_VOICE_SECONDS or int(voice.get("file_size") or 0) > MAX_VOICE_BYTES:
                await reply("Слишком длинное голосовое. / That voice note is too long.")
                return
            try:
                await typing()
                text = await self.transcribe(await self._download(voice["file_id"]))
            except TelegramError as e:
                log.warning("voice note from %s failed: %s", uid, e)
                await reply("Не получилось разобрать голосовое. / I couldn't make out that voice note.")
                return
            if not text:
                await reply("Ничего не разобрала. / I couldn't hear any words.")
                return
            log.info("voice note from %s: %s", uid, text)
            await reply(f"🎤 «{text}»")
        if not text:
            return
        await self.chat.handle(self, uid, display, text, reply, typing)
