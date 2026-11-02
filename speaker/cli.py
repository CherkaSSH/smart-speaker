"""speaker up | llm | asr | brain | voice | ask TEXT | settings | yandex-login"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import urllib.request

from . import config as config_mod


def llm_cmd(cfg: config_mod.Config) -> list[str]:
    c = cfg["llm"]
    return ["llama-server", "-m", str(cfg.file(c["model"])), "--host", "127.0.0.1", "--port", str(c["port"]),
            "-ngl", str(c["gpu_layers"]), "-c", str(c["context"]), "-np", "1", "--jinja", "--log-disable"]


def embed_cmd(cfg: config_mod.Config) -> list[str]:
    c = cfg["memory"]
    return ["llama-server", "-m", str(cfg.file(c["model"])), "--host", "127.0.0.1", "--port", str(c["port"]),
            "--embeddings", "--pooling", "mean", "-ngl", str(c["gpu_layers"]), "-c", "2048", "-np", "1",
            "-t", str(c["threads"]), "--log-disable"]


def asr_cmd(cfg: config_mod.Config) -> list[str]:
    c = cfg["asr"]
    # asr.backend.threads sizes the CPU compute pool; `serve --threads` would only
    # size the HTTP worker pool, leaving compute at min(8, cores) busy-waiting threads.
    cmd = ["nemo-speech", "serve", "--asr.model.path", str(cfg.file(c["model"])), "--device", c["device"],
           "--asr.backend.threads", str(c["threads"]), "--endpointing",
           "--stop-history-eou-ms", str(c["endpoint_ms"]), "--port", str(c["port"]), "--no-ui"]
    if c["device"] != "cpu":
        # Cross-stream batching aborts on Vulkan at startup (GGML_ASSERT in mul_mat,
        # nemo-speech 0.3.0); two streams don't need it.
        cmd.append("--asr.batching.enabled=false")
    return cmd


async def _wait_http(url: str, timeout: float = 120) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            await asyncio.to_thread(urllib.request.urlopen, url, timeout=2)
            return
        except Exception:
            await asyncio.sleep(0.5)
    raise TimeoutError(f"{url} did not come up")


async def up(cfg: config_mod.Config, args) -> None:
    """Start every part as its own process; stop all if one dies."""
    me = [sys.executable, "-m", "speaker.cli", "--config", str(cfg.path)]
    procs: dict[str, asyncio.subprocess.Process] = {}

    async def start(name: str, cmd: list[str], quiet: bool = False) -> None:
        out = asyncio.subprocess.DEVNULL if quiet else None
        procs[name] = await asyncio.create_subprocess_exec(*cmd, stdout=out, stderr=out)
        logging.info("started %s (pid %d)", name, procs[name].pid)

    try:
        await start("llm", llm_cmd(cfg), quiet=not args.verbose)
        await start("embed", embed_cmd(cfg), quiet=not args.verbose)
        await start("asr", asr_cmd(cfg), quiet=not args.verbose)
        await _wait_http(f"http://127.0.0.1:{cfg['llm']['port']}/health")
        await _wait_http(f"http://127.0.0.1:{cfg['memory']['port']}/health")
        await _wait_http(f"http://127.0.0.1:{cfg['asr']['port']}/health")
        await start("brain", [*me, "brain"])
        await asyncio.sleep(1)
        await start("voice", [*me, "voice"])
        done = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, done.set)
        waits = {asyncio.create_task(p.wait()): n for n, p in procs.items()}
        stopper = asyncio.create_task(done.wait())
        finished, _ = await asyncio.wait([*waits, stopper], return_when=asyncio.FIRST_COMPLETED)
        for t in finished:
            if t in waits:
                logging.error("%s exited with %s; stopping everything", waits[t], t.result())
    finally:
        for p in procs.values():
            if p.returncode is None:
                p.terminate()
        for p in procs.values():
            try:
                await asyncio.wait_for(p.wait(), 10)
            except TimeoutError:
                p.kill()


def main() -> None:
    ap = argparse.ArgumentParser(prog="speaker")
    ap.add_argument("--config", help="config.toml (default: repo config or $SPEAKER_CONFIG)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up", help="run everything")
    sub.add_parser("llm", help="run llama-server with LFM3")
    sub.add_parser("embed", help="run llama-server with EmbeddingGemma 2 for memory")
    sub.add_parser("asr", help="run the Nemotron ASR server")
    sub.add_parser("brain", help="run the thinking service")
    sub.add_parser("voice", help="run the audio service")
    sub.add_parser("settings", help="edit config.toml in a terminal UI")
    sub.add_parser("yandex-login", help="sign in to Yandex Music and print the token to export")
    ask = sub.add_parser("ask", help="ask the running brain in text")
    ask.add_argument("text")
    mode = ask.add_mutually_exclusive_group()
    mode.add_argument("--big", action="store_true", help="use the configured remote LLM")
    mode.add_argument("--harness", action="store_true", help="run the configured agent harness")
    ask.add_argument("--speak", action="store_true", help="also say it through the voice service")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = config_mod.load(args.config)
    if args.cmd == "llm":
        os.execvp("llama-server", llm_cmd(cfg))
    elif args.cmd == "embed":
        os.execvp("llama-server", embed_cmd(cfg))
    elif args.cmd == "asr":
        os.execvp("nemo-speech", asr_cmd(cfg))
    elif args.cmd == "brain":
        from .brain.server import serve
        asyncio.run(serve(cfg))
    elif args.cmd == "voice":
        from .voice.service import run
        asyncio.run(run(cfg))
    elif args.cmd == "settings":
        from .settings import run as edit_settings
        edit_settings(cfg.path)
    elif args.cmd == "yandex-login":
        from .brain.music import yandex_login
        asyncio.run(yandex_login(cfg["music"]["yandex"]["token_env"]))
    elif args.cmd == "up":
        asyncio.run(up(cfg, args))
    elif args.cmd == "ask":
        b = cfg["brain"]
        req = urllib.request.Request(f"http://{b['host']}:{b['port']}/ask",
                                     json.dumps({"text": args.text, "speak": args.speak,
                                                 "big": args.big, "harness": args.harness}).encode(),
                                     {"Content-Type": "application/json"})
        print(json.dumps(json.load(urllib.request.urlopen(req)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
