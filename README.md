# Smart speaker

A homemade, Yandex-Station-style voice assistant. Speech recognition, the default LLM and speech synthesis run locally; weather and Wikipedia lookups use the internet. Explicit requests can escalate to a remote LLM, and coding agents (omp sessions) can be started and talked to by voice.

The default character is a sarcastic Russian speaker woken by **«Эй, колонка»** (or just **«колонка»**). It answers **«Чё такое?»** to a bare wake phrase. Russian and English commands are supported. Wake phrases live in `[wake] phrases`; the small always-on wake model also needs their spellings in `[gate] grammar` (see the comment there). Use real words: a made-up wake word such as «гптшка» is biased into recognition everywhere, and Nemotron started hearing it in song names («газ гптшкат» for «Хас-Булат») and in loud music.

## Parts

- **Voice** (`speaker/voice/`): microphone capture, acoustic echo cancellation, wake/stop phrases, echo rejection, barge-in, follow-up listening and Silero TTS playback. The output device is opened only while something plays.
- **Speech gate** (`speaker/voice/gate.py`): an energy check and Silero VAD (~2 MB) skip silence, and Vosk small Russian (~50 MB) detects the wake phrase. Two seconds of pre-roll preserve the complete command. Our own playback (plus a 0.5 s tail) is ignored so answers can't wake the speaker.
- **ASR**: Nemotron 0.6B streaming through `nemo-speech`, on the Radeon iGPU via Vulkan. It activates after the small wake gate and stays active during an answer or ring. A Russian stream verifies the wake phrase and handles Russian commands; an optional auto-detect stream handles English once a conversation is open.
- **Brain** (`speaker/brain/`): LFM3-1.7B through `llama-server` on the Radeon iGPU, with timers, alarms, volume, arithmetic, weather and Wikipedia tools. Common commands bypass the LLM. Optional escalation streams a larger OpenAI-compatible API model. Personal memory uses EmbeddingGemma 2 semantic search through a second `llama-server`. omp coding-agent sessions connect to it as threads (`speaker/brain/agents.py`, `omp/speaker.ts`).
- **Configuration**: one `config.toml` shared by every process.

## Setup

Needs an ALSA audio device (a mic and a speaker, or a headset) and Linux. The speaker was developed on NixOS with the flake below; without Nix, see [Setup without Nix](#setup-without-nix).

```sh
nix develop
uv sync
source .venv/bin/activate
```

The model files are not downloaded by the flake. Put these files under `models/` (or change the paths in `config.toml`):

```text
models/LFM3-1.7B-QAD-Q4_0.gguf
models/embeddinggemma-2-Q8_0.gguf
models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
models/silero/v5_6_ru.pt
models/silero/v3_en.pt
models/silero/vad.jit
models/vosk-model-small-ru-0.22/
models/localvqe-v1.4-aec-200K-f32.gguf
models/localvqe-pi-aec-v1-49k-f32.gguf
```

LFM3 GGUF files are published at `LiquidAI/LFM3-1.7B-GGUF` on Hugging Face. LFM3 requires the llama.cpp build from this flake, not the older nixpkgs build. The embedding model is `embeddinggemma-2-Q8_0.gguf` from `ggml-org/embeddinggemma-2-GGUF`. The VAD file is `silero_vad.jit` from `snakers4/silero-vad`; the wake model is `vosk-model-small-ru-0.22` from `alphacephei.com/vosk/models`. The two echo-canceller models, `localvqe-v1.4-aec-200K-f32.gguf` and `localvqe-pi-aec-v1-49k-f32.gguf`, are from `LocalAI-io/LocalVQE` on Hugging Face; the flake builds their library.

`alsa/speaker.conf` currently targets the Razer Kraken V3 X (`CARD=X`) and exposes shared `speaker_in`/`speaker_out` devices. Change the card there for standalone microphones or speakers, then re-enter `nix develop` (the dev shell copies the file). The capture period is 100 ms and the playback period 40 ms to keep idle CPU wakeups low. The current headset test setup plays the simulated user into the **right earcup**, near the microphone.

### Optional escalation

It is disabled by default. Once enabled, LFM3 can also call `ask_big_model` on its own for requests beyond its abilities. It then asks first («Это сложновато. Спросить большую модель?»): "да" sends the question to the big model, "нет" makes LFM3 answer itself. Any other request drops the question. Turn this off with `confirm = false`. Explicit requests («спроси большую модель…», `--big`) never ask.

For a larger API model, set `enabled = true` in `[remote_llm]`, configure its OpenAI-compatible URL and model, and export the key:

```sh
export REMOTE_LLM_API_KEY=...
speaker ask "Prove that there are infinitely many primes" --big
```

### Coding agents (omp)

Every omp session that loads `omp/speaker.ts` connects to the brain (`ws://127.0.0.1:8770/omp`, or `$SPEAKER_BRAIN_URL`) and becomes a numbered thread; it retries quietly while the brain is down. Load it in every session:

```sh
mkdir -p ~/.omp/agent/extensions
ln -s "$PWD/omp/speaker.ts" ~/.omp/agent/extensions/speaker.ts
```

- **«Запусти агента, …»** starts a new omp session in `[omp] cwd` with that task, as a tab of the Tern session `[omp] tern_session` («agents»). Tern's daemon keeps it running across brain restarts, and it shows in your Tern window, where you can watch it or type into it. Without a task it asks «Что сделать агенту?».
- **«Скажи агенту …»** sends what follows as a message: it starts a turn, or steers the running one at its next step. It shows in the session as «[голосом через колонку] …».
- **«Спроси агента …»** asks a side question, like `/btw`: answered from the session's context in a separate model call, without touching the running turn.
- **«Какие агенты?»** lists the threads: number, title and whether each is working.
- With several threads, name one by number, «скажи второму агенту …», or by name: the tag omp puts at the start of its title («скажи агенту алекса …» for "ALEXA: …") or its folder («спроси агента смарт спикер …»). Without either, the request goes to the thread you last talked with or that last spoke, or to the only one. Only main sessions become threads: subagents, the advisor and `omp -p` runs don't connect, and subagents don't get the speaker tools.
- **Closed threads reopen.** Talking to a thread that is no longer open, the one you last talked with or any session named as above (the brain looks through the 15 most recently used sessions in `[omp] sessions`), resumes it with `omp --resume` in a new tab of the «agents» Tern session, with its whole history, then passes on the message or side question («Открываю агента и передаю»). A session still open in an omp that isn't connected (started before the extension was installed) would be resumed a second time; omp then writes the copy to a new session file instead of mixing the two.

Once you have spoken to a session, its agent gets two tools: `speaker_say` (say something aloud: progress, the outcome) and `speaker_ask` (ask aloud and wait up to two minutes for the spoken answer; the speaker listens right after the question, and later answers need the wake phrase within 30 seconds). Sessions the speaker started have them from the start, and their built-in `ask` is redirected to `speaker_ask`, since you started them by voice and may not be at the screen. When a run you asked for by voice ends without the agent speaking, the speaker says «Агент закончил.» and the start of its final answer. Agents' messages wait until you are done talking to the speaker, then play over ducked music; with several threads they start with «Агент два: …».

Any omp session on the machine can be driven this way once the extension is installed, with the same tools and permissions as when you type into it; the brain listens on localhost only by default (`[brain] host`).

### Web search

LFM3 calls `web_search` for news, prices, schedules and other current information that Wikipedia (`lookup`) wouldn't have. It writes the search in English, and answers from the English excerpts in the user's language. Searches go to [Parallel](https://parallel.ai): without a key, its free hosted endpoint (about 1–2 s per search, lower rate limits); with `PARALLEL_API_KEY` set, its API in the configured `mode` (`turbo` takes about 0.2 s and needs English queries, which the tool already uses). Parallel can't cap spending, so `max_per_day` caps searches here.

When the big model is configured, a web search first asks «Ищу в интернете. Пусть ответит большая модель?» while the search already runs: "да" lets the big model answer from the results, "нет" lets LFM3 answer. Set `offer_big_model = false` in `[web_search]` to always answer with LFM3.

While a slow tool runs (web search, Wikipedia, weather, big model) the speaker says a short line first («Сейчас поищу», «Сейчас гляну», «Смотрю погоду»), then continues with the answer.

### Music

Say «включи музыку» (shuffle the library), «включи май бонни» (a song or artist; Cyrillic spellings of Latin titles work), «включи радио» or «включи радио секрет эйджент» (internet radio from `[music] radio`), then «пауза», «продолжи», «следующая», «предыдущая», «что играет?», «выключи музыку». Each needs the wake phrase, as on Yandex Station. Pause, resume and skip act silently. Anything vaguer goes to LFM3 through the `play_music` and `music_control` tools; when nothing matches, it gets the library's titles and can retry. Waves, radio and «включи музыку» play on without end (a new shuffle, the next wave batch).

More commands: «какие слова?», «какой там текст?» and «о чём поётся» read out the line that is playing and the next one when the track has timed (LRC) lyrics and the playback position is known, otherwise the first two to four lines, and «нет текста» when the song has none. «мне нравится»/«поставь лайк» likes the current Yandex Music track, «не нравится»/«дизлайк» marks it «не рекомендовать» and skips to the next track; tracks from a wave also send the app's rotor feedback, and other sources answer that likes only work in Yandex Music. «повтори этот трек»/«поставь на повтор» repeats the current track until «выключи повтор» or another request, «перемешай» shuffles the rest of a queue (a wave answers that there is nothing to shuffle), and «включи режим диджея»/«выключи режим диджея» turn on and off saying «Дальше: …» before each next track (`[music] announce`, off by default; audiobook chapters and radio are never announced). «включи подкаст <название>» plays the latest episodes of the best-matching Yandex Music podcast, newest first.

When the songs you asked for by name end (one song, or all of an artist's or album's), `[music] after_request` decides what happens: `"ask"` (the default) asks «Включить мою волну?» and listens as after any question; «да» starts it, «нет» answers «Ладно», and anything else drops the question. `"continue"` starts it without asking, `"stop"` stops. It is My Wave when Yandex Music has a token, otherwise more from the same source («Включить ещё музыку?»). «следующая» on the last requested song goes straight on. The question is the brain's own `ask` message to the voice service, which says it only when idle (not in the middle of another conversation).

The library is `music/` (any format ffmpeg reads). Optional sidecars per file: `NAME.json` with `title`, `artist` and `album`, and `NAME.txt` with the lyrics. Without a `.json`, the file's own tags are used. With `echo_cancellation = false`, lyrics are added to echo removal while the song plays, so sung words are not taken for your voice. With it on, the canceller removes the singer, and lyric words in a transcript are yours («включи хас булат удалой» over Хас-Булат). Yandex Music brings the text lyrics of a streamed track along with it the same way, and keeps the timed ones for «какие слова?».

`music/` is not in git. The test library is four public-domain recordings with vocals from Wikimedia Commons, each with lyrics and metadata sidecars:

| File | Song | Performer | Year | License on Commons |
|---|---|---|---|---|
| `my-bonnie-lies-over-the-ocean.mp3` | My Bonnie Lies Over the Ocean | Haydn Quartet | 1901 | PD-US-record-expired |
| `shine-on-harvest-moon.ogg` | Shine On, Harvest Moon (unrestored) | Ada Jones and Billy Murray | 1909 | PD-US-record-expired |
| `ey-ukhnem.ogg` | Эй, ухнем! | Фёдор Шаляпин | 1902 | PD-RusEmpire |
| `khas-bulat.mp3` | Хас-Булат удалой | Надежда Плевицкая | 1909 | PD-RusEmpire |

Each `.json` sidecar records the source and lyrics URLs. The Russian lyrics are the full song texts; the old discs may sing fewer verses.

Music turns down to `duck_volume` while the speaker listens, thinks or answers, and when you start talking over it: the wake gate learns how loud the music normally sounds in the mic and ducks it when the mic gets `talk_factor` times louder than that while it hears speech. That lowers the music before the wake phrase is even recognized.

#### Streaming: Navidrome and Yandex Music

`[music] providers` lists where music comes from; a request plays from the first one that finds something. `"radio"` only answers radio requests. Playback takes anything ffmpeg can open, so streams need no voice-service changes; each provider resolves a fresh stream URL right before a track plays (`speaker/brain/music.py`).

**Navidrome** (or any Subsonic server): set `url` and `user` in `[music] navidrome` and export the password under the name in `password_env` (default `NAVIDROME_PASSWORD`). Requests use Subsonic token auth, so the password itself is not sent. Navidrome's search is word-based and wouldn't find "My Bonnie" from «май бонни», so the provider fetches the whole song list (500 songs per request, refreshed every 10 minutes) and matches it like the local library: title, artist and album.

**Yandex Music** (through the [`yandex-music`](https://github.com/MarshalX/yandex-music-api) library): run `speaker yandex-login`, open the printed page and enter the code. The token is saved to `state/yandex_token` (readable only by you, git-ignored) and survives restarts; `$YANDEX_MUSIC_TOKEN`, if set, takes precedence. Full tracks need an account with Yandex Plus; without a token the API serves 30-second previews and there are no waves. Yandex's own search handles misheard names, and its best result decides what plays: a song, an artist's popular songs («включи земфиру»), an album or a playlist. On this machine about 1 in 4 new TLS connections to the API stalled for 5 s and was reset, so the provider keeps connections alive and retries a failed request once. Starting a track then takes 0.6–0.9 s (search, download info, signed link); the first one also signs in (3.7 s).

Yandex's radio is its waves, each an endless stream in batches of five:

| Say | Plays |
|---|---|
| «включи музыку», «включи радио», «включи мою волну» | My Wave |
| «включи радио рок», «включи волну джаз», «включи рок», «включи спокойную музыку», «радио для тренировки», «радио девяностых» | the wave for that genre, mood, activity or decade («Рок», «Джаз», «Спокойное», «Тренируюсь», «1990-е»); 600 of them, by name |
| «включи волну по этому треку», «включи похожее» | a wave seeded by the song playing; a local or Navidrome song is looked up on Yandex first |
| «следующая» at the end of a batch | the next batch |

Wave names are matched by whole words and Russian stems (тренировки → Тренируюсь, рэп → Русский рэп). A name needs 0.8 similarity, and a name under four letters must match whole, so «радио groove salad» still reaches the internet station and «музыку ддт» plays ДДТ instead of the «Даб» wave. «…музыку» with a description tries a wave first, then a normal search («музыку битлз»). Like the app, the speaker reports each started, finished and skipped wave track: without that, 7–9 of 20 tracks in the next batches repeated; with it, none did, and skips teach My Wave.

**Audiobooks** come from Yandex Books (through the [`yandex-book-api`](https://github.com/stepan163s/yandex-book-api) library) with the same Yandex token, no extra login. Say «включи аудиокнигу мастер и маргарита», «почитай книгу …» or «включи книгу …»: the speaker picks the best-matching audiobook, plays one chapter per queue item (titled «Мастер и Маргарита, глава 3»), and says «Включаю аудиокнигу «…», глава N.» «следующая глава» and «предыдущая глава» work like next/previous, and «что за книга?» names the book and chapter. Chapters are signed HLS streams resolved right before each one plays. The brain saves the chapter and the second you stopped at in `[music] books.state_file` (`state/books.json`), on pause, stop, skip, chapter change, and every 30 seconds while playing: «включи аудиокнигу …» starts that book where you left it (chapter 1 the first time) and «продолжи книгу» resumes the most recent one. The last chapter stops the queue with «Книга закончилась.» rather than offering My Wave. Books are only searched for these explicit requests, so «включи музыку» and song titles never return chapters.

Spotify doesn't hand out stream URLs, so it would need a Spotify Connect receiver such as librespot instead of a provider.

### Echo cancellation

With `[audio] echo_cancellation = true` (the default), the mic signal passes through [LocalVQE](https://github.com/localai-org/LocalVQE) before the wake gate and ASR see it. LocalVQE is a small neural echo canceller (an adaptive filter followed by a neural mask, trained on double talk). It subtracts what the speaker itself plays (music, answers, sounds) and keeps the user's voice. Every output stream records its audio, resampled to 16 kHz, on a shared timeline counted in samples (`speaker/voice/echo.py`), so the reference lines up with the mic to within a few milliseconds. LocalVQE finds the remaining delay itself. The flake builds `liblocalvqe.so`, and the voice service calls it through ctypes, delaying the mic by one 16 ms hop.

Two models run, one at a time, because each failed where the other worked:

| Test, same mic | v1.4-AEC 200K (`echo_model`) | pi-AEC 49K (`echo_music_model`) | No cancellation |
|---|---|---|---|
| «стоп» over a long spoken answer | 3/4 | 0/4 | 0/4 |
| «Эй, колонка, что играет?» over Bostaurus at volume 7, in the service | 2/12 | 27/30 | — |

While our own speech or a sound plays (and for 1 s after), the 200K model runs: 1.4 ms of CPU per 16 ms hop, 9% of one core. While only music plays, the 49K model runs: 1.1 ms per hop, 7%. Otherwise the mic passes through unchanged, and the voice service idles at about 1% CPU. In the service, «стоп» or «хватит» then stopped 5 of 5 long answers, one of them over playing music.

Music measurements with the 49K model went through the real capture and playback path, with a recorded «Эй, колонка, поставь на паузу» played over each song from the same earcup. The loud test track is Bostaurus, «Will Bullets Stop The Violence» (2006, hardcore rock mastered at −8 LUFS; [CC BY-NC-ND 2.0 DE](https://archive.org/details/Bostaurus.Demo)), saved locally as `music/bostaurus-will-bullets.mp3`:

| Song, volume | Music at the mic (the voice is −34 dBFS) | Music removed | Wake phrase verified, raw / cancelled |
|---|---|---|---|
| My Bonnie, 5 | −46 dBFS | 23 dB | ✗ (lyrics mixed in) / ✓ |
| Хас-Булат, 5 | −39 dBFS | 28 dB | ✓ / ✓ |
| Bostaurus, 5 | −38 dBFS | 14 dB | ✓ / ✓ |
| Bostaurus, 7 | −32 dBFS | 14 dB | ✗ / ✓ |
| Bostaurus, 10 | −23 dBFS | 13 dB | ✗ / ✗ |

At volume 10, the service answered «Эй, колонка, что играет?» over Bostaurus 3 times out of 24. There the voice is 10 dB below the music at the mic, the speech detector never fires even after cancellation, and the music is never turned down.

WebRTC's AEC3 was tried first and dropped. Its echo suppressor also removed the user's voice: with Хас-Булат at volume 5, only 47% of the voice was left. With Bostaurus at volume 7, 34% was left, and recognition failed where the raw mic still worked.

Spoken answers stay in the text-level echo removal as a second line of defence. A wake phrase that comes after our own echoed words is dropped as one of them misheard («Я расскажу тебе длинную колонка» for «…длинную сказку»). Set `echo_cancellation = false` if the output and input are different sound cards: their clocks drift apart, so the reference no longer lines up with the mic.

### Personal memory

`save_memory` and `search_memory` let LFM3 persist durable facts and retrieve them when relevant. For example, say **«запомни, что я люблю овсяное молоко»**, then later ask what milk you prefer. Text and normalized 768-dimensional embeddings are stored in `state/memory.sqlite3`.

Retrieval uses **EmbeddingGemma 2** (text-only Q8_0 GGUF) served by `speaker embed` on the iGPU: about 18 ms per embedding, ~200 MB of memory, and no PyTorch in the brain. Before each LLM answer, memories closely related to the request (`recall_threshold`) are added to the prompt automatically, because LFM3 does not reliably decide to search on its own; broader questions such as «что ты обо мне помнишь?» use the `search_memory` tool. If the embedding server is down, answers still work without recall.

## Run

```sh
speaker up
```

For separate terminals, start in this order:

```sh
speaker llm
speaker embed
speaker asr
speaker brain
speaker voice
```

Use `speaker -v up` to include llama.cpp and ASR server logs. Stop `speaker up` with Ctrl-C; it stops all five processes.

Ask the running brain without using the microphone:

```sh
speaker ask "какая завтра погода в Питере?"
speaker ask "поставь таймер на пять минут" --speak
speaker ask "спроси большую модель: объясни квантовую запутанность"
speaker ask "запусти агента: проверь этот проект"
```

Use another configuration with `speaker --config /path/to/config.toml up` or `$SPEAKER_CONFIG`. A `config.local.toml` in the repo root is git-ignored, for your own settings.

### As systemd services

`systemd/` has a user unit per process and `speaker.target` for all five; each restarts on failure and logs to the journal. They work without Nix (the dev shell's variables come from an environment file instead):

```sh
systemd/install.sh                          # units into ~/.config/systemd/user, env into ~/.config/speaker/env
$EDITOR ~/.config/speaker/env               # PATH to llama-server/nemo-speech/ffmpeg, LOCALVQE_LIB, ALSA_CONFIG_PATH, keys
systemctl --user enable --now speaker.target
sudo loginctl enable-linger $USER           # start at boot without logging in
journalctl --user -u speaker-voice -f       # or speaker-brain, speaker-llm, ...
```

`systemd/install.sh uninstall` removes them. The voice service needs access to `/dev/snd`; with linger and no desktop session that usually means adding yourself to the `audio` group.

## Setup without Nix

These steps mirror the flake's builds (same versions and CMake flags); they have only been exercised through the flake on NixOS. Package names are Debian/Ubuntu's.

```sh
sudo apt install ffmpeg alsa-utils libportaudio2 libsndfile1-dev libasound2-dev git cmake ninja-build pkg-config \
    glslc libvulkan-dev spirv-headers libssl-dev libsentencepiece-dev libabsl-dev libprotobuf-dev protobuf-compiler
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv fetches Python 3.12 itself

# llama.cpp with Vulkan (LFM3 needs b11960 or newer)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout b12017
cmake -B build -G Ninja -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build && sudo cmake --install build && cd ..

# nemo-speech (Nemotron ASR server)
git clone --recursive https://github.com/NVIDIA/NeMo-Speech.cpp && cd NeMo-Speech.cpp
git checkout 20999cc921d2b14aa58708f61756a8b2e787ba20 && git submodule update --init --recursive
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_LLAMAFILE=ON -DGGML_VULKAN=ON \
    -DNEMO_SPEECH_GGML_PATCHED=ON -DNEMO_SPEECH_BUILD_ASR=ON -DNEMO_SPEECH_BUILD_DIAR=OFF -DNEMO_SPEECH_BUILD_TTS=OFF \
    -DNEMO_SPEECH_BUILD_NMT=OFF -DNEMO_SPEECH_BUILD_SPEECHLM=ON -DNEMO_SPEECH_BUILD_HTTP=ON -DNEMO_SPEECH_BUILD_GRPC=OFF \
    -DNEMO_SPEECH_BUILD_TESTS=OFF
cmake --build build && sudo cmake --install build && cd ..

# LocalVQE echo canceller (optional: or set [audio] echo_cancellation = false)
git clone --recursive https://github.com/localai-org/LocalVQE && cd LocalVQE/ggml
git checkout 7f61cd05095d42e56a135b303f5bb4d761da22b9 && git submodule update --init --recursive
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLOCALVQE_BUILD_SHARED=ON && cmake --build build
sudo cp -P build/bin/*.so* /usr/local/lib/ && sudo ldconfig && cd ../..

uv sync
```

Then download the models (see [Setup](#setup)), point `alsa/speaker.conf` at your card, and either export what the dev shell sets (`LOCALVQE_LIB=/usr/local/lib/liblocalvqe.so`, `ALSA_CONFIG_PATH=/usr/share/alsa/alsa.conf:$PWD/alsa/speaker.conf`) before `speaker up`, or install the [systemd services](#as-systemd-services). Without a Vulkan GPU, set `[asr] device = "cpu"` and `[llm] gpu_layers = 0` (and `[memory] gpu_layers = 0`).

## Settings

```sh
speaker settings
```

Opens `config.toml` in a terminal editor: sections on the left, fields on the right, with each comment from the file shown above the keys it describes. Switches edit booleans, numbers are checked as you type (thresholds, volume and temperatures also against their ranges), lists take one item per line, and the ASR device and Russian voice are drop-downs. **Ctrl+S** saves, keeping the file's comments and layout, and lists the processes to restart; **Ctrl+R** reverts unsaved edits; **Ctrl+Q** quits (twice if there are unsaved changes).

## Voice behavior

- Say **«Эй, колонка, поставь таймер на пять минут»**, or say the wake phrase alone and wait for the acknowledgement.
- During speech or a ringing timer, any speech interrupts and becomes the next command; the wake phrase is not needed. A short **«стоп»**, **«хватит»** or another configured stop word just stops the response.
- Ordinary answers return to idle. A follow-up without the wake phrase is accepted only when the speaker explicitly asked you a question. Requests have no conversation history; only timers, alarms and volume persist.
- Start with **«спроси большую модель»** to use `[remote_llm]`; **«запусти агента»**, **«скажи агенту»** and **«спроси агента»** reach omp threads (see Coding agents). English triggers also work.
- While a slow answer is on its way (the big model, a web search, an agent's reply, a tool after «Смотрю погоду»), soft music plays after `[voice] hold_music_after` seconds of silence (2.5) and stops the moment the answer is ready. `hold_music = "builtin"` is a synthesized bell loop; give the path of your own audio file instead, or set `hold_music_after = 0` to turn it off. Nothing extra plays when music is already on.
- Volume commands use levels 0–10. Timers, alarms and the sleep timer persist in `state/brain.json`; the shopping list in `state/shopping.json`; personal memories in `state/memory.sqlite3`; audiobook places in `state/books.json`.
- Smart-home control is not implemented.

Tune wake matching, stop words, microphone gain, output channels, TTS voices, endpoint silence, the 4096-token generation ceiling, escalation backends and the character prompt in `config.toml`, by hand or with `speaker settings`.

## Everyday extras

These commands are answered instantly, without the LLM, like timers and the clock.

**Sleep timer.** «выключи музыку через 30 минут», «останови музыку через 10 минут», «таймер сна на час» or "turn off the music in 30 minutes": when the time comes the music stops silently, with no ring. There is one sleep timer at a time; a new one replaces it, and «отмени таймер сна» cancels it. It persists in `state/brain.json` like the other timers.

**Alarms with music.** «разбуди меня в 7 под Кино» and «поставь будильник на 7:30 с музыкой» store the artist or song (or «с музыкой» = any music) on the alarm. When it fires the speaker says «Подъём, будильник!» and starts the music; a plain alarm only rings. The LLM's `set_alarm` tool takes the same optional `music` argument, and `set_sleep_timer` and `exchange_rate` cover the other free phrasing. Alarms persist with their music in `state/brain.json`.

**Morning show.** «доброе утро», «утреннее шоу» or «включи утреннее шоу» (also "good morning" / "morning show") answers in one go: a greeting for the time of day, the clock, today's weather from `[brain] home_city`, the alarms and reminders still due today, three headlines from `[news]`, then «Включаю музыку» and My Wave or a library shuffle.

**News.** «что нового?», «новости», «расскажи новости» (or "what's new", "headlines") read the five latest headlines from the RSS feeds in `[news] feeds_ru`, or `feeds_en` for English answers, taking a few from each feed in turn. Headlines are cached for `cache_minutes`; if every feed fails the last batch is reused. No API key and no LLM are involved.

**Exchange rates.** «курс доллара», «курс евро», «сколько стоит юань», «сколько будет 100 долларов в рублях» answer from the Central Bank of Russia's daily rates (`cbr-xml-daily.ru`, falling back to `cbr.ru`), cached for an hour: «Доллар — восемьдесят один рубль шестьдесят семь копеек.» The `exchange_rate` tool does the same for free phrasing. Currencies without a rate in the bank's list are reported as such.

**Shopping list.** «добавь молоко в список покупок», «добавь в список покупок хлеб и яйца», «что в списке покупок?», «удали молоко из списка покупок», «очисти список покупок». Items are added one per phrase part, listed in the order they were added, removed by fuzzy word match (the first four letters, so «молока» removes «молоко»), and kept in the JSON file named by `[shopping] state_file` across restarts.

**Translation.** «как по-английски яблоко», «переведи на английский я люблю музыку», «как будет спасибо по-немецки» get a one-line translation («Перевод: danke»). A short dedicated prompt is sent to LFM3 with no tools, because the chat prompt alone made it define the word, call Wikipedia, or answer German wrongly in testing.
