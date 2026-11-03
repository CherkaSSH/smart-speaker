/**
 * Connects an omp session to the smart speaker's brain, so it can be talked to by voice.
 *
 * Load it in every session (symlink it into ~/.omp/agent/extensions/). Each main session
 * dials the brain (SPEAKER_BRAIN_URL, default ws://127.0.0.1:8770/omp) and keeps retrying
 * quietly while the brain is down. The brain can then:
 *   prompt  - send what the user said («скажи агенту …»); it starts a turn or steers the running one
 *   btw     - ask a side question («спроси агента …»), answered from the context without touching the run
 *   answer  - deliver the spoken reply to a speaker_ask question
 * Once the user has spoken to a session, the agent gets two tools: speaker_say and speaker_ask.
 * Sessions the speaker starts itself (SPEAKER_THREAD set) have them from the start, and their
 * built-in `ask` is redirected to speaker_ask: they were started by voice, the user may not be at the screen.
 * Only main sessions connect; subagents, the advisor and print/json runs don't, and subagents can't
 * get the tools (restricted children don't admit extension tools).
 *
 * Protocol (JSON text frames):
 *   session -> brain  {"type": "hello", "session", "name", "cwd", "busy", "thread"}  on connect and session switch
 *                     {"type": "state", "busy", "name"}
 *                     {"type": "say", "text"}                    speaker_say
 *                     {"type": "ask", "id", "text"}              speaker_ask; the brain answers with "answer"
 *                     {"type": "done", "text"}                   a voice-requested run ended without speaking
 *                     {"type": "btw_reply", "id", "text", "error"?}
 *   brain -> session  {"type": "prompt", "text"}
 *                     {"type": "btw", "id", "text"}
 *                     {"type": "answer", "id", "text": str | null}   null: nobody answered
 */
import type { ExtensionAPI, ExtensionContext } from "@oh-my-pi/pi-coding-agent";

const URL = process.env.SPEAKER_BRAIN_URL || "ws://127.0.0.1:8770/omp";
const THREAD = process.env.SPEAKER_THREAD || ""; // set when the speaker started this session
const RECONNECT_MS = 5000;
const ASK_TIMEOUT_MS = 120_000;
const VOICE_TOOLS = ["speaker_say", "speaker_ask"];
const VOICE_MARK = "[голосом через колонку]";

type Incoming =
	| { type: "prompt"; text: string }
	| { type: "btw"; id: string; text: string }
	| { type: "answer"; id: string; text: string | null };

function parseIncoming(data: unknown): Incoming | undefined {
	if (typeof data !== "string") return undefined;
	const m: unknown = JSON.parse(data);
	if (!m || typeof m !== "object" || !("type" in m)) return undefined;
	const id = "id" in m && typeof m.id === "string" ? m.id : undefined;
	const text = "text" in m && typeof m.text === "string" ? m.text : undefined;
	if (m.type === "prompt" && text !== undefined) return { type: "prompt", text };
	if (m.type === "btw" && id !== undefined && text !== undefined) return { type: "btw", id, text };
	if (m.type === "answer" && id !== undefined) return { type: "answer", id, text: text ?? null };
	return undefined;
}

/** The text of the last assistant message among `messages` (agent_end's payload). */
function lastAssistantText(messages: unknown): string {
	if (!Array.isArray(messages)) return "";
	for (let i = messages.length - 1; i >= 0; i--) {
		const m: unknown = messages[i];
		if (!m || typeof m !== "object" || !("role" in m) || m.role !== "assistant" || !("content" in m)) continue;
		const content = m.content;
		if (typeof content === "string") return content.trim();
		if (!Array.isArray(content)) continue;
		const text = content
			.map((b: unknown) => (b && typeof b === "object" && "type" in b && b.type === "text" && "text" in b ? String(b.text) : ""))
			.join("")
			.trim();
		if (text) return text;
	}
	return "";
}

export default function speaker(pi: ExtensionAPI) {
	const z = pi.zod;
	let ctx: ExtensionContext | undefined;
	let ws: WebSocket | undefined;
	let connected = false;
	let busy = false;
	let voiceRun = false; // the current run was asked for by voice
	let spoke = false; // ...and the agent already said something on the speaker
	let toolsOn = false;
	const asks = new Map<string, (text: string | null) => void>();

	const send = (msg: object): boolean => {
		if (!ws || !connected) return false;
		ws.send(JSON.stringify(msg));
		return true;
	};

	const hello = () =>
		send({
			type: "hello",
			session: ctx?.sessionManager.getSessionId() ?? "",
			name: pi.getSessionName() ?? "",
			cwd: ctx?.cwd ?? "",
			busy,
			thread: THREAD,
		});

	const enableVoiceTools = async () => {
		if (toolsOn) return;
		toolsOn = true;
		const active = pi.getActiveTools();
		await pi.setActiveTools([...active, ...VOICE_TOOLS.filter(t => !active.includes(t))]);
	};

	const onPrompt = async (text: string) => {
		await enableVoiceTools();
		voiceRun = true;
		spoke = false;
		// Idle: starts a turn. Running: steers it at the next step.
		pi.sendUserMessage(`${VOICE_MARK} ${text}`);
	};

	const onBtw = async (id: string, text: string) => {
		try {
			if (!ctx?.runEphemeralTurn) throw new Error("side questions are not supported here");
			const { replyText } = await ctx.runEphemeralTurn({
				promptText:
					`${text}\n\n(The user asked this by voice through a smart speaker. Answer in one to three short ` +
					"spoken sentences, in the language of the question, without markdown, code or file paths.)",
				tools: false,
				maxTokens: 800,
			});
			send({ type: "btw_reply", id, text: replyText });
		} catch (err) {
			send({ type: "btw_reply", id, text: "", error: String(err) });
		}
	};

	const onMessage = async (data: unknown) => {
		const msg = parseIncoming(data);
		if (!msg) return;
		if (msg.type === "prompt") await onPrompt(msg.text);
		else if (msg.type === "btw") await onBtw(msg.id, msg.text);
		else {
			asks.get(msg.id)?.(msg.text);
			asks.delete(msg.id);
		}
	};

	const connect = () => {
		if (ws) return;
		const sock = new WebSocket(URL);
		ws = sock;
		// Raw socket callbacks run outside omp's handler isolation: a throw here would end the session.
		sock.onopen = () => {
			connected = true;
			try {
				hello();
			} catch {}
		};
		sock.onmessage = ev => {
			onMessage(ev.data).catch(() => {});
		};
		sock.onclose = () => {
			if (ws === sock) ws = undefined;
			connected = false;
			for (const resolve of asks.values()) resolve(null);
			asks.clear();
		};
		sock.onerror = () => {};
	};

	pi.on("session_start", async (_event, c) => {
		if (c.agent.kind !== "main" || c.mode === "print" || c.mode === "json") return;
		ctx = c;
		if (THREAD) await enableVoiceTools();
		connect();
		c.setInterval(connect, RECONNECT_MS);
	});

	pi.on("session_switch", async (_event, c) => {
		if (!ctx) return;
		ctx = c;
		hello();
	});

	pi.on("session_shutdown", async () => {
		ws?.close();
	});

	pi.on("agent_start", async () => {
		busy = true;
		send({ type: "state", busy, name: pi.getSessionName() ?? "" });
	});

	pi.on("agent_end", async event => {
		busy = false;
		if (voiceRun && !spoke) {
			const text = lastAssistantText("messages" in event ? event.messages : undefined);
			send({ type: "done", text });
		}
		voiceRun = false;
		send({ type: "state", busy, name: pi.getSessionName() ?? "" });
	});

	// A session the speaker started by voice asks by voice.
	pi.on("tool_call", async event => {
		if (THREAD && event.toolName === "ask")
			return {
				block: true,
				reason: "The user is away from the screen and talks to you through the speaker. Ask with speaker_ask.",
			};
	});

	pi.registerTool({
		name: "speaker_say",
		label: "Speaker: say",
		description:
			`Say something aloud on the user's smart speaker. Messages marked ${VOICE_MARK} were spoken by a user ` +
			"who may be away from the screen: tell them what they need to know this way (progress worth knowing, " +
			"the outcome when you finish), in their language, in one to three short spoken sentences. No markdown, " +
			"code, file paths or URLs; the speaker reads it out as is.",
		parameters: z.object({ text: z.string().describe("What to say, plain spoken text") }),
		defaultInactive: true,
		async execute(_id, params) {
			spoke = true;
			const ok = send({ type: "say", text: params.text });
			return {
				content: [{ type: "text", text: ok ? "Said on the speaker." : "The speaker is not connected; nobody heard it." }],
				details: { ok },
			};
		},
	});

	pi.registerTool({
		name: "speaker_ask",
		label: "Speaker: ask",
		description:
			"Ask the user a question aloud on the smart speaker and wait for their spoken answer (up to two " +
			"minutes). Use it instead of `ask` when the user talks to you by voice. One short question, answerable " +
			"in a few words; no markdown.",
		parameters: z.object({ question: z.string().describe("The question, plain spoken text") }),
		defaultInactive: true,
		async execute(_id, params, signal) {
			spoke = true;
			const id = crypto.randomUUID();
			const { promise, resolve } = Promise.withResolvers<string | null>();
			asks.set(id, resolve);
			const timer = setTimeout(() => resolve(null), ASK_TIMEOUT_MS);
			signal?.addEventListener("abort", () => resolve(null));
			if (!send({ type: "ask", id, text: params.question })) {
				clearTimeout(timer);
				asks.delete(id);
				return {
					content: [{ type: "text", text: "The speaker is not connected; the question was not asked." }],
					details: { answer: null },
				};
			}
			const text = await promise;
			clearTimeout(timer);
			asks.delete(id);
			return {
				content: [
					{
						type: "text",
						text:
							text === null
								? "No spoken answer. Go on with your best judgment, or ask again later."
								: `The user answered: ${text}`,
					},
				],
				details: { answer: text },
			};
		},
	});
}
