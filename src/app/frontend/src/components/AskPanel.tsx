import { type ReactNode, useEffect, useId, useRef, useState } from "react";
import useSWR from "swr";
import { type Agencies, type Answer, type AskStarted, getJson, postJson } from "../lib/api";

// Genie's steps, as the person waiting would put them
const PROGRESS: Record<string, string> = {
  SUBMITTED: "Reading the question…",
  FETCHING_METADATA: "Reading the question…",
  FILTERING_CONTEXT: "Reading the question…",
  ASKING_AI: "Writing the query…",
  PENDING_WAREHOUSE: "Starting the warehouse. The first question can take a minute…",
  EXECUTING_QUERY: "Running the query…",
};
const POLL_MS = 1500;
const GIVE_UP_MS = 180_000;

type Turn = { id: number; question: string; answer?: Answer; error?: string };

type AskProps = {
  links: Agencies;
  questions: string[]; // to start from
  label?: string;
  placeholder?: string;
};

/** "Ask a question" for the Now, replay and reliability views: the Genie space the agencies' view asks, under a
 * heading, with questions that fit the view. It stays out of the way when the app has no Genie space. */
export function AskSection({ questions, intro }: { questions: string[]; intro: string }) {
  const { data } = useSWR<Agencies>("/api/agencies", getJson, { revalidateOnFocus: false });
  const title = useId();
  if (!data?.genie) return null;
  return (
    <section className="ask-section" aria-labelledby={title}>
      <h2 id={title}>Ask a question</h2>
      <p className="ask-intro">{intro}</p>
      <Ask links={data} questions={questions} label="Your question" />
    </section>
  );
}

/** Questions to the Genie space, one conversation at a time: a follow-up keeps the conversation going. */
export function Ask({
  links,
  questions,
  label = "Ask about PUB's alerts, flood-prone land or how FloodSense did",
  placeholder = "Which areas had the most flash floods since 2022?",
}: AskProps) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [conversation, setConversation] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const alive = useRef(true);
  const input = useId();
  useEffect(
    () => () => {
      alive.current = false;
    },
    [],
  );

  if (!links.genie) {
    return <p className="state">This copy of the app has no Genie space set up.</p>;
  }

  async function ask(question: string) {
    const q = question.trim();
    if (q.length < 3 || busy) return;
    const id = Date.now();
    const update = (patch: Partial<Turn>) => {
      if (alive.current) setTurns((all) => all.map((t) => (t.id === id ? { ...t, ...patch } : t)));
    };
    setTurns((all) => [...all, { id, question: q }]);
    setDraft("");
    setBusy(true);
    try {
      const started = await postJson<AskStarted>("/api/ask", {
        question: q,
        conversation_id: conversation ?? undefined,
      });
      if (alive.current) setConversation(started.conversation_id);
      const deadline = Date.now() + GIVE_UP_MS;
      for (;;) {
        const answer = await getJson<Answer>(`/api/ask/${started.conversation_id}/${started.message_id}`);
        update({ answer });
        if (answer.done || !alive.current) break;
        if (Date.now() > deadline) throw new Error("Genie is taking too long. Try asking again.");
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      }
    } catch (e) {
      update({ error: e instanceof Error ? e.message : String(e) });
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  const last = turns[turns.length - 1];
  return (
    <section className="ask">
      <form
        className="ask-form"
        onSubmit={(e) => {
          e.preventDefault();
          void ask(draft);
        }}
      >
        <label htmlFor={input}>{conversation ? "Ask a follow-up" : label}</label>
        <div className="ask-row">
          <input
            id={input}
            type="text"
            value={draft}
            maxLength={400}
            autoComplete="off"
            placeholder={placeholder}
            onChange={(e) => setDraft(e.target.value)}
          />
          <button type="submit" disabled={busy || draft.trim().length < 3}>
            Ask
          </button>
        </div>
      </form>

      {turns.length === 0 ? <Suggestions title="Try one of these" questions={questions} onAsk={ask} busy={busy} /> : null}

      <ol className="thread">
        {turns.map((t) => (
          <li key={t.id} className="turn">
            <p className="turn-q">{t.question}</p>
            <TurnBody turn={t} />
            {t === last && t.answer?.status === "COMPLETED" && t.answer.suggested.length ? (
              <Suggestions title="Ask next" questions={t.answer.suggested} onAsk={ask} busy={busy} />
            ) : null}
          </li>
        ))}
      </ol>

      {turns.length ? (
        <button
          type="button"
          className="link"
          disabled={busy}
          onClick={() => {
            setTurns([]);
            setConversation(null);
          }}
        >
          Start again with a new question
        </button>
      ) : null}

      <p className="fine-print">
        Answers come from Genie, Databricks' AI assistant for data, which writes SQL over the tables behind the
        scoreboard. Check the SQL under each answer before you rely on it.{" "}
        <a href={links.genie.url} target="_blank" rel="noreferrer">
          Open the Genie space in Databricks
        </a>
        .
      </p>
    </section>
  );
}

function Suggestions(props: { title: string; questions: string[]; onAsk: (q: string) => void; busy: boolean }) {
  return (
    <div className="ask-examples">
      <p>{props.title}:</p>
      <ul>
        {props.questions.map((q) => (
          <li key={q}>
            <button type="button" disabled={props.busy} onClick={() => props.onAsk(q)}>
              {q}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function TurnBody({ turn }: { turn: Turn }) {
  const a = turn.answer;
  if (turn.error) {
    return (
      <p className="turn-error" role="alert">
        {turn.error}
      </p>
    );
  }
  if (!a || !a.done) {
    return (
      <p className="turn-wait" role="status">
        {PROGRESS[a?.status ?? "SUBMITTED"] ?? "Working on it…"}
      </p>
    );
  }
  if (a.status !== "COMPLETED") {
    return (
      <p className="turn-error" role="alert">
        Genie couldn't answer that{a.error ? ` (${a.error})` : ""}. Try asking another way.
      </p>
    );
  }
  return (
    <div className="turn-a">
      {a.text ? <Prose text={a.text} /> : null}
      {a.columns.length ? <ResultTable answer={a} /> : null}
      {a.sql ? (
        <details className="turn-sql">
          <summary>The SQL behind this answer</summary>
          {a.description ? <p>{a.description}</p> : null}
          <pre>
            <code>{a.sql}</code>
          </pre>
        </details>
      ) : null}
    </div>
  );
}

/** Genie's text: paragraphs, "- " lists and **bold**, built as elements, never as HTML. */
function Prose({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let items: string[] = [];
  const flush = () => {
    if (items.length) {
      blocks.push(
        <ul key={blocks.length}>
          {items.map((item, i) => (
            <li key={i}>{bold(item)}</li>
          ))}
        </ul>,
      );
      items = [];
    }
  };
  for (const line of text.split("\n")) {
    const item = line.match(/^\s*[-*]\s+(.*)$/);
    if (item) {
      items.push(item[1]);
      continue;
    }
    flush();
    if (line.trim()) blocks.push(<p key={blocks.length}>{bold(line)}</p>);
  }
  flush();
  return <div className="prose">{blocks}</div>;
}

function bold(line: string): ReactNode[] {
  return line.split(/\*\*(.+?)\*\*/g).map((part, i) => (i % 2 ? <strong key={i}>{part}</strong> : part));
}

function ResultTable({ answer }: { answer: Answer }) {
  const shown = answer.rows.length;
  return (
    <div className="result-wrap">
      <table className="leads result">
        <thead>
          <tr>
            {answer.columns.map((c) => (
              <th key={c} scope="col">
                {c.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {answer.rows.map((row, i) => (
            <tr key={i}>
              {row.map((value, j) => (
                <td key={j}>{cell(value)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {answer.row_count != null && answer.row_count > shown ? (
        <p className="fine-print">
          The first {shown} of {answer.row_count} rows.
        </p>
      ) : null}
    </div>
  );
}

/** A result cell as text: times to the minute (they're Singapore time), long decimals to 3 significant figures,
 * flags as yes or no, and nothing as a dash. */
function cell(value: string | null): string {
  if (value == null || value === "") return "–";
  const time = value.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/);
  if (time) return `${time[1]} ${time[2]}`;
  if (value === "true" || value === "false") return value === "true" ? "Yes" : "No";
  if (/^-?\d+\.\d{3,}(E-?\d+)?$/i.test(value)) return String(Number(Number(value).toPrecision(3)));
  return value;
}
