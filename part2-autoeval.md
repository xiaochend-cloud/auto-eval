---
marp: true
theme: default
paginate: true
title: "Part II: Automatic Agent Evaluation (AutoEval)"
---

# Part II: Automatic Agent Evaluation (AutoEval)
### Agent Safety · one grader, three runtimes; measure *where* it failed and *whether* it was unsafe

---

## Building Reliable Agent Evaluation  ⭐ NARRATE

- **Thesis.** Outcome-only benchmarks reveal *that* an agent failed, not **where**, and not **whether it acted unsafely**. This work contributes a **repurposable grading pipeline [MINE]** that begins to address both.
- **Scope of contribution [MINE].** The **automated grading infrastructure**, off-device transcript reconstruction, LLM-judge customization, the Streamlit UI, and the task suite.
- **Attribution.** The **agent runtime and PEVA governance are the team's [TEAM]**; the **evaluation layer** is the contribution here.

🎤 **SAY:** "This is agent safety. Outcome-only eval misses where an agent went wrong and whether it acted unsafely. The contribution here is the grading pipeline that starts to measure both."

---

## Agent Evaluation: Why It Matters, Why It's Hard  ⭐ NARRATE

::: cols
Why it matters
- **First: does it do the task correctly?** Today's agent eval, including this pipeline, is built for **accuracy verification**, and reliability is far from solved: human vs agent is **GAIA 92 vs 15**, **WebArena 78 vs 14** (SWE-bench, τ-bench climb fast; the gap holds).
- **Acting makes being wrong costlier.** A wrong tool call carries an **irreversible side effect** (email sent, money moved, data deleted). Side-effect verification is the **frontier**, barely built; this work names it and gives a taxonomy, not a solution.
- **So evaluation is the bottleneck, not capability.** As in PID, the missing piece is honest evaluation, not a cleverer model.
+++
Why it's hard
- **Agents derail mid-task.** Wrong tool, collapsing reasoning chain, and you **can't localize which step failed.**
- **Outcomes aren't enough.** A lucky shortcut passes; good tool-use that ends wrong fails, neither is diagnosed.
- **Irreversible, and manual eval doesn't scale.** You can't reset between runs, and hand-grading every trajectory is neither reproducible nor scalable.
:::

🎤 **SAY:** "Two things. First, can the agent do the task at all? That is what evaluation is built for today, and agents are still far below humans, ninety-two versus fifteen on GAIA. Second, because agents now act, being wrong is costlier, but side-effect verification is barely built, so I name it as the frontier. And it is hard because agents derail mid-task, outcomes hide where they failed, and manual grading does not scale."

---

## Current Evaluation Limitations (literature)  📄 DOCUMENT

- **Outcome-dominant.** Across 9 surveyed benchmarks, most grade the **final state** (WebArena, Zhou et al., 2024; AndroidWorld, Rawles et al., 2025; GAIA; AgentBench). Per-step credit assignment is **rare**, only AgentBoard progress-rate and AgentBench per-step attempt it.
- **Side-effect verification** exists only in **heavy sandboxes / emulators** (AndroidWorld is an emulator); real-device verification is essentially absent.
- **LLM judges are widely used but under-audited.** ToolEval reports **87.1% pass agreement** with humans (Qin et al., 2023); ReasoningBank's self-judge is **72.7% vs ground truth** (Ouyang et al., 2025).
- **Agents remain far from reliable.** Human vs agent success: GAIA **92 vs 15**, WebArena **78 vs 14**, AndroidWorld **80 vs 31**.
- **Gap we target:** portable grading + the seed of **process-based** evaluation + honest judge auditing.

🎤 **SAY:** "This work surveys nine agent-eval benchmarks. They mostly grade the final outcome, verify side effects only in sandboxes, and use LLM judges that are 70 to 87 percent aligned with humans but rarely audited. And the human-agent gap is still huge. That's exactly the space this pipeline sits in."

---

## Related Work: How the Field Grades Agents  📄 DOCUMENT

The 9 surveyed benchmarks span one axis, **what counts as ground truth:**

| Ground truth | Method | Papers | Side effects? | Per-step? |
|---|---|---|---|---|
| structural | AST sub-tree match | Gorilla (Patil et al., 2023) | ✗ (never executes) | ✗ |
| final answer | quasi-exact-match | GAIA (2023) | ✗ | ✗ |
| LLM-judge | ToolEval (≈87% human agr.) | ToolLLM (2023) | ✗ | weak |
| **state / side-effect** | DB/device inspection | WebArena, AndroidWorld, AgentBench | ✓ (sandbox/emulator) | partial |
| per-step process | progress rate | AgentBoard, AgentBench | ✗ | **yes (rare)** |

- **Two takeaways.** (1) The field moved from matching text → checking outcomes; **per-step "where did it fail" is still rare.** (2) LLM judges are ubiquitous but **under-audited** (ReasoningBank's judge ≈ 72.7% accurate).
- **This work's position:** a **portable** grader (transcript-as-contract), the **seed of process-based** evaluation, and an **audited** judge, exactly the field's open gaps.

🎤 **SAY:** "Nine benchmarks, one axis: what you compare the agent against. The field checks the final outcome but rarely *where* it failed, and uses LLM judges without auditing them. This pipeline is portable across runtimes, starts to diagnose the failing step, and audits the judge, the three open gaps."

---

## Goal + Pipeline Overview  ⭐ NARRATE

- **Goal.** **Automatic, reproducible** grading that returns both a **score** and a **diagnosis.**
- **The grader is a staged pipeline [MINE].** Task-typing → trajectory normalization → a **regime evaluator** (objective / constraint+utility / judge-based) → execution-trace analysis → latency analysis → an MDP view → a weighted-score presenter. Deterministic checks and LLM-judge stages are kept separate.
- **Task-typed scoring.** A task's family selects a scoring regime with fixed weights (objective = outcome 0.70 / tool-use 0.30; hybrid = outcome 0.30 / constraint 0.30 / preference 0.20 / tool-use 0.20), so scoring is consistent rather than ad hoc.
- **Three runtimes, one grader.** OpenClaw (CLI) · a local mini-agent · **KClaw on a real Galaxy S24**, all emit the same transcript contract.

![grader pipeline: eight stages from lib_evaluator.evaluate_episode](slides-honing/figures/grader-pipeline.png)

🎤 **SAY:** "Transcript in, score and reasoning out. The grader is a staged pipeline: task-typing picks the scoring regime, deterministic checks and the LLM judge run separately, and a trace analyzer flags where it failed. The same grader runs three different agents."

---

## Transcript-as-Contract  ⭐ NARRATE

- **The key design.** Every runtime emits the **same JSONL event stream**: `{"type":"message","message":{role, content[]}}`, roles `user / assistant / toolResult`, blocks `text | toolCall{name, arguments, id}`. The grader consumes only this contract, so it **never changes per runtime.**
- **Provider-agnostic by construction.** Anthropic `tool_use` blocks are translated to the canonical `toolCall` shape at the adapter boundary, so the judge and checks stay provider-independent.
- **Normalization.** `TrajectoryNormalizer` walks the stream into an `Episode` of steps, attaching each `toolResult` to its preceding call and recovering per-tool durations from the raw log.
- **Stress test [MINE].** The **OpenClaw** route was added *after* the pipeline was built around the local agent, **without touching the grader.** It passed, evidence the contract holds.

🎤 **SAY:** "The whole platform rests on one idea: the transcript is a fixed contract, three roles and a canonical tool-call block. A new agent needs just one adapter. This was validated by adding the OpenClaw route without touching the grader."

---

## Reconstructing a Transcript Off a Real Phone  📄 DOCUMENT  · [MINE]

- **The problem.** KClaw runs on a physical S24; there is no clean transcript to grade.
- **The adapter (`lib_android.py`) [MINE].** All I/O over `adb shell`. **Inject** the prompt by writing a `user` row into the on-device **SQLite** `messages` table (`kclaw.db`) exactly as the UI does → **poll** that table every 3s for the bot reply (`is_bot_message=1`) → **read tool events** by `tail`-ing the on-device `agent.log` (`[ToolDispatcher]`: `Executing tool` / `Tool result` / `Tool completed`) → **merge both streams by timestamp** → emit the grader's JSONL.
- **Robust injection.** SQL is piped to `sqlite3` over stdin so the Android shell cannot mangle values (e.g. `$800` stays `$800`).
- **Honest limit.** Two **asynchronous** streams (DB messages, log tool events) are correlated **heuristically by timestamp**, and parsing is coupled to the `agent.log` line format `[VERIFY: confirm the current honest limitation]`.

🎤 **SAY:** "To grade a real phone agent, the transcript is reconstructed over ADB: inject the prompt as a database row the way the UI does, poll the database for the reply, and read tool calls from the on-device agent log, then merge the two streams by timestamp. The honest gap is that the merge is heuristic and tied to the log format."

---

## LLM Judge: design + reliability  ⭐ NARRATE

- **Input / output.** A summarized transcript + task rubric → a **single JSON** object `{"scores":{…},"total":…,"notes":…}`, where `total` is constrained to the **arithmetic mean** of the criterion scores. Tool use is forbidden in the judge call.
- **Where it runs.** `claude-opus-4-5` (180s timeout) in the standalone grader, used for the **local and OpenClaw** modes. *(The KClaw device run was graded by deterministic autograders, not this judge.)*
- **Scoring discipline.** An early *strict* anchor ("reserve 1.0 for excellent") under-scored good work; the current design **starts at 1.0** and requires a **stated reason for every deduction** (see next slide).
- **Why so defensive.** LLM-as-judge carries known biases (position, verbosity, self-preference; Zheng et al., 2023) and returns unreliable JSON. Parsing is **3-pass**: `json.loads` → escape stray control characters and retry → **regex field extraction by key** (structure-immune); failures are dumped to a temp file for audit.

🎤 **SAY:** "The judge returns one JSON object against a weighted rubric, with an anchor so 'average' is 0.6, not 1.0. Because judges are biased and their JSON is unreliable, the parser has three fallbacks and logs anything it can't parse."

---

## LLM Judge: Calibrating a Conservative Judge  ⭐ NARRATE

- **The problem: LLM judges are conservative.** Under a plain "be strict" instruction the judge **under-scores work that was actually done well**, docking points it cannot justify.
- **The fix: force justified deductions.** The judge starts at **1.0**, treats a perfect score as correct when criteria are met, and must give a **specific reason for every deduction** (a required `deduction_reasons` field; per-step impacts sum to the shortfall).
- **The result, on a well-done task.** The trip-planning agent completed all components under budget. The calibrated judge scored **0.823**, deducting only **−0.15** (agent proposed **June 3**, a *reasoned* deviation outside the June 4–6 window) and **−0.05** (research-only tools), each with a stated reason, not a reflexive low score.

![calibrated LLM judge verdict on a planning task](slides-honing/figures/judge-demo.png)

🎤 **SAY:** "LLM judges tend to be harsh, they dock points they can't defend, so they under-score good work. My fix was to make the judge start at a perfect score and justify every deduction. On this trip-planning task the agent did well, and the calibrated judge scored it 0.82, taking off only for the June-3 date and the missing booking tools, each with a written reason. That is a judge you can actually trust and audit."

---

## Result: End-to-End on a Real Phone, and the Honest Limits  ⭐ NARRATE

- **The pipeline runs end-to-end on 49 real device tasks (physical S24) [MINE].** Transcript reconstruction → staged grading → per-step diagnosis, on **real hardware, not an emulator.** That the system runs on a live phone at all is the result.
- **Autograders handle the easy cases, and they are not enough.** Audited against human judgment they agree only **~74.5%** (35/47) `[VERIFY]`, and their false positives **miss real side effects**, they check the tool *call*, not whether the SMS actually fired.
- **That limit is the argument, not a footnote.** It is exactly why the **calibrated LLM judge** and **per-step diagnosis** exist, and why **real side-effect verification** is the next step. Don't trust a grader you haven't audited.
- **Attribution.** PEVA is the team's **[TEAM]**; the **pipeline, real-device reconstruction, and judge calibration** are **[MINE]**.

🎤 **SAY:** "The result I'd highlight isn't a benchmark score, it's that the whole pipeline runs end-to-end on a real phone, forty-nine tasks, reconstruction to diagnosis. And I'm honest about the limits: deterministic autograders agree with humans only about seventy-five percent, and their false positives miss real side effects. That gap is exactly why the calibrated judge and side-effect verification matter."

---

## Task Suite: 49 Real-Device Tasks by Side-Effect Tier  📄 DOCUMENT

- **Organized by the safety-relevant axis, side-effect tier**, not by app: **22 read-only**, **21 local-persistent**, **6 external / irreversible**. The 6 external tasks (SMS, email, Telegram, call, booking) are where a **user-confirmation gate** and real **side-effect verification** matter most.

![49 KClaw tasks organized by side-effect tier](slides-honing/figures/task-taxonomy.png)

🎤 **SAY:** "The full suite is forty-nine real device tasks. I organize them not by app but by side-effect tier, because that is what matters for safety: twenty-two are read-only, twenty-one change reversible on-device state, and six are external and irreversible. Those six are exactly where you want a confirmation gate."

---

## AutoEval Lessons + Common Theme  ⭐ NARRATE

- **LLM judges fail in diagnosable ways**, audit them; measure agreement; engineer the context.
- **Beyond pass/fail.** The evaluator already flags a **first-wrong-step / failure-stage / root-cause**, the seed of **process-based** evaluation. **Next:** real side-effect verification.
- **The spine.** **PID prevents · AutoEval measures**, the basis for **trustworthy LLM systems.** Reliable evaluation infrastructure is the bridge from **model security** to **agent safety.**

🎤 **SAY:** "Judges fail in ways you can diagnose, so audit them. And the evaluator is starting to say where an agent failed, not just whether. Prevent, measure, that's the through-line of both projects."
