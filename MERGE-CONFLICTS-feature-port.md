# Merge Conflict Summary — Feature Port onto internal `main`

**Goal:** Port the KClaw screen-understanding + note-taking feature from
`KClaw-dogfooding` (`feature/memo`, 158 commits) onto
`kclaw-dogfooding-internal` (`main`), as a clean feature-only merge — not a
raw 158-commit rebase.

**Method:** 3-way merge, base = shared root `06b610ea` (the common ancestor of
both repos), layering our feature (`base → feature/memo`) onto internal `main`
(the target). Counts below are **actual conflict regions**, not diff hunks.

---

## Totals

- **25 conflict regions across 8 files.**
- **39 pure-new files add zero conflicts** (internal has none of them — drop in verbatim).
- **One file (`KClawInputActivity.kt`) holds 14 of the 25** conflicts. The rest are near-mechanical.

---

## Per-file breakdown

| File | Conflicts | Nature | Difficulty |
|---|---:|---|---|
| **AndroidManifest.xml** | 0 | Clean auto-merge — our service/activity/permission registrations don't collide | ✅ None |
| **KClawAccessibilityService.kt** | 1 | Import block only: union internal's `Flags`+`KClawInputActivity` with our `AccessibilityWindowInfo`. All 195 lines of our onKeyEvent/openMemo/volume logic merged cleanly | 🟢 Trivial |
| **ToolRegistry.kt** | 1 | Both added a ToolDef at the same spot — internal's `send_feedback` vs our `list_memos`. Keep both | 🟢 Trivial |
| **AppDatabase.kt** | 2 | (a) `version` 20 vs 21 → take 21; (b) migration list → union = internal's `12_13…19_20` chain + our `MIGRATION_20_21`. `MemoEntity`/`memoDao()` merged clean | 🟢 Trivial |
| **AgentForegroundService.kt** | 2 | (a) import block → union (internal superset); (b) adjacency around `ConsolidationWorker…Provider` / dogfood-summary block — keep internal's newer code, graft our `listMemos = { db.memoDao()… }` config in | 🟡 Moderate |
| **ChatHistoryMessageMapper.kt** | 1 | **add/add** — both forks created this file independently. Reconcile our `ScreenInsightPrompt` marker-strip + attachment mapping against internal's attachment mapping | 🟡 Moderate |
| **ToolDispatcher.kt** | 4 | 2 conflicts are import-order unions (trivial); 2 are `TOOL_NAMES`/policy-list insertions — union the entries, ensure `list_memos` + our `read_memory` land, **watch dup `create_contact`/`delete_contact`** appearing on both sides | 🟡 Moderate |
| **KClawInputActivity.kt** | 14 | The entangled one — 1347 our lines vs 1669 internal lines. Our attachment-picker UI + "back-from-Memo" drawer interleaved with internal's newer brief/memory. Keep internal's brief/memory, graft only our attachment/memo-drawer parts | 🔴 Hard |

---

## Shape of the work

- **13 of 25 conflicts are mechanical** — import unions, a version bump, additive list entries. Minutes to resolve.
- **~4 conflicts are moderate** — `ToolDispatcher` list membership, `AgentForegroundService` config insertion, `ChatHistoryMessageMapper` add/add reconciliation.
- **14 conflicts (56%) live in one file** — `KClawInputActivity.kt`. This is where the real judgment goes: the only file where our feature genuinely tangles with internal's independently-evolved brief/memory code.

---

## Two things that make this cheaper than it looks

- **`build.gradle.kts` is OUT of the merge** — our diff there is dogfood Firestore config, not the feature. Take internal's version. (This removes the 3 conflicts the earlier attempt hit.)
- **Attachment substrate already exists in internal** — `ChatImageAttachment`, `ChatFileAttachment`, `PendingChatAttachment`, `AttachmentConstraints`, `ChatImageAttachmentDecoder`, `PickedAttachmentMime` are all already on `main`. So `ChatHistoryMessageMapper.kt` and `KClawInputActivity.kt` reconcile against an existing API rather than porting the whole subsystem.

---

## The 39 pure-new files (zero conflict — `git checkout ourdev/feature/memo -- <file>`)

| Group | Files |
|---|---|
| **Memo domain** (`app/.../memo/`) | MemoAnchor, MemoCaptureController, MemoConsolidator, MemoLinkResolver, MemoListItem, MemoMarkdownExporter, MemoNotifier, MemoRepository, MemoTags |
| **Memo data** (`agent-core/.../db/`) | MemoDao, MemoEntity |
| **Memo UI** | MemoActivity, MemoChips, MemoDetailActivity, MemoAdapter + 4 layouts (activity_memo, activity_memo_detail, item_memo_card, item_memo_header) + ic_keyboard drawable |
| **Screen-insight / voice** (`app/.../services/`) | AssistStructureReader, KClawRecognitionService, KClawVoiceInteraction, KClawVoiceSessionService, MemoOverlay, OrbView, OverlayConversationController, ScreenInsightOverlay, ScreenInsightPrompt, WaveformView + recognition_service.xml, voice_interaction.xml |
| **Attachments / audio / misc** | SpeechInputManager, AttachPickerActivity, ImageAttachmentEncoder, memo-operations.md (skill), schemas/…/21.json, docs-shared/screen-insight.md, OverlayConversationControllerTest |

Plus test companion to port with its subject: `ChatHistoryMessageMapperTest.kt`.

---

## Files explicitly NOT merged (take internal's version)

- `build.gradle.kts` — our diff is dogfood Firestore buildConfig + a lockdown test helper, not the feature.
- `KClawNotificationListener.kt` — our add is `redactedCacheLog` (security/logging hygiene); internal already has it.
- `HttpCategoryAnnotationLockdownTest.kt` — false match, no feature content.
- 4 byte-identical wiring files: `AgentWatchdogWorker`, `BridgeConfirmationGrantProofStore`, `BridgeForegroundService`, `NotificationListenerStatus`.

---

## Recommended merge sequence (Phase 2)

1. Branch off internal `main` (fresh, or rebuild `feature/screen-understanding-note-taking`).
2. `git checkout ourdev/feature/memo -- <the 39 files>` (verbatim, zero conflict).
3. Hand-merge the 7 easy/moderate wiring files first (Manifest → Accessibility → ToolRegistry → AppDatabase → AgentForegroundService → ChatHistoryMessageMapper → ToolDispatcher).
4. Resolve `KClawInputActivity.kt` last, with review — the only genuinely hard file.
5. `./gradlew :app:assembleSystemDebug` → fix compile gaps → run memo/screen-insight unit tests → deploy.
6. One commit per self-contained file (not a giant batch).

**Single risk to watch:** `KClawInputActivity.kt` — the only file where feature and internal's newer brief/memory truly interleave. Everything else is additive or superset-graft.

---

## Reference: DB version alignment (why AppDatabase is trivial)

- Internal `main` is at `@Database(version = 20)`, latest exported schema `20.json`.
- Our `feature/memo` is at `version = 21`, adding `MIGRATION_20_21` (CREATE TABLE memos) + `21.json`.
- Our memo migration lands **cleanly on top of internal's chain with no renumbering** — a lucky, clean seam.
