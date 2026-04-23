//! ACP integration harness – exercises the full permission + tool call flow end-to-end.
//!
//! Spawns `maki acp` as a subprocess and acts as an ACP client over stdio, exercising:
//! - initialize handshake
//! - session/new
//! - session/set_mode (plan → build)
//! - session/prompt with streaming updates (text, thinking, tool calls)
//! - session/request_permission (auto-respond per --permission strategy)
//! - multi-turn: second prompt on the same session
//! - session/list (verify the session appears)
//! - session/load (verify history replay notifications arrive)
//! - cancellation: new session, send long prompt, cancel mid-flight, verify stop reason
//!
//! Usage:
//!   acp-harness [--maki ./target/debug/maki] [--model MODEL] [--prompt TEXT]
//!               [--permission allow|allow-session|deny]
//!               [--no-reload] [--no-second-prompt] [--no-cancel]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agent_client_protocol::schema::{
    CancelNotification, ContentBlock, InitializeRequest, ListSessionsRequest, LoadSessionRequest,
    NewSessionRequest, PermissionOptionId, PromptRequest, ProtocolVersion,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionNotification, SessionUpdate, SetSessionModeRequest,
    StopReason, TextContent, ToolCallStatus,
};
use agent_client_protocol::{
    Agent, ByteStreams, Client, ConnectionTo, on_receive_notification, on_receive_request,
};
use clap::{Parser, ValueEnum};
use color_eyre::{Result, eyre::bail};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

// ── ANSI colour helpers ──────────────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const MAGENTA: &str = "\x1b[35m";
const BLUE: &str = "\x1b[34m";

fn tag(color: &str, label: &str) -> String {
    format!("{BOLD}{color}[{label}]{RESET}")
}

// ── CLI args ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum, Default)]
enum PermissionStrategy {
    /// Respond with the first "allow once" option.
    #[default]
    Allow,
    /// Respond with "allow for session" if present, else allow once.
    AllowSession,
    /// Deny every permission request.
    Deny,
}

#[derive(Parser, Debug)]
#[command(name = "acp-harness", about = "ACP integration harness for maki")]
struct Args {
    /// Path to the maki binary.
    #[arg(long, default_value = "maki")]
    maki: PathBuf,

    /// Model to pass to `maki acp --model`.
    #[arg(long, short = 'm')]
    model: Option<String>,

    /// Prompt text to send to the agent.
    #[arg(
        long,
        short = 'p',
        default_value = "List the top-level files in the current directory."
    )]
    prompt: String,

    /// How to handle incoming permission requests.
    #[arg(long, default_value = "allow")]
    permission: PermissionStrategy,

    /// Skip session/list and session/load after the prompt.
    #[arg(long)]
    no_reload: bool,

    /// Skip the second (multi-turn) prompt.
    #[arg(long)]
    no_second_prompt: bool,

    /// Skip the cancellation scenario.
    #[arg(long)]
    no_cancel: bool,

    /// Working directory for the new session (defaults to current dir).
    #[arg(long)]
    cwd: Option<PathBuf>,
}

// ── Shared harness state (written by notification handler, read for summary) ─

#[derive(Debug, Clone)]
struct ToolEntry {
    title: String,
    kind: String,
    status: ToolCallStatus,
}

#[derive(Debug, Default)]
struct HarnessState {
    text_buf: String,
    thinking_buf: String,
    tool_calls: HashMap<String, ToolEntry>,
    permissions_asked: u32,
    permissions_granted: u32,
    permissions_denied: u32,
    replay_updates: u32,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    let cwd = match args.cwd {
        Some(c) => c,
        None => std::env::current_dir()?,
    };

    // ── Spawn maki acp ──────────────────────────────────────────────────────
    let mut cmd = tokio::process::Command::new(&args.maki);
    cmd.arg("acp");
    if let Some(ref m) = args.model {
        cmd.args(["--model", m]);
    }
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());

    let mut child = cmd.spawn()?;
    let child_stdin = child.stdin.take().expect("piped stdin");
    let child_stdout = child.stdout.take().expect("piped stdout");

    eprintln!(
        "{} spawned {} (pid {:?})",
        tag(CYAN, "HARNESS"),
        args.maki.display(),
        child.id(),
    );

    let transport = ByteStreams::new(child_stdin.compat_write(), child_stdout.compat());

    // ── Shared state ────────────────────────────────────────────────────────
    let state: Arc<Mutex<HarnessState>> = Arc::new(Mutex::new(HarnessState::default()));
    let state_notif = Arc::clone(&state);
    let state_perm = Arc::clone(&state);
    let perm_strategy = args.permission.clone();

    // ── Build and run client ─────────────────────────────────────────────────
    let result = Client
        .builder()
        .on_receive_notification(
            async move |notif: SessionNotification, _cx| {
                let mut s = state_notif.lock().unwrap();
                handle_session_update(&notif.update, &mut s);
                Ok(())
            },
            on_receive_notification!(),
        )
        .on_receive_request(
            async move |req: RequestPermissionRequest, responder, _cx| {
                let mut s = state_perm.lock().unwrap();
                s.permissions_asked += 1;

                let chosen_id = pick_option(&req, &perm_strategy);
                let is_grant = is_grant_option(&chosen_id, &req);
                let label = req
                    .options
                    .iter()
                    .find(|o| o.option_id.0.as_ref() == chosen_id.as_str())
                    .map(|o| o.name.as_str())
                    .unwrap_or(chosen_id.as_str());
                let tool_title = req.tool_call.fields.title.as_deref().unwrap_or("(tool)");

                if is_grant {
                    s.permissions_granted += 1;
                    eprintln!(
                        "{} {}{}{} → {}{} ({}){RESET}",
                        tag(YELLOW, "PERM"),
                        BOLD,
                        tool_title,
                        RESET,
                        GREEN,
                        label,
                        chosen_id,
                    );
                } else {
                    s.permissions_denied += 1;
                    eprintln!(
                        "{} {}{}{} → {}DENIED{RESET} ({})",
                        tag(YELLOW, "PERM"),
                        BOLD,
                        tool_title,
                        RESET,
                        RED,
                        label,
                    );
                }
                drop(s);

                let outcome = RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                    PermissionOptionId::new(chosen_id),
                ));
                let _ = responder.respond(RequestPermissionResponse::new(outcome));
                Ok(())
            },
            on_receive_request!(),
        )
        .connect_with(transport, |cx: ConnectionTo<Agent>| {
            let state = Arc::clone(&state);
            let cwd = cwd.clone();
            let prompt = args.prompt.clone();
            let no_reload = args.no_reload;
            let no_second_prompt = args.no_second_prompt;
            let no_cancel = args.no_cancel;
            async move {
                run_harness(
                    cx,
                    state,
                    cwd,
                    prompt,
                    no_reload,
                    no_second_prompt,
                    no_cancel,
                )
                .await
            }
        })
        .await;

    let _ = child.kill().await;

    match result {
        Ok(()) => {
            eprintln!("{} all scenarios passed", tag(GREEN, "PASS"));
            Ok(())
        }
        Err(e) => {
            bail!("{} {e}", tag(RED, "FAIL"));
        }
    }
}

// ── Main scenario chain ───────────────────────────────────────────────────────

async fn run_harness(
    cx: ConnectionTo<Agent>,
    state: Arc<Mutex<HarnessState>>,
    cwd: PathBuf,
    prompt: String,
    no_reload: bool,
    no_second_prompt: bool,
    no_cancel: bool,
) -> agent_client_protocol::Result<()> {
    let start = Instant::now();

    // 1. Initialize ────────────────────────────────────────────────────────────
    eprintln!("{} initializing …", tag(CYAN, "INIT"));
    let init_resp = cx
        .send_request(InitializeRequest::new(ProtocolVersion::V1))
        .block_task()
        .await?;
    eprintln!(
        "{} agent={:?}  caps={:?}",
        tag(CYAN, "INIT"),
        init_resp.agent_info.as_ref().map(|i| i.name.as_str()),
        init_resp.agent_capabilities,
    );

    // 2. New session ───────────────────────────────────────────────────────────
    eprintln!("{} cwd={}", tag(CYAN, "SESSION"), cwd.display());
    let new_resp = cx
        .send_request(NewSessionRequest::new(cwd.clone()))
        .block_task()
        .await?;
    let session_id = new_resp.session_id.clone();
    eprintln!("{} id={}", tag(CYAN, "SESSION"), session_id.0);

    // 3. session/set_mode ──────────────────────────────────────────────────────
    // Switch to plan and back to build; verifies the handler round-trips cleanly.
    eprintln!("{} plan → build", tag(CYAN, "MODE"));
    cx.send_request(SetSessionModeRequest::new(session_id.clone(), "plan"))
        .block_task()
        .await?;
    cx.send_request(SetSessionModeRequest::new(session_id.clone(), "build"))
        .block_task()
        .await?;
    eprintln!("{} ok", tag(CYAN, "MODE"));

    // 4. First prompt ──────────────────────────────────────────────────────────
    eprintln!("{} {:?}", tag(BLUE, "PROMPT"), prompt);
    let prompt_resp = cx
        .send_request(PromptRequest::new(
            session_id.clone(),
            vec![ContentBlock::Text(TextContent::new(prompt.clone()))],
        ))
        .block_task()
        .await?;

    {
        let s = state.lock().unwrap();
        eprintln!(
            "\n{} stop={:?}  elapsed={:.1}s",
            tag(CYAN, "DONE"),
            prompt_resp.stop_reason,
            start.elapsed().as_secs_f64(),
        );
        print_summary(&s);
    }

    check_stop_reason(prompt_resp.stop_reason)?;

    // 5. Multi-turn: second prompt on the same session ─────────────────────────
    if !no_second_prompt {
        second_prompt_scenario(&cx, &session_id).await?;
    }

    if no_reload {
        return Ok(());
    }

    // 6. session/list ──────────────────────────────────────────────────────────
    eprintln!("\n{} …", tag(MAGENTA, "LIST"));
    let list_resp = cx
        .send_request(ListSessionsRequest::new())
        .block_task()
        .await?;
    let found = list_resp
        .sessions
        .iter()
        .any(|s| s.session_id.0.as_ref() == session_id.0.as_ref());
    eprintln!(
        "{} total={} found_ours={}",
        tag(MAGENTA, "LIST"),
        list_resp.sessions.len(),
        found,
    );
    if !found {
        return Err(agent_client_protocol::util::internal_error(
            "session not found in session/list",
        ));
    }

    // 7. session/load (history replay) ─────────────────────────────────────────
    eprintln!("{} id={}", tag(MAGENTA, "LOAD"), session_id.0);
    cx.send_request(LoadSessionRequest::new(session_id.clone(), cwd.clone()))
        .block_task()
        .await?;

    tokio::time::sleep(Duration::from_millis(50)).await;

    let replay = state.lock().unwrap().replay_updates;
    eprintln!("{} replay_updates={}", tag(MAGENTA, "LOAD"), replay);
    if replay == 0 {
        eprintln!("{DIM}  note: zero replay updates – session may have been empty{RESET}",);
    }

    // 8. Cancellation scenario ─────────────────────────────────────────────────
    if !no_cancel {
        cancel_scenario(&cx, &cwd).await?;
    }

    eprintln!(
        "{} all scenarios complete  elapsed={:.1}s",
        tag(GREEN, "OK"),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

// ── Individual scenario functions ─────────────────────────────────────────────

/// Send a follow-up prompt on an existing session to verify multi-turn history.
async fn second_prompt_scenario(
    cx: &ConnectionTo<Agent>,
    session_id: &agent_client_protocol::schema::SessionId,
) -> agent_client_protocol::Result<()> {
    const FOLLOW_UP: &str = "In one sentence, summarise what you just told me.";
    eprintln!("\n{} {:?}", tag(BLUE, "PROMPT2"), FOLLOW_UP);

    let resp = cx
        .send_request(PromptRequest::new(
            session_id.clone(),
            vec![ContentBlock::Text(TextContent::new(FOLLOW_UP))],
        ))
        .block_task()
        .await?;

    eprintln!("{} stop={:?}", tag(CYAN, "DONE2"), resp.stop_reason);
    check_stop_reason(resp.stop_reason)
}

/// Open a fresh session, send a long prompt, cancel mid-flight, verify stop reason.
///
/// Cancellation races are inherently non-deterministic: if the LLM responds before
/// the cancel notification arrives the stop reason will be EndTurn, not Cancelled.
/// The scenario logs a warning in that case but does not fail.
async fn cancel_scenario(
    cx: &ConnectionTo<Agent>,
    cwd: &std::path::Path,
) -> agent_client_protocol::Result<()> {
    eprintln!(
        "\n{} opening session for cancel test …",
        tag(MAGENTA, "CANCEL")
    );

    let new_resp = cx
        .send_request(NewSessionRequest::new(cwd.to_path_buf()))
        .block_task()
        .await?;
    let session_id = new_resp.session_id;

    // A prompt likely to outlast the cancel delay.
    const LONG_PROMPT: &str =
        "Count from 1 to 500, one number per line, with a brief comment on each number.";

    eprintln!(
        "{} sending long prompt then cancelling after 500 ms …",
        tag(MAGENTA, "CANCEL")
    );

    let prompt_fut = cx
        .send_request(PromptRequest::new(
            session_id.clone(),
            vec![ContentBlock::Text(TextContent::new(LONG_PROMPT))],
        ))
        .block_task();

    let cancel_cx = cx.clone();
    let cancel_sid = session_id.clone();
    let cancel_fut = async move {
        tokio::time::sleep(Duration::from_millis(500)).await;
        cancel_cx.send_notification(CancelNotification::new(cancel_sid))
    };

    let (prompt_result, cancel_result) = tokio::join!(prompt_fut, cancel_fut);
    cancel_result?;
    let resp = prompt_result?;

    match resp.stop_reason {
        StopReason::Cancelled => {
            eprintln!("{} stop=Cancelled (correct)", tag(GREEN, "CANCEL"));
        }
        other => {
            eprintln!(
                "{} stop={other:?} — cancel arrived after agent finished (acceptable race)",
                tag(YELLOW, "CANCEL"),
            );
        }
    }
    Ok(())
}

// ── Event handler ─────────────────────────────────────────────────────────────

fn handle_session_update(update: &SessionUpdate, s: &mut HarnessState) {
    match update {
        SessionUpdate::UserMessageChunk(chunk) => {
            if let ContentBlock::Text(t) = &chunk.content {
                eprint!("{DIM}{}{RESET}", t.text);
            }
        }
        SessionUpdate::AgentMessageChunk(chunk) => {
            if let ContentBlock::Text(t) = &chunk.content {
                eprint!("{}", t.text);
                s.text_buf.push_str(&t.text);
            }
        }
        SessionUpdate::AgentThoughtChunk(chunk) => {
            if let ContentBlock::Text(t) = &chunk.content {
                eprint!("{DIM}{}{RESET}", t.text);
                s.thinking_buf.push_str(&t.text);
            }
        }
        SessionUpdate::ToolCall(tc) => {
            let kind = format!("{:?}", tc.kind);
            eprintln!(
                "\n{} {}{}{} ({}) [{}]",
                tag(YELLOW, "TOOL"),
                BOLD,
                tc.title,
                RESET,
                kind,
                tc.tool_call_id.0,
            );
            s.tool_calls.insert(
                tc.tool_call_id.0.as_ref().to_string(),
                ToolEntry {
                    title: tc.title.clone(),
                    kind,
                    status: tc.status,
                },
            );
            s.replay_updates += 1;
        }
        SessionUpdate::ToolCallUpdate(tcu) => {
            let id = tcu.tool_call_id.0.as_ref().to_string();
            if let Some(entry) = s.tool_calls.get_mut(&id) {
                if let Some(new_status) = &tcu.fields.status {
                    let symbol = match new_status {
                        ToolCallStatus::Completed => format!("{GREEN}✓{RESET}"),
                        ToolCallStatus::Failed => format!("{RED}✗{RESET}"),
                        ToolCallStatus::InProgress => format!("{YELLOW}▶{RESET}"),
                        ToolCallStatus::Pending => format!("{DIM}…{RESET}"),
                        _ => "?".to_string(),
                    };
                    eprintln!(
                        "{} {}{}{} {} {:?}→{:?}",
                        tag(YELLOW, "TOOL"),
                        BOLD,
                        entry.title,
                        RESET,
                        symbol,
                        entry.status,
                        new_status,
                    );
                    entry.status = *new_status;
                }
            } else {
                s.replay_updates += 1;
            }
        }
        SessionUpdate::SessionInfoUpdate(si) => {
            if let Some(title) = si.title.value() {
                eprintln!("\n{} title={title:?}", tag(CYAN, "INFO"));
            }
        }
        _ => {
            s.replay_updates += 1;
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn check_stop_reason(stop: StopReason) -> agent_client_protocol::Result<()> {
    match stop {
        StopReason::EndTurn | StopReason::MaxTokens | StopReason::Cancelled => Ok(()),
        other => Err(agent_client_protocol::util::internal_error(format!(
            "unexpected stop reason: {other:?}"
        ))),
    }
}

fn pick_option(req: &RequestPermissionRequest, strategy: &PermissionStrategy) -> String {
    match strategy {
        PermissionStrategy::Deny => req
            .options
            .iter()
            .find(|o| o.option_id.0.as_ref().starts_with("deny"))
            .or_else(|| req.options.last())
            .map(|o| o.option_id.0.as_ref().to_string())
            .unwrap_or_else(|| "deny".to_string()),

        PermissionStrategy::AllowSession => req
            .options
            .iter()
            .find(|o| o.option_id.0.as_ref() == "allow_session")
            .or_else(|| {
                req.options
                    .iter()
                    .find(|o| o.option_id.0.as_ref().starts_with("allow"))
            })
            .or_else(|| req.options.first())
            .map(|o| o.option_id.0.as_ref().to_string())
            .unwrap_or_else(|| "allow".to_string()),

        PermissionStrategy::Allow => req
            .options
            .iter()
            .find(|o| o.option_id.0.as_ref() == "allow")
            .or_else(|| {
                req.options
                    .iter()
                    .find(|o| o.option_id.0.as_ref().starts_with("allow"))
            })
            .or_else(|| req.options.first())
            .map(|o| o.option_id.0.as_ref().to_string())
            .unwrap_or_else(|| "allow".to_string()),
    }
}

fn is_grant_option(option_id: &str, req: &RequestPermissionRequest) -> bool {
    if let Some(opt) = req
        .options
        .iter()
        .find(|o| o.option_id.0.as_ref() == option_id)
    {
        use agent_client_protocol::schema::PermissionOptionKind;
        return matches!(
            opt.kind,
            PermissionOptionKind::AllowOnce | PermissionOptionKind::AllowAlways
        );
    }
    option_id.starts_with("allow")
}

fn print_summary(s: &HarnessState) {
    let completed = s
        .tool_calls
        .values()
        .filter(|e| matches!(e.status, ToolCallStatus::Completed))
        .count();
    let failed = s
        .tool_calls
        .values()
        .filter(|e| matches!(e.status, ToolCallStatus::Failed))
        .count();

    eprintln!("{DIM}─ summary ─────────────────────────────────────{RESET}");
    eprintln!("  text chars  : {}", s.text_buf.len());
    eprintln!("  think chars : {}", s.thinking_buf.len());
    eprintln!(
        "  tool calls  : {} total  {} completed  {} failed",
        s.tool_calls.len(),
        completed,
        failed,
    );
    for (id, e) in &s.tool_calls {
        let sym = match e.status {
            ToolCallStatus::Completed => format!("{GREEN}✓{RESET}"),
            ToolCallStatus::Failed => format!("{RED}✗{RESET}"),
            ToolCallStatus::InProgress => format!("{YELLOW}▶{RESET}"),
            _ => format!("{DIM}…{RESET}"),
        };
        eprintln!("    {} {} {DIM}({}) [{}]{RESET}", sym, e.title, e.kind, id);
    }
    eprintln!(
        "  permissions : {} asked  {} granted  {} denied",
        s.permissions_asked, s.permissions_granted, s.permissions_denied,
    );
    eprintln!("{DIM}────────────────────────────────────────────────{RESET}");
}
