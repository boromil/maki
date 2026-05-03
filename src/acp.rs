#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    AgentCapabilities, AuthenticateRequest, AuthenticateResponse, CancelNotification, ContentBlock,
    ContentChunk, Implementation, InitializeRequest, InitializeResponse, ListSessionsRequest,
    ListSessionsResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
    NewSessionResponse, PermissionOption, PermissionOptionKind, PromptRequest, PromptResponse,
    RequestPermissionOutcome, RequestPermissionRequest, SessionCapabilities, SessionId,
    SessionInfo, SessionInfoUpdate, SessionListCapabilities, SessionMode, SessionModeState,
    SessionNotification, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse, StopReason,
    TextContent, ToolCall, ToolCallId, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
    ToolKind,
};
use agent_client_protocol::{
    ByteStreams, Client, ConnectionTo, Dispatch, on_receive_dispatch, on_receive_notification,
    on_receive_request, role::acp::Agent,
};
use maki_agent::tools::{
    BASH_TOOL_NAME, BATCH_TOOL_NAME, CODE_EXECUTION_TOOL_NAME, EDIT_TOOL_NAME, GLOB_TOOL_NAME,
    GREP_TOOL_NAME, MULTIEDIT_TOOL_NAME, READ_TOOL_NAME, WRITE_TOOL_NAME,
};
use maki_providers::ContentBlock as ProviderBlock;
use maki_providers::Role as ProviderRole;

use color_eyre::Result;
use maki_agent::tools::{
    DescriptionContext, FileReadTracker, QUESTION_TOOL_NAME, ToolFilter, ToolRegistry,
};
use maki_agent::{
    Agent as MakiAgent, AgentConfig, AgentEvent, AgentInput, AgentMode, AgentParams,
    AgentRunParams, CancelToken, CancelTrigger, Envelope, EventSender, History, PermissionsConfig,
    agent, template,
};
use maki_config::ToolOutputLines;
use maki_providers::model::Model;
use maki_providers::provider;
use maki_providers::{Message, StopReason as MakiStopReason, Timeouts, TokenUsage};
use maki_storage::StateDir;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing::error;
use uuid::Uuid;

type AcpSession = maki_storage::sessions::Session<Message, TokenUsage, ()>;

const SESSION_PAGE_SIZE: usize = 50;
const PLAN_FILENAME: &str = "PLAN.md";

struct SessionState {
    history: Vec<Message>,
    cwd: PathBuf,
    cancel: Option<CancelTrigger>,
    mode: AgentMode,
}

type Sessions = Arc<Mutex<HashMap<String, SessionState>>>;

pub async fn run(
    model: Model,
    config: AgentConfig,
    permissions_config: PermissionsConfig,
    timeouts: Timeouts,
    storage: StateDir,
) -> Result<()> {
    let sessions: Sessions = Arc::new(Mutex::new(HashMap::new()));
    let storage: Arc<StateDir> = Arc::new(storage);

    let sessions_new = Arc::clone(&sessions);
    let sessions_load = Arc::clone(&sessions);
    let sessions_prompt = Arc::clone(&sessions);
    let sessions_cancel = Arc::clone(&sessions);
    let sessions_set_mode = Arc::clone(&sessions);

    let storage_new = Arc::clone(&storage);
    let storage_load = Arc::clone(&storage);
    let storage_list = Arc::clone(&storage);
    let storage_prompt = Arc::clone(&storage);

    let model_new = model.clone();
    let model_prompt = model.clone();
    let config_prompt = config.clone();
    let perms_prompt = permissions_config.clone();
    let timeouts_prompt = timeouts;

    Agent
        .builder()
        .name("maki")
        .on_receive_request(
            async move |req: InitializeRequest, responder, _cx| {
                responder.respond(
                    InitializeResponse::new(req.protocol_version)
                        .agent_capabilities(
                            AgentCapabilities::new()
                                .load_session(true)
                                .session_capabilities(
                                    SessionCapabilities::new().list(SessionListCapabilities::new()),
                                ),
                        )
                        .agent_info(Implementation::new("maki", env!("CARGO_PKG_VERSION"))),
                )
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: NewSessionRequest, responder, _cx| {
                let session_id = Uuid::new_v4().to_string();
                let cwd_str = req.cwd.to_string_lossy().into_owned();

                let mut acp_session = AcpSession::new(&model_new.id, &cwd_str);
                acp_session.id = session_id.clone();
                let _ = acp_session.save(&storage_new);

                sessions_new.lock().unwrap().insert(
                    session_id.clone(),
                    SessionState {
                        history: Vec::new(),
                        cwd: req.cwd.clone(),
                        cancel: None,
                        mode: AgentMode::Build,
                    },
                );
                responder.respond(
                    NewSessionResponse::new(session_id).modes(make_mode_state(&AgentMode::Build)),
                )
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: LoadSessionRequest, responder, cx| {
                let id = req.session_id.0.to_string();
                match AcpSession::load(&id, &storage_load) {
                    Ok(session) => {
                        let mode = stored_mode_to_agent(&session.meta.mode, &req.cwd);
                        let mode_state = make_mode_state(&mode);
                        if let Err(e) = replay_history(&session.messages, &req.session_id, &cx) {
                            error!(error = %e, "ACP: session/load replay failed");
                        }
                        sessions_load.lock().unwrap().insert(
                            id,
                            SessionState {
                                history: session.messages,
                                cwd: req.cwd,
                                cancel: None,
                                mode,
                            },
                        );
                        responder.respond(LoadSessionResponse::new().modes(mode_state))
                    }
                    Err(_) => responder.respond_with_error(
                        agent_client_protocol::util::internal_error("session not found"),
                    ),
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: ListSessionsRequest, responder, _cx| {
                let summaries = match &req.cwd {
                    Some(cwd) => AcpSession::list(&cwd.to_string_lossy(), &storage_list),
                    None => AcpSession::list_all(&storage_list),
                };
                match summaries {
                    Ok(list) => {
                        let start = req
                            .cursor
                            .as_deref()
                            .and_then(decode_cursor)
                            .and_then(|(cursor_ts, cursor_id)| {
                                list.iter().position(|s| {
                                    s.updated_at < cursor_ts
                                        || (s.updated_at == cursor_ts && s.id.as_str() > cursor_id)
                                })
                            })
                            .unwrap_or(0);

                        let page = &list[start..];
                        let page = &page[..page.len().min(SESSION_PAGE_SIZE)];
                        let next_cursor = if start + SESSION_PAGE_SIZE < list.len() {
                            page.last().map(|s| encode_cursor(s.updated_at, &s.id))
                        } else {
                            None
                        };

                        let session_infos: Vec<SessionInfo> = page
                            .iter()
                            .map(|s| {
                                let updated = unix_secs_to_iso8601(s.updated_at);
                                SessionInfo::new(s.id.clone(), PathBuf::from(&s.cwd))
                                    .title(s.title.clone())
                                    .updated_at(updated)
                            })
                            .collect();
                        responder.respond(
                            ListSessionsResponse::new(session_infos).next_cursor(next_cursor),
                        )
                    }
                    Err(e) => {
                        error!(error = %e, "ACP: session/list storage error");
                        responder.respond(ListSessionsResponse::new(vec![]))
                    }
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: PromptRequest, responder, cx| {
                let session_id = req.session_id.clone();
                let prompt_text = extract_text(&req.prompt);
                let session_id_str = session_id.0.to_string();

                let (history, cwd, session_mode) = {
                    let sessions = sessions_prompt.lock().unwrap();
                    match sessions.get(&session_id_str) {
                        Some(state) => {
                            (state.history.clone(), state.cwd.clone(), state.mode.clone())
                        }
                        None => {
                            return responder.respond_with_error(
                                agent_client_protocol::util::internal_error("unknown session"),
                            );
                        }
                    }
                };

                let (cancel_trigger, cancel_token) = CancelToken::new();
                if let Some(s) = sessions_prompt.lock().unwrap().get_mut(&session_id_str) {
                    s.cancel = Some(cancel_trigger);
                }

                let (permission_tx, permission_rx) = flume::unbounded::<String>();

                let model = model_prompt.clone();
                let config = config_prompt.clone();
                let perms = perms_prompt.clone();
                let timeouts_t = timeouts_prompt;
                let sessions = Arc::clone(&sessions_prompt);
                let storage = Arc::clone(&storage_prompt);
                let run_mode = session_mode.clone();

                let cx_spawn = cx.clone();
                cx.spawn({
                    let cx = cx_spawn;
                    async move {
                        let (raw_tx, event_rx) = flume::unbounded::<Envelope>();
                        let (hist_tx, hist_rx) = tokio::sync::oneshot::channel::<Vec<Message>>();

                        let model_t = model.clone();
                        let config_t = config.clone();
                        let cwd_t = cwd.clone();

                        let session_id_str2 = session_id_str.clone();
                        tokio::task::spawn_blocking(move || {
                            smol::block_on(async move {
                                let session_id_str = session_id_str2;
                                let event_sender = EventSender::new(raw_tx, 0);

                                let prov =
                                    match provider::from_model_async(&model_t, timeouts_t).await {
                                        Ok(p) => Arc::from(p),
                                        Err(e) => {
                                            error!(error = %e, "ACP: provider error");
                                            return;
                                        }
                                    };

                                let vars = template::env_vars();
                                let instructions =
                                    agent::load_instructions(&cwd_t.to_string_lossy());
                                let filter =
                                    ToolFilter::from_config(&config_t, &[QUESTION_TOOL_NAME]);
                                let ctx = DescriptionContext { filter: &filter };
                                let tools = ToolRegistry::native().definitions(
                                    &vars,
                                    &ctx,
                                    model_t.supports_tool_examples(),
                                );
                                let system = agent::build_system_prompt(
                                    &vars,
                                    &run_mode,
                                    &instructions.text,
                                );

                                let maki_agent = MakiAgent::new(
                                    AgentParams {
                                        provider: prov,
                                        model: model_t,
                                        config: config_t,
                                        tool_output_lines: ToolOutputLines::default(),
                                        permissions: Arc::new(
                                            maki_agent::permissions::PermissionManager::new(
                                                perms,
                                                cwd_t.clone(),
                                            ),
                                        ),
                                        session_id: Some(session_id_str.clone()),
                                        timeouts: timeouts_t,
                                        file_tracker: FileReadTracker::fresh(),
                                    },
                                    AgentRunParams {
                                        history: History::new(history),
                                        system,
                                        event_tx: event_sender,
                                        tools,
                                    },
                                )
                                .with_loaded_instructions(instructions.loaded)
                                .with_cancel(cancel_token)
                                .with_user_response_rx(Arc::new(async_lock::Mutex::new(
                                    permission_rx,
                                )));

                                let outcome = maki_agent
                                    .run(AgentInput {
                                        message: prompt_text,
                                        mode: run_mode,
                                        ..Default::default()
                                    })
                                    .await;

                                let _ = hist_tx.send(outcome.history.into_vec());
                            });
                        });

                        // Default to Cancelled; only overridden when Done event fires.
                        let mut acp_stop_reason = StopReason::Cancelled;

                        while let Ok(envelope) = event_rx.recv_async().await {
                            match &envelope.event {
                                AgentEvent::TextDelta { text } => {
                                    cx.send_notification(session_update(
                                        &session_id,
                                        SessionUpdate::AgentMessageChunk(ContentChunk::new(
                                            ContentBlock::Text(TextContent::new(text.clone())),
                                        )),
                                    ))?;
                                }
                                AgentEvent::ThinkingDelta { text } => {
                                    cx.send_notification(session_update(
                                        &session_id,
                                        SessionUpdate::AgentThoughtChunk(ContentChunk::new(
                                            ContentBlock::Text(TextContent::new(text.clone())),
                                        )),
                                    ))?;
                                }
                                AgentEvent::ToolStart(e) => {
                                    let title = if e.summary.is_empty() {
                                        e.tool.to_string()
                                    } else {
                                        e.summary.clone()
                                    };
                                    cx.send_notification(session_update(
                                        &session_id,
                                        SessionUpdate::ToolCall(
                                            ToolCall::new(e.id.clone(), title)
                                                .kind(tool_kind(&e.tool))
                                                .status(ToolCallStatus::InProgress),
                                        ),
                                    ))?;
                                }
                                AgentEvent::ToolDone(e) => {
                                    let status = if e.is_error {
                                        ToolCallStatus::Failed
                                    } else {
                                        ToolCallStatus::Completed
                                    };
                                    cx.send_notification(session_update(
                                        &session_id,
                                        SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
                                            ToolCallId::new(e.id.as_str()),
                                            ToolCallUpdateFields::new().status(status),
                                        )),
                                    ))?;
                                }
                                AgentEvent::PermissionRequest { id, tool, scopes } => {
                                    let scope_label = scopes.join(", ");
                                    let tool_call_update = ToolCallUpdate::new(
                                        ToolCallId::new(id.as_str()),
                                        ToolCallUpdateFields::new()
                                            .status(ToolCallStatus::Pending)
                                            .kind(tool_kind(tool)),
                                    );
                                    let options = vec![
                                        PermissionOption::new(
                                            "allow",
                                            format!("Allow {tool} ({scope_label}) once"),
                                            PermissionOptionKind::AllowOnce,
                                        ),
                                        PermissionOption::new(
                                            "allow_session",
                                            format!("Allow {tool} for session"),
                                            PermissionOptionKind::AllowAlways,
                                        ),
                                        PermissionOption::new(
                                            "deny",
                                            "Deny",
                                            PermissionOptionKind::RejectOnce,
                                        ),
                                        PermissionOption::new(
                                            "deny_always_local",
                                            "Deny always",
                                            PermissionOptionKind::RejectAlways,
                                        ),
                                    ];
                                    let answer = match cx
                                        .send_request(RequestPermissionRequest::new(
                                            session_id.clone(),
                                            tool_call_update,
                                            options,
                                        ))
                                        .block_task()
                                        .await
                                    {
                                        Ok(resp) => match resp.outcome {
                                            RequestPermissionOutcome::Selected(sel) => {
                                                let option_id = sel.option_id.0.to_string();
                                                if is_permission_grant(&option_id) {
                                                    let _ = cx.send_notification(session_update(
                                                        &session_id,
                                                        SessionUpdate::ToolCallUpdate(
                                                            ToolCallUpdate::new(
                                                                ToolCallId::new(id.as_str()),
                                                                ToolCallUpdateFields::new().status(
                                                                    ToolCallStatus::InProgress,
                                                                ),
                                                            ),
                                                        ),
                                                    ));
                                                }
                                                option_id
                                            }
                                            _ => "deny".to_string(),
                                        },
                                        Err(e) => {
                                            error!(error = %e, "ACP: permission request failed");
                                            "deny".to_string()
                                        }
                                    };
                                    let _ = permission_tx.send(answer);
                                }
                                AgentEvent::Done {
                                    stop_reason: sr, ..
                                } => {
                                    acp_stop_reason = map_stop_reason(*sr);
                                    break;
                                }
                                AgentEvent::Error { message } => {
                                    error!(error = %message, "ACP: agent error");
                                    acp_stop_reason = StopReason::Cancelled;
                                    break;
                                }
                                _ => {}
                            }
                        }

                        // Persist updated history, save to storage, clear cancel trigger
                        let hist_result = hist_rx.await;
                        // Always clear cancel trigger, even if the agent task failed or panicked.
                        if let Ok(mut sessions) = sessions.lock() {
                            if let Some(state) = sessions.get_mut(&session_id_str) {
                                state.cancel = None;
                                if let Ok(ref new_history) = hist_result {
                                    state.history = new_history.clone();
                                }
                            }
                        }
                        if let Ok(new_history) = hist_result {
                            let cwd_str = cwd.to_string_lossy().into_owned();
                            let mut acp_session = AcpSession::load(&session_id_str, &storage)
                                .unwrap_or_else(|_| {
                                    let mut s = AcpSession::new(&model.id, &cwd_str);
                                    s.id = session_id_str.clone();
                                    s
                                });
                            acp_session.model = model.id.clone();
                            acp_session.messages = new_history;
                            acp_session.meta.mode = Some(agent_mode_to_stored(&session_mode));
                            let old_title = acp_session.title.clone();
                            acp_session.update_title_if_default();
                            let title_changed = old_title != acp_session.title;
                            let _ = acp_session.save(&storage);
                            if title_changed {
                                let updated_at = jiff::Timestamp::now().to_string();
                                let _ = cx.send_notification(session_update(
                                    &session_id,
                                    SessionUpdate::SessionInfoUpdate(
                                        SessionInfoUpdate::new()
                                            .title(acp_session.title.clone())
                                            .updated_at(updated_at),
                                    ),
                                ));
                            }
                        }

                        responder.respond(PromptResponse::new(acp_stop_reason))
                    }
                })?;
                Ok(())
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |req: SetSessionModeRequest, responder, _cx| {
                let session_id_str = req.session_id.0.to_string();
                let mode_id = req.mode_id.0.as_ref();
                let mut sessions = sessions_set_mode.lock().unwrap();
                match sessions.get_mut(&session_id_str) {
                    Some(state) => {
                        let new_mode = match mode_id {
                            "build" => AgentMode::Build,
                            "plan" => AgentMode::Plan(state.cwd.join(PLAN_FILENAME)),
                            _ => {
                                return responder.respond_with_error(
                                    agent_client_protocol::util::internal_error("unknown mode id"),
                                );
                            }
                        };
                        state.mode = new_mode;
                        responder.respond(SetSessionModeResponse::new())
                    }
                    None => responder.respond_with_error(
                        agent_client_protocol::util::internal_error("unknown session"),
                    ),
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |_req: AuthenticateRequest, responder, _cx| {
                responder.respond(AuthenticateResponse::new())
            },
            on_receive_request!(),
        )
        .on_receive_request(
            async move |_req: SetSessionConfigOptionRequest, responder, _cx| {
                responder.respond(SetSessionConfigOptionResponse::new(vec![]))
            },
            on_receive_request!(),
        )
        .on_receive_notification(
            async move |notif: CancelNotification, _cx| {
                if let Ok(mut sessions) = sessions_cancel.lock() {
                    if let Some(state) = sessions.get_mut(&*notif.session_id.0) {
                        state.cancel = None; // dropping CancelTrigger fires cancellation
                    }
                }
                Ok(())
            },
            on_receive_notification!(),
        )
        .on_receive_dispatch(
            async move |msg: Dispatch, cx: ConnectionTo<Client>| {
                msg.respond_with_error(
                    agent_client_protocol::util::internal_error("method not supported"),
                    cx,
                )
            },
            on_receive_dispatch!(),
        )
        .connect_to(ByteStreams::new(
            tokio::io::stdout().compat_write(),
            tokio::io::stdin().compat(),
        ))
        .await
        .map_err(|e| color_eyre::eyre::eyre!("{e}"))
}

fn tool_kind(name: &str) -> ToolKind {
    match name {
        READ_TOOL_NAME | GLOB_TOOL_NAME | GREP_TOOL_NAME => ToolKind::Read,
        EDIT_TOOL_NAME | WRITE_TOOL_NAME | MULTIEDIT_TOOL_NAME => ToolKind::Edit,
        BASH_TOOL_NAME | BATCH_TOOL_NAME | CODE_EXECUTION_TOOL_NAME => ToolKind::Execute,
        _ => ToolKind::Other,
    }
}

fn history_to_updates(messages: &[Message]) -> Vec<SessionUpdate> {
    let mut tool_results: HashMap<&str, (&str, bool)> = HashMap::new();
    for msg in messages {
        if matches!(msg.role, ProviderRole::User) {
            for block in &msg.content {
                if let ProviderBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } = block
                {
                    tool_results.insert(tool_use_id.as_str(), (content.as_str(), *is_error));
                }
            }
        }
    }

    let mut updates = Vec::new();
    for msg in messages {
        match msg.role {
            ProviderRole::User => {
                if let Some(text) = msg.user_text() {
                    if !text.is_empty() {
                        updates.push(SessionUpdate::UserMessageChunk(ContentChunk::new(
                            ContentBlock::Text(TextContent::new(text.to_string())),
                        )));
                    }
                }
            }
            ProviderRole::Assistant => {
                for block in &msg.content {
                    match block {
                        ProviderBlock::Text { text } if !text.is_empty() => {
                            updates.push(SessionUpdate::AgentMessageChunk(ContentChunk::new(
                                ContentBlock::Text(TextContent::new(text.clone())),
                            )));
                        }
                        ProviderBlock::Thinking { thinking, .. } => {
                            updates.push(SessionUpdate::AgentThoughtChunk(ContentChunk::new(
                                ContentBlock::Text(TextContent::new(thinking.clone())),
                            )));
                        }
                        ProviderBlock::ToolUse { id, name, .. } => {
                            let (status, raw_output) =
                                if let Some((result, is_error)) = tool_results.get(id.as_str()) {
                                    let s = if *is_error {
                                        ToolCallStatus::Failed
                                    } else {
                                        ToolCallStatus::Completed
                                    };
                                    (s, Some(serde_json::Value::String((*result).to_string())))
                                } else {
                                    (ToolCallStatus::Completed, None)
                                };
                            let mut tc =
                                ToolCall::new(ToolCallId::new(id.as_str()), name.to_string())
                                    .kind(tool_kind(name.as_ref()))
                                    .status(status);
                            if let Some(output) = raw_output {
                                tc = tc.raw_output(output);
                            }
                            updates.push(SessionUpdate::ToolCall(tc));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    updates
}

fn replay_history(
    messages: &[Message],
    session_id: &SessionId,
    cx: &ConnectionTo<Client>,
) -> color_eyre::Result<()> {
    for update in history_to_updates(messages) {
        cx.send_notification(session_update(session_id, update))?;
    }
    Ok(())
}

fn extract_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| {
            if let ContentBlock::Text(t) = b {
                Some(t.text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn session_update(session_id: &SessionId, update: SessionUpdate) -> SessionNotification {
    SessionNotification::new(session_id.clone(), update)
}

fn map_stop_reason(reason: Option<MakiStopReason>) -> StopReason {
    match reason {
        Some(MakiStopReason::MaxTokens) => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

fn is_permission_grant(option_id: &str) -> bool {
    option_id.starts_with("allow")
}

fn encode_cursor(updated_at: u64, id: &str) -> String {
    format!("{updated_at}:{id}")
}

fn decode_cursor(cursor: &str) -> Option<(u64, &str)> {
    let (ts, id) = cursor.split_once(':')?;
    Some((ts.parse().ok()?, id))
}

fn unix_secs_to_iso8601(secs: u64) -> String {
    jiff::Timestamp::from_second(secs as i64)
        .map(|ts| ts.to_string())
        .unwrap_or_default()
}

fn make_mode_state(mode: &AgentMode) -> SessionModeState {
    let current = match mode {
        AgentMode::Build => "build",
        AgentMode::Plan(_) => "plan",
    };
    SessionModeState::new(
        current,
        vec![
            SessionMode::new("build", "Build"),
            SessionMode::new("plan", "Plan"),
        ],
    )
}

fn agent_mode_to_stored(mode: &AgentMode) -> maki_storage::sessions::StoredMode {
    match mode {
        AgentMode::Build => maki_storage::sessions::StoredMode::Build,
        AgentMode::Plan(_) => maki_storage::sessions::StoredMode::Plan,
    }
}

fn stored_mode_to_agent(
    mode: &Option<maki_storage::sessions::StoredMode>,
    cwd: &std::path::Path,
) -> AgentMode {
    match mode {
        Some(maki_storage::sessions::StoredMode::Plan) => AgentMode::Plan(cwd.join(PLAN_FILENAME)),
        _ => AgentMode::Build,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::StopReason;
    use maki_providers::{
        ContentBlock as ProviderBlock, Message, Role as ProviderRole, StopReason as MakiStopReason,
    };
    use maki_storage::sessions::SessionSummary;
    use test_case::test_case;

    // --- map_stop_reason ---

    #[test_case(Some(MakiStopReason::MaxTokens) ; "max_tokens")]
    #[test_case(Some(MakiStopReason::EndTurn)   ; "end_turn")]
    #[test_case(Some(MakiStopReason::ToolUse)   ; "tool_use")]
    #[test_case(None                            ; "none")]
    fn map_stop_reason_never_produces_cancelled(reason: Option<MakiStopReason>) {
        assert!(!matches!(map_stop_reason(reason), StopReason::Cancelled));
    }

    #[test]
    fn map_stop_reason_max_tokens_maps_to_max_tokens() {
        assert!(matches!(
            map_stop_reason(Some(MakiStopReason::MaxTokens)),
            StopReason::MaxTokens
        ));
    }

    #[test]
    fn map_stop_reason_other_maps_to_end_turn() {
        assert!(matches!(map_stop_reason(None), StopReason::EndTurn));
    }

    // --- is_permission_grant ---

    #[test_case("allow",            true  ; "allow_once")]
    #[test_case("allow_session",    true  ; "allow_session")]
    #[test_case("deny",             false ; "deny")]
    #[test_case("deny_always_local",false ; "deny_always")]
    #[test_case("",                 false ; "empty")]
    fn permission_grant_detection(option_id: &str, expected: bool) {
        assert_eq!(is_permission_grant(option_id), expected);
    }

    // --- history_to_updates ---

    fn assistant(content: Vec<ProviderBlock>) -> Message {
        Message {
            role: ProviderRole::Assistant,
            content,
            ..Default::default()
        }
    }

    fn tool_result(tool_use_id: &str, content: &str, is_error: bool) -> Message {
        Message {
            role: ProviderRole::User,
            content: vec![ProviderBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn replay_user_text_produces_user_chunk() {
        let updates = history_to_updates(&[Message::user("hello".into())]);
        assert_eq!(updates.len(), 1);
        assert!(matches!(updates[0], SessionUpdate::UserMessageChunk(_)));
    }

    #[test]
    fn replay_synthetic_message_skipped() {
        let updates = history_to_updates(&[Message::synthetic("internal".into())]);
        assert!(updates.is_empty());
    }

    #[test]
    fn replay_assistant_text_produces_agent_chunk() {
        let updates = history_to_updates(&[assistant(vec![ProviderBlock::Text {
            text: "response".into(),
        }])]);
        assert_eq!(updates.len(), 1);
        assert!(matches!(updates[0], SessionUpdate::AgentMessageChunk(_)));
    }

    #[test]
    fn replay_thinking_produces_thought_chunk() {
        let updates = history_to_updates(&[assistant(vec![ProviderBlock::Thinking {
            thinking: "thoughts".into(),
            signature: None,
        }])]);
        assert_eq!(updates.len(), 1);
        assert!(matches!(updates[0], SessionUpdate::AgentThoughtChunk(_)));
    }

    #[test]
    fn replay_tool_use_with_completed_result() {
        let msgs = vec![
            assistant(vec![ProviderBlock::ToolUse {
                id: "t1".into(),
                name: maki_agent::tools::READ_TOOL_NAME.into(),
                input: serde_json::Value::Null,
            }]),
            tool_result("t1", "file content", false),
        ];
        let updates = history_to_updates(&msgs);
        assert_eq!(updates.len(), 1);
        let SessionUpdate::ToolCall(tc) = &updates[0] else {
            panic!("expected ToolCall");
        };
        assert_eq!(tc.tool_call_id.0.as_ref(), "t1");
        assert_eq!(tc.status, ToolCallStatus::Completed);
        assert_eq!(tc.kind, ToolKind::Read);
        assert!(tc.raw_output.is_some());
    }

    #[test]
    fn replay_tool_use_with_error_result() {
        let msgs = vec![
            assistant(vec![ProviderBlock::ToolUse {
                id: "t2".into(),
                name: maki_agent::tools::BASH_TOOL_NAME.into(),
                input: serde_json::Value::Null,
            }]),
            tool_result("t2", "oops", true),
        ];
        let updates = history_to_updates(&msgs);
        assert_eq!(updates.len(), 1);
        let SessionUpdate::ToolCall(tc) = &updates[0] else {
            panic!("expected ToolCall");
        };
        assert_eq!(tc.status, ToolCallStatus::Failed);
        assert_eq!(tc.kind, ToolKind::Execute);
    }

    #[test]
    fn replay_tool_use_without_result_defaults_completed() {
        let msgs = vec![assistant(vec![ProviderBlock::ToolUse {
            id: "t3".into(),
            name: maki_agent::tools::READ_TOOL_NAME.into(),
            input: serde_json::Value::Null,
        }])];
        let updates = history_to_updates(&msgs);
        assert_eq!(updates.len(), 1);
        let SessionUpdate::ToolCall(tc) = &updates[0] else {
            panic!("expected ToolCall");
        };
        assert_eq!(tc.status, ToolCallStatus::Completed);
        assert!(tc.raw_output.is_none());
    }

    #[test]
    fn replay_conversation_ordering() {
        let msgs = vec![
            Message::user("question".into()),
            assistant(vec![
                ProviderBlock::Text {
                    text: "thinking out loud".into(),
                },
                ProviderBlock::ToolUse {
                    id: "t4".into(),
                    name: maki_agent::tools::READ_TOOL_NAME.into(),
                    input: serde_json::Value::Null,
                },
            ]),
            tool_result("t4", "result", false),
        ];
        let updates = history_to_updates(&msgs);
        assert_eq!(updates.len(), 3);
        assert!(matches!(updates[0], SessionUpdate::UserMessageChunk(_)));
        assert!(matches!(updates[1], SessionUpdate::AgentMessageChunk(_)));
        assert!(matches!(updates[2], SessionUpdate::ToolCall(_)));
    }

    fn summary(id: &str, updated_at: u64) -> SessionSummary {
        SessionSummary {
            id: id.to_string(),
            title: id.to_string(),
            updated_at,
            cwd: "/tmp".to_string(),
        }
    }

    fn paginate(
        mut list: Vec<SessionSummary>,
        cursor: Option<&str>,
    ) -> (Vec<String>, Option<String>) {
        list.sort_unstable_by(|a, b| b.updated_at.cmp(&a.updated_at).then(a.id.cmp(&b.id)));
        let start = cursor
            .and_then(decode_cursor)
            .and_then(|(cursor_ts, cursor_id)| {
                list.iter().position(|s| {
                    s.updated_at < cursor_ts
                        || (s.updated_at == cursor_ts && s.id.as_str() > cursor_id)
                })
            })
            .unwrap_or(0);
        let page = &list[start..];
        let page = &page[..page.len().min(SESSION_PAGE_SIZE)];
        let next_cursor = if start + SESSION_PAGE_SIZE < list.len() {
            page.last().map(|s| encode_cursor(s.updated_at, &s.id))
        } else {
            None
        };
        (page.iter().map(|s| s.id.clone()).collect(), next_cursor)
    }

    #[test_case("100:abc", Some((100, "abc")) ; "valid_cursor")]
    #[test_case("0:z", Some((0, "z"))         ; "zero_ts")]
    #[test_case("abc",   None                 ; "no_colon")]
    #[test_case("x:abc", None                 ; "non_numeric_ts")]
    fn decode_cursor_cases(input: &str, expected: Option<(u64, &str)>) {
        assert_eq!(decode_cursor(input), expected);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let encoded = encode_cursor(42, "some-uuid");
        assert_eq!(decode_cursor(&encoded), Some((42, "some-uuid")));
    }

    #[test]
    fn first_page_no_cursor() {
        let list: Vec<_> = (0..60)
            .map(|i| summary(&format!("id-{i:02}"), 100 - i))
            .collect();
        let (ids, next) = paginate(list, None);
        assert_eq!(ids.len(), SESSION_PAGE_SIZE);
        assert_eq!(ids[0], "id-00");
        assert!(next.is_some());
    }

    #[test]
    fn second_page_exhausts_results() {
        let list: Vec<_> = (0..60)
            .map(|i| summary(&format!("id-{i:02}"), 100 - i))
            .collect();
        let (_, next1) = paginate(list.clone(), None);
        let (ids2, next2) = paginate(list, next1.as_deref());
        assert_eq!(ids2.len(), 10);
        assert!(next2.is_none());
    }

    #[test]
    fn no_next_cursor_when_exactly_one_page() {
        let list: Vec<_> = (0..SESSION_PAGE_SIZE)
            .map(|i| summary(&format!("id-{i:02}"), 100 - i as u64))
            .collect();
        let (ids, next) = paginate(list, None);
        assert_eq!(ids.len(), SESSION_PAGE_SIZE);
        assert!(next.is_none());
    }

    #[test]
    fn tie_breaking_by_id_asc() {
        let list = vec![
            summary("bbb", 100),
            summary("aaa", 100),
            summary("ccc", 100),
        ];
        let (ids, _) = paginate(list, None);
        assert_eq!(ids, vec!["aaa", "bbb", "ccc"]);
    }

    #[test]
    fn cursor_resumes_after_tie_broken_item() {
        let list = vec![
            summary("aaa", 100),
            summary("bbb", 100),
            summary("ccc", 100),
            summary("ddd", 99),
        ];
        // cursor points at "bbb" (second item)
        let cursor = encode_cursor(100, "bbb");
        let (ids, next) = paginate(list, Some(&cursor));
        assert_eq!(ids, vec!["ccc", "ddd"]);
        assert!(next.is_none());
    }

    #[test]
    fn invalid_cursor_falls_back_to_first_page() {
        let list: Vec<_> = (0..5)
            .map(|i| summary(&format!("id-{i}"), i as u64))
            .collect();
        let (ids_no_cursor, _) = paginate(list.clone(), None);
        let (ids_bad_cursor, _) = paginate(list, Some("not-a-valid-cursor"));
        assert_eq!(ids_no_cursor, ids_bad_cursor);
    }
}
