import os

import streamlit as st

from answer_memory import load_answers
from monitor_worker import run_monitoring_cycle
from dynamic_rag import DynamicRetriever, query_dynamic_rag
from monitor_pipeline import (
    run_controlled_scenario_a,
    run_controlled_scenario_b,
)
from notification import load_notifications
from v2_monitor_pipeline import run_v2_monitor
from source_registry import load_sources


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Knowledge Reliability AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 17px;
        color: #8b949e;
        margin-bottom: 28px;
    }

    .answer-card {
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #30363d;
        background-color: #161b22;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .current-card {
        padding: 10px 16px;
        border-radius: 10px;
        background-color: #17351f;
        border: 1px solid #2ea043;
        color: #56d364;
        font-weight: 600;
        display: inline-block;
    }

    .stale-card {
        padding: 10px 16px;
        border-radius: 10px;
        background-color: #3a2020;
        border: 1px solid #f85149;
        color: #ff7b72;
        font-weight: 600;
        display: inline-block;
    }

    .source-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #30363d;
        background-color: #0d1117;
        margin-bottom: 12px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #30363d;
        background-color: #161b22;
        text-align: center;
    }

    .pipeline-step {
        padding: 15px 8px;
        border-radius: 12px;
        border: 1px solid #30363d;
        background-color: #161b22;
        text-align: center;
        min-height: 75px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## 🛡️ Knowledge Reliability AI"
)

st.sidebar.caption(
    "Evidence-backed answers with knowledge-change awareness."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "💬 Ask AI",
        "📚 Answer History",
        "🔔 Reliability Monitor",
        "🌐 Sources",
        "🧪 Technical Validation",
        "🏗️ System Architecture",
    ],
)


# =========================================================
# SESSION STATE
# =========================================================

if "rag_result" not in st.session_state:
    st.session_state.rag_result = None

if "selected_source_id" not in st.session_state:
    st.session_state.selected_source_id = None

if "scenario_a_result" not in st.session_state:
    st.session_state.scenario_a_result = None

if "scenario_b_result" not in st.session_state:
    st.session_state.scenario_b_result = None

if "live_monitor_results" not in st.session_state:
    st.session_state.live_monitor_results = []


# =========================================================
# ASK AI
# =========================================================

if page == "💬 Ask AI":

    st.markdown(
        '<div class="main-title">Ask AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        Ask a question and receive an evidence-backed answer from
        a selected knowledge source.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # LOAD ENABLED SOURCES
    # -----------------------------------------------------

    sources = load_sources(enabled_only=True)

    if not sources:

        st.warning(
            "No enabled knowledge sources are configured."
        )

    else:

        source_options = {
            source.get(
                "source_id",
                "unknown_source",
            ): source
            for source in sources
        }

        source_ids = list(source_options.keys())

        selected_source_id = st.selectbox(
            "Knowledge Source",
            source_ids,
            format_func=lambda source_id: (
                f"{source_options[source_id].get('source_identifier', 'Unnamed Source')} "
                f"— {source_options[source_id].get('source_url', '')}"
            ),
            help="Choose the source that should be used to answer your question.",
        )

        selected_source = source_options[selected_source_id]

        st.caption(
            f"Source URL: {selected_source.get('source_url', 'N/A')}"
        )

        question = st.text_input(
            "Your question",
            placeholder="Ask something about the selected source...",
        )

        ask = st.button(
            "🔎 Ask AI",
            type="primary",
            use_container_width=False,
        )

        if ask:

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                api_key = os.environ.get(
                    "GEMINI_API_KEY",
                    "",
                ).strip()

                if not api_key:

                    st.error(
                        "GEMINI_API_KEY is not configured."
                    )

                else:

                    with st.spinner(
                        "Retrieving evidence and generating answer..."
                    ):

                        try:

                            # IMPORTANT:
                            # Only the selected source is passed
                            # to the DynamicRetriever.
                            retriever = DynamicRetriever(
                                sources=[selected_source],
                                api_key=api_key,
                            )

                            result = query_dynamic_rag(
                                question.strip(),
                                retriever,
                                api_key=api_key,
                            )

                            st.session_state.rag_result = result
                            st.session_state.selected_source_id = (
                                selected_source.get("source_id")
                            )

                        except Exception as error:

                            st.error(
                                f"Unable to process the question: {error}"
                            )

    # -----------------------------------------------------
    # DISPLAY ANSWER
    # -----------------------------------------------------

    result = st.session_state.rag_result

    if result:

        st.divider()

        st.subheader("Answer")

        answer = result.get(
            "answer",
            "No answer generated.",
        )

        st.markdown(
            f"""
            <div class="answer-card">
                <div style="font-size:18px; line-height:1.6;">
                    {answer}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        if not result.get("gate_passed"):

            st.markdown(
                '<div class="stale-card">⚠ INSUFFICIENT RETRIEVAL EVIDENCE</div>',
                unsafe_allow_html=True,
            )

        elif not result.get("answerable", True) or result.get("reason") == "llm_found_insufficient_evidence":

            st.markdown(
                '<div class="stale-card">⚠ EVIDENCE RELEVANT, BUT ANSWER NOT SUPPORTED</div>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="current-card">✓ GROUNDED ANSWER GENERATED</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # -------------------------------------------------
        # SOURCE + EVIDENCE
        # -------------------------------------------------

        chunk = result.get(
            "retrieved_chunk"
        )

        if chunk:

            st.subheader("Evidence")

            evidence_chunks = result.get(
                "retrieved_chunks",
                [],
            )

            if not evidence_chunks:
                evidence_chunks = [chunk]

            for evidence in evidence_chunks:

                st.info(
                    evidence.text
                )

            st.subheader("Source")

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Source:** {chunk.source_identifier}"
                )

                st.write(
                    f"**URL:** {chunk.source_url}"
                )

            with col2:

                st.write(
                    f"**Source Version:** `{chunk.source_version_id}`"
                )

                st.write(
                    f"**Retrieval Score:** "
                    f"`{result.get('retrieval_score', result.get('similarity_score', 0)):.4f}`"
                )

                if "semantic_similarity" in result:
                    st.write(
                        f"**Semantic Similarity:** "
                        f"`{result.get('semantic_similarity', 0):.4f}`"
                    )

        # -------------------------------------------------
        # PROVENANCE
        # -------------------------------------------------

        record = result.get(
            "answer_record"
        )

        if record:

            with st.expander(
                "🔗 View answer provenance"
            ):

                st.write(
                    f"**Answer ID:** `{record.get('answer_id')}`"
                )

                st.write(
                    f"**Source ID:** `{record.get('source_id')}`"
                )

                st.write(
                    f"**Source Version ID:** "
                    f"`{record.get('source_version_id')}`"
                )

                st.write(
                    "The answer is linked to the source evidence "
                    "that supported it."
                )


# =========================================================
# ANSWER HISTORY
# =========================================================

elif page == "📚 Answer History":

    st.markdown(
        '<div class="main-title">Answer History</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        Previously generated answers and their stored provenance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    answers = load_answers()

    if not answers:

        st.info(
            "No saved answers found yet."
        )

    else:

        st.write(
            f"**{len(answers)} stored answer record(s)**"
        )

        for answer in reversed(answers):

            history = (
                answer
                .get("provenance", {})
                .get("validity_history", [])
            )

            status = "CURRENT"

            if history:

                latest = history[-1].get(
                    "status",
                    "current",
                )

                if latest == "stale":
                    status = "STALE"

            with st.container():

                st.markdown(
                    '<div class="source-card">',
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(
                    [4, 1]
                )

                with col1:

                    st.markdown(
                        f"### {answer.get('question', 'Unknown question')}"
                    )

                    st.write(
                        answer.get(
                            "generated_answer",
                            "",
                        )
                    )

                    st.caption(
                        f"Source: "
                        f"{answer.get('source_identifier', 'Unknown')}"
                    )

                with col2:

                    if status == "STALE":

                        st.markdown(
                            '<div class="stale-card">⚠ STALE</div>',
                            unsafe_allow_html=True,
                        )

                    else:

                        st.markdown(
                            '<div class="current-card">✓ CURRENT</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )


# =========================================================
# RELIABILITY MONITOR
# =========================================================

elif page == "🔔 Reliability Monitor":

    st.markdown(
        '<div class="main-title">Reliability Monitor</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        Monitor registered knowledge sources and detect changes that may affect
        previously generated answers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # LIVE REGISTERED-SOURCE MONITOR
    # -----------------------------------------------------

    st.subheader("🌐 Registered Source Monitoring")

    registered_sources = load_sources(enabled_only=True)

    if registered_sources:
        st.write(
            f"Monitoring **{len(registered_sources)} enabled source(s)** "
            "from the source registry."
        )

        if st.button(
            "🔄 Run Source Monitoring Now",
            type="primary",
            key="run_source_monitoring",
        ):

            api_key = os.environ.get(
                "GEMINI_API_KEY",
                "",
            ).strip()

            if not api_key:
                st.error(
                    "GEMINI_API_KEY is not configured."
                )

            else:

                with st.spinner(
                    "Checking registered sources for knowledge changes..."
                ):

                    cycle_results = run_monitoring_cycle()

                    monitor_results = []

                    for item in cycle_results:

                        monitor_results.append(
                            {
                                "source": item,
                                "result": item.get(
                                    "result",
                                    {},
                                ),
                            }
                        )

                    st.session_state.live_monitor_results = (
                        monitor_results
                    )

        st.divider()
        st.subheader("Monitoring Results")

        for item in st.session_state.live_monitor_results:

            source = item["source"]
            result = item["result"]

            status = result.get(
                "status",
                "unknown",
            )

            if status == "changed":

                st.warning(
                    f"⚠️ **{source.get('source_identifier', 'Unknown Source')}** "
                    f"— knowledge change detected."
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Changed Chunks",
                        len(result.get("changes", [])),
                    )

                with col2:
                    st.metric(
                        "Candidate Claims",
                        result.get("candidate_claims", 0),
                    )

                with col3:
                    st.metric(
                        "Notifications",
                        len(result.get("notifications", [])),
                    )

                if result.get("notifications"):

                    st.subheader("🔔 Generated Notifications")

                    for notification in result["notifications"]:

                        st.error(
                            f"""
                            **{notification.get('title', 'Knowledge Reliability Alert')}**

                            **Question:** {notification.get('question', '')}

                            **Previous answer:** {notification.get('previous_answer', '')}

                            **Changed information:**
                            {notification.get('old_information', '')}
                            →
                            {notification.get('new_information', '')}

                            **Corrected answer:**
                            {notification.get('corrected_answer', '')}
                            """
                        )

                elif result.get("reevaluations"):

                    st.info(
                        "The changed source was checked against affected "
                        "historical answers, but no user notification was generated."
                    )

            elif status == "initial_snapshot":

                st.info(
                    f"🟢 **{source.get('source_identifier', 'Unknown Source')}** "
                    "— initial monitoring snapshot created."
                )

            elif status == "unchanged":

                st.success(
                    f"✓ **{source.get('source_identifier', 'Unknown Source')}** "
                    "— no source change detected."
                )

            elif status == "unavailable":

                st.error(
                    f"❌ **{source.get('source_identifier', 'Unknown Source')}** "
                    "— source unavailable."
                )

            elif status == "normalization_failed":

                st.error(
                    f"❌ **{source.get('source_identifier', 'Unknown Source')}** "
                    "— source content could not be normalized."
                )

            else:

                st.error(
                    f"❌ **{source.get('source_identifier', 'Unknown Source')}** "
                    f"— monitoring error: {result.get('error', 'Unknown error')}"
                )

    persisted_notifications = load_notifications()
    if persisted_notifications:
        st.divider()
        st.subheader(f"🔔 Stored Reliability Notifications ({len(persisted_notifications)})")
        st.caption("Active alerts persisted from source monitoring runs.")
        for notif in persisted_notifications:
            st.error(
                f"""
                **{notif.get('title', 'Knowledge Reliability Alert')}**  
                *Recorded at: {notif.get('timestamp', 'N/A')}*

                **Source:** {notif.get('source_identifier', 'Referenced Source')}  
                **Question:** {notif.get('question', '')}  
                **Previous Answer:** {notif.get('previous_answer', '')}  

                **Changed Information:** {notif.get('old_information', 'N/A')} → {notif.get('new_information', 'N/A')}  
                **Updated Answer:** {notif.get('corrected_answer', 'N/A')}  

                **Explanation:** {notif.get('explanation', 'N/A')}
                """
            )

    st.divider()

    st.markdown(
        """
        <div class="subtitle">
        Monitor knowledge changes that may affect previously generated answers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Initial Knowledge",
            "30 days",
        )

    with col2:

        st.metric(
            "Updated Knowledge",
            "15 days",
        )

    with col3:

        st.metric(
            "Change",
            "Material",
        )

    st.divider()

    st.subheader(
        "Latest Demonstrated Change"
    )

    st.error(
        """
        ⚠️ **Knowledge change detected**

        Customer Refund Policy

        **Previous:** Refund period is 30 days.

        **Updated:** Refund period is 15 days.
        """
    )

    st.subheader(
        "Affected Answer"
    )

    st.warning(
        """
        **Question:** What is the refund period for customer purchases?

        **Previous answer:** According to the policy, the refund period is 30 days.

        **Status:** STALE

        **Updated answer:** According to the updated policy, the refund period is 15 days.
        """
    )

    st.success(
        """
        The system identifies the affected historical answer instead of
        re-evaluating unrelated answers.
        """
    )


# =========================================================
# SOURCES
# =========================================================

elif page == "🌐 Sources":

    st.markdown(
        '<div class="main-title">Knowledge Sources</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        Register and manage external knowledge sources.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # ADD NEW SOURCE
    # -----------------------------------------------------

    st.subheader("➕ Add Knowledge Source")

    with st.form("add_source_form"):

        source_name = st.text_input(
            "Source Name",
            placeholder="Example: Python Official",
        )

        source_url = st.text_input(
            "Source URL",
            placeholder="https://example.com/",
        )

        add_source_button = st.form_submit_button(
            "➕ Register Source",
            type="primary",
        )

    if add_source_button:

        source_name = source_name.strip()
        source_url = source_url.strip()

        if not source_name:
            st.error("Please enter a source name.")

        elif not source_url:
            st.error("Please enter a source URL.")

        elif not (
            source_url.startswith("https://")
            or source_url.startswith("http://")
        ):
            st.error("URL must start with http:// or https://")

        else:

            try:
                import subprocess

                result = subprocess.run(
                    [
                        "python3",
                        "source_registry.py",
                        "add",
                        source_name,
                        source_url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:

                    st.success("Source registered successfully.")

                    if result.stdout.strip():
                        st.code(result.stdout.strip())

                    st.rerun()

                else:

                    error_message = (
                        result.stderr.strip()
                        or result.stdout.strip()
                        or "Unable to register source."
                    )

                    st.error(error_message)

            except Exception as error:

                st.error(
                    f"Unable to register source: {error}"
                )

    st.divider()

    # -----------------------------------------------------
    # REGISTERED SOURCES
    # -----------------------------------------------------

    st.subheader("Registered Sources")

    sources = load_sources()

    if not sources:

        st.info("No registered sources found.")

    else:

        for source in sources:

            enabled = source.get(
                "enabled",
                False,
            )

            status = (
                "● ENABLED"
                if enabled
                else "○ DISABLED"
            )

            st.markdown(
                f"""
                <div class="source-card">

                    <h3>
                        {source.get(
                            'source_identifier',
                            'Unnamed Source'
                        )}
                    </h3>

                    <p>
                        <b>Type:</b>
                        {source.get(
                            'source_type',
                            'Unknown'
                        )}
                    </p>

                    <p>
                        <b>URL:</b>
                        {source.get(
                            'source_url',
                            'N/A'
                        )}
                    </p>

                    <p>
                        <b>Status:</b>
                        {status}
                    </p>

                    <p>
                        <b>Source ID:</b>
                        {source.get(
                            'source_id',
                            'N/A'
                        )}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# TECHNICAL VALIDATION
# =========================================================

elif page == "🧪 Technical Validation":

    st.markdown(
        '<div class="main-title">Technical Validation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        Controlled end-to-end tests for validating the reliability pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "Scenario A — Material Knowledge Change"
    )

    st.write(
        """
        Controlled change:

        **Refund period: 30 days → 15 days**

        Expected behavior:

        **identify the affected answer, re-evaluate it, invalidate the old
        answer, and generate a notification.**
        """
    )

    if st.button(
        "▶ Run Material Change Test",
        type="primary",
    ):

        with st.spinner(
            "Running end-to-end reliability pipeline..."
        ):

            try:

                result = run_controlled_scenario_a()

                st.session_state.scenario_a_result = result

            except Exception as error:

                st.error(
                    f"Scenario A failed: {error}"
                )

    if st.session_state.scenario_a_result:

        result = st.session_state.scenario_a_result

        st.success(
            "Material-change scenario completed."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Source Changed",
                "Yes"
                if result.get(
                    "source_changed"
                )
                else "No",
            )

        with col2:

            st.metric(
                "Answers Evaluated",
                result.get(
                    "potentially_affected_count",
                    0,
                ),
            )

        with col3:

            st.metric(
                "Invalidated",
                result.get(
                    "became_invalid_count",
                    0,
                ),
            )

        with col4:

            st.metric(
                "Notifications",
                result.get(
                    "notifications_generated",
                    0,
                ),
            )

        if result.get(
            "notifications"
        ):

            st.subheader(
                "Generated Notification"
            )

            for notification in result[
                "notifications"
            ]:

                st.error(
                    f"""
                    **{notification.get('title')}**

                    **Question:** {notification.get('question')}

                    **Previous answer:** {notification.get('previous_answer')}

                    **Changed information:**
                    {notification.get('old_information')}
                    →
                    {notification.get('new_information')}

                    **Corrected answer:**
                    {notification.get('corrected_answer')}
                    """
                )

    st.divider()

    st.subheader(
        "Scenario B — Unrelated Change"
    )

    st.write(
        """
        Controlled change:

        The support channel changes, but the refund period remains 30 days.

        Expected behavior:

        **the previous answer remains valid and no notification is generated.**
        """
    )

    if st.button(
        "▶ Run Unrelated Change Test"
    ):

        with st.spinner(
            "Running unrelated-change validation..."
        ):

            try:

                result = run_controlled_scenario_b()

                st.session_state.scenario_b_result = result

            except Exception as error:

                st.error(
                    f"Scenario B failed: {error}"
                )

    if st.session_state.scenario_b_result:

        result = st.session_state.scenario_b_result

        st.success(
            "Unrelated-change scenario completed."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Source Changed",
                "Yes",
            )

        with col2:

            st.metric(
                "Answers Remaining Valid",
                result.get(
                    "remained_valid_count",
                    0,
                ),
            )

        with col3:

            st.metric(
                "Notifications",
                result.get(
                    "notifications_generated",
                    0,
                ),
            )


# =========================================================
# SYSTEM ARCHITECTURE
# =========================================================

elif page == "🏗️ System Architecture":

    st.markdown(
        '<div class="main-title">System Architecture</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        Technical view of the reliability-aware RAG architecture.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "Answer Generation"
    )

    pipeline = [
        "👤 User Question",
        "🌐 Source Selection",
        "🔎 Retrieval",
        "🛡️ Evidence Gate",
        "📝 Answer + Provenance",
    ]

    cols = st.columns(
        len(pipeline)
    )

    for col, step in zip(
        cols,
        pipeline,
    ):

        with col:

            st.markdown(
                f'<div class="pipeline-step">{step}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader(
        "Knowledge Reliability"
    )

    pipeline = [
        "🌐 Source",
        "🔄 Snapshot",
        "🔍 Change Detection",
        "🎯 Impact Analysis",
        "🧠 Re-evaluation",
        "🔔 Notification",
    ]

    cols = st.columns(
        len(pipeline)
    )

    for col, step in zip(
        cols,
        pipeline,
    ):

        with col:

            st.markdown(
                f'<div class="pipeline-step">{step}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    st.subheader(
        "Core Technical Idea"
    )

    st.info(
        """
        The system maintains a dependency relationship between generated
        answers and their supporting evidence.

        When source knowledge changes, the system uses that dependency
        information to identify potentially affected historical answers
        and selectively re-evaluate them.
        """
    )