# app/trace_helpers.py
# Langfuse SDK v4 manual tracing for guardrail and intent classifier.
#
# v4 breaking changes that affected us:
#   - span.update_trace()  → REMOVED. Use propagate_attributes() for session/tags
#   - update_current_trace() → REMOVED. Same replacement.
#   - start_span() / start_generation() → replaced by start_as_current_observation(as_type=...)
#
# v4 correct pattern:
#   with propagate_attributes(session_id=..., tags=[...]):
#       with lf.start_as_current_observation(as_type="span", name=...) as span:
#           with lf.start_as_current_observation(as_type="generation", name=...) as gen:
#               gen.update(output=...)

from langfuse import propagate_attributes


def _trace_guardrail(lf_client, query: str, result: str, session_id: str):
    """
    Traces one guardrail call to Langfuse using SDK v4 API.

    What you'll see in Langfuse dashboard:
      Trace
        └── Span: guardrail
              └── Generation: guardrail_classification
                    input:  <the user query>
                    output: YES or NO
                    tags:   [guardrail, guardrail_blocked] or [guardrail, guardrail_allowed]

    Filter in dashboard:
      - Tag = guardrail_blocked  → all queries that were blocked
      - Tag = guardrail_allowed  → all queries that passed
      - Tag = guardrail          → every single guardrail call
    """
    tags = ["guardrail", "guardrail_blocked" if result == "NO" else "guardrail_allowed"]

    # propagate_attributes() is the v4 way to attach session_id + tags to a trace
    with propagate_attributes(
        trace_name="guardrail",
        session_id=session_id,
        tags=tags,
    ):
        with lf_client.start_as_current_observation(
            as_type="span",
            name="guardrail",
            input={"query": query},
        ):
            with lf_client.start_as_current_observation(
                as_type="generation",
                name="guardrail_classification",
                model="llama-3.3-70b-versatile",
                input=query,
            ) as gen:
                gen.update(
                    output=result,
                    metadata={
                        "blocked": result == "NO",
                        "classifier": "insurance_guardrail"
                    }
                )


def _trace_intent(lf_client, query: str, intent: str, session_id: str):
    """
    Traces one intent classification call to Langfuse using SDK v4 API.

    What you'll see in Langfuse dashboard:
      Trace
        └── Span: intent_classifier
              └── Generation: intent_classification
                    input:  <the user query>
                    output: RECOMMEND / FACTUAL / ANALYTICAL / IRDAI
                    tags:   [intent_classifier, intent_recommend] (etc)

    Filter in dashboard:
      - Tag = intent_recommend   → all recommendation requests
      - Tag = intent_irdai       → all IRDAI data queries
      - Tag = intent_factual     → factual policy questions
      - Tag = intent_classifier  → every single intent call
    """
    tags = ["intent_classifier", f"intent_{intent.lower()}"]

    with propagate_attributes(
        trace_name="intent_classifier",
        session_id=session_id,
        tags=tags,
    ):
        with lf_client.start_as_current_observation(
            as_type="span",
            name="intent_classifier",
            input={"query": query},
        ):
            with lf_client.start_as_current_observation(
                as_type="generation",
                name="intent_classification",
                model="llama-3.3-70b-versatile",
                input=query,
            ) as gen:
                gen.update(
                    output=intent,
                    metadata={
                        "routed_to": intent.lower(),
                        "classifier": "intent_router"
                    }
                )