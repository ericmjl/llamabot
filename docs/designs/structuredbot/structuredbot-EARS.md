# StructuredBot and AsyncStructuredBot — EARS

**Parent LLD**: [./LLD.md](./LLD.md)

## Model and initialization

- [x] **CORE-STRUCT-001**: Where the model name contains `ollama_chat`, `StructuredBot.__init__` shall not require `response_format` and `response_schema` in supported OpenAI params.

- [x] **CORE-STRUCT-002**: For other models, where `response_format` with `response_schema` is not supported, `StructuredBot.__init__` shall raise `ValueError` with guidance to use a compatible model.

- [x] **CORE-STRUCT-003**: `StructuredBot` shall call `SimpleBot.__init__` with `json_mode=True` and the supplied `model_name` and `stream_target`.

## Structured message composition

- [x] **CORE-STRUCT-010**: `compose_structured_first_attempt_messages` shall return `full_messages` equal to `[system_prompt]` plus `to_basemessage(user_messages)` with no memory retrieval.

## Validation loop

- [x] **CORE-STRUCT-020**: `StructuredBot.__call__` shall attempt up to `num_attempts` completions; on each attempt it shall parse assistant content as JSON and validate with `pydantic_model.model_validate`.

- [x] **CORE-STRUCT-021**: On `ValidationError` before the last attempt, the system shall extend the conversation with the prior assistant message and the `HumanMessage` from `get_validation_error_message`.

- [x] **CORE-STRUCT-022**: On the last attempt, if validation fails and `allow_failed_validation` is true and a parsed dict exists, the system shall return `pydantic_model.model_construct(**last_codeblock)`; otherwise it shall re-raise the validation error.

## AsyncStructuredBot

- [x] **CORE-STRUCT-030**: `AsyncStructuredBot.__call__` shall execute the synchronous `StructuredBot.__call__` via `asyncio.to_thread` with the same `num_attempts` and `verbose` arguments.

## Verification

| ID | Tests / code |
| --- | --- |
| CORE-STRUCT-001–003 | `tests/bot/test_structuredbot.py` (model support / init) |
| CORE-STRUCT-010 | `tests/bot/test_structuredbot.py` (`compose_structured_first_attempt_messages`) |
| CORE-STRUCT-020–022 | `tests/bot/test_structuredbot.py` (validation loop, `allow_failed_validation`) |
| CORE-STRUCT-030 | `llamabot/bot/structuredbot.py` (`AsyncStructuredBot.__call__` → `asyncio.to_thread`); no dedicated pytest for `__call__` yet |

## Related Documents

- [StructuredBot LLD](./LLD.md)
- [SimpleBot LLD](../simplebot/LLD.md)
- [High-Level Design](../../high-level-design.md)
