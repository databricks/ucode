"""Real file-reading tasks, including launcher-style options after `--`."""

import json
import uuid

import pytest

pytestmark = pytest.mark.live


@pytest.mark.parametrize(
    "model_form,prompt_form",
    [
        pytest.param("separate", "argument", id="model-value"),
        pytest.param("equals", "argument", id="model-equals"),
        pytest.param("short", "argument", id="model-short"),
        pytest.param("separate", "stdin", id="stdin-prompt"),
        pytest.param("separate", "separator", id="nested-separator"),
    ],
)
def test_agent_reads_file_through_gateway(configured, agent, model_form, prompt_form):
    session = configured
    model = session.model_for_explicit_case(agent)
    nonce = uuid.uuid4().hex
    (session.cwd / "input.txt").write_text(nonce + "\n")
    prompt = "Read input.txt in the current directory using a tool. Reply with only its contents."
    # An explicit model must bypass routing even when globally enabled.
    session.env["ENABLE_SMART_ROUTING_V2"] = "1"
    model_args = {
        "separate": ["--model", model],
        "equals": [f"--model={model}"],
        "short": ["-m", model],
    }[model_form]
    input_text = prompt + "\n" if prompt_form == "stdin" else None
    # Both separators are real: the outer one belongs to ug, the inner one
    # belongs to the selected agent. The prompt remains one argument.
    prompt_args = ["--", prompt] if prompt_form == "separator" else [prompt]
    if agent == "claude":
        # A real caller-supplied settings file exercises the merge needed by
        # launchers such as Isaac. Its hook must execute without losing ug auth.
        caller_settings = session.cwd / "caller settings.json"
        caller_settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionStart": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "echo caller-hook-ran > caller-hook.txt",
                                    }
                                ]
                            }
                        ]
                    },
                }
            )
        )
        args = [
            "-p",
            *model_args,
            "--max-turns",
            "4",
            "--output-format",
            "json",
            "--allowedTools",
            "Read",
            "--settings",
            str(caller_settings),
            *(prompt_args if input_text is None else []),
        ]
    else:
        args = [
            "exec",
            "--skip-git-repo-check",
            "--json",
            *model_args,
            *(prompt_args if input_text is None else ["-"]),
        ]
    result = session.run(agent, "--", *args, timeout=180, input_text=input_text)
    # Look in the agent's structured final output, never in the echoed prompt,
    # a tool request, or a banner. The nonce was not given to the model.
    payloads = []
    for line in result.stdout.splitlines():
        try:
            payloads.append(json.loads(line))
        except ValueError:
            pass
    if agent == "claude":
        final = [p for p in payloads if isinstance(p, dict) and p.get("type") == "result"]
        assert final and not final[-1].get("is_error"), result.stdout
        assert nonce in final[-1].get("result", ""), result.stdout
        assert (session.cwd / "caller-hook.txt").read_text().strip() == "caller-hook-ran"
    else:
        final = [
            p["item"].get("text", "")
            for p in payloads
            if isinstance(p, dict)
            and p.get("type") == "item.completed"
            and p.get("item", {}).get("type") == "agent_message"
        ]
        assert any(nonce in text for text in final), result.stdout
    session.assert_not_routed()
