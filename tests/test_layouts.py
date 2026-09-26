"""Layouts: chat templates in the decorator, the shipped adapters, reasoning,
tools and memory."""

import dataclasses

import lm15
import lmcc
import pytest

import functai
from functai import _ai, ai, assistant, system, turns, user


# ------------------------------------------------------------------ templates


def test_a_template_without_an_output_pattern_reads_the_whole_reply(fake):
    @ai(template=[system("You are a helpful pirate. {instruction}"), user("Text: {text}")])
    def pirate(text: str) -> str:
        """Summarize in 10 words."""
        return _ai

    r = fake("Arr! Foundation models be ready, matey!")
    assert pirate("Foundation models are mature.") == "Arr! Foundation models be ready, matey!"
    assert r.requests[0].system == "You are a helpful pirate. Function: pirate\n\nSummarize in 10 words."
    assert r.user() == "Text: Foundation models are mature."


def test_openai_style_messages_work_too(fake):
    @ai(messages=[{"role": "system", "content": "Be terse. {instruction}"},
                  {"role": "user", "content": "{question}"}], include_fn_name_in_instructions=False)
    def ask(question: str) -> str:
        """Answer."""

    r = fake("Paris.")
    assert ask("Capital of France?") == "Paris."
    assert r.system() == "Be terse. Answer."


def test_a_template_with_a_pattern_reads_several_outputs(fake):
    @ai(template=[
        system("{instruction}\n\n{% for f in outputs %}## {f.name}\n{f.value}\n{% endfor %}"),
        turns(),
        user("{% for f in inputs %}{f.name}: {f.value}\n{% endfor %}"),
    ])
    def analyze(text: str) -> str:
        """Analyze the sentiment."""
        thinking: str = _ai["Think about the tone."]
        return _ai

    r = fake("## thinking\nupbeat\n## result\npositive\n")
    pred = analyze("This is amazing!", all=True)
    assert (pred.thinking, pred.result) == ("upbeat", "positive")
    assert r.user() == "text: This is amazing!\n"


def test_several_outputs_without_a_pattern_refuse_before_any_call(fake):
    @ai(template=[system("{instruction}"), user("{text}")])
    def two(text: str) -> str:
        thinking: str = _ai["think"]
        return _ai

    r = fake()
    with pytest.raises(lmcc.Refusal, match="no output pattern"):
        two("x")
    assert not r.requests


def test_template_errors_surface_at_definition():
    with pytest.raises(lmcc.Refusal):
        @ai(template=[system("{instruction"), user("{x}")])
        def f(x: str) -> str: ...


def test_demos_and_memory_go_where_turns_is(fake):
    @ai(template=[system("{instruction}"), user("Q: {q}")], stateful=True,
        examples=[("2+2", "4")])
    def calc(q: str) -> str:
        """Compute."""

    r = fake("6", "8")
    calc("3+3")
    assert r.roles(0) == ["user", "assistant", "user"]
    assert [p.text for m in r.requests[0].messages for p in m.parts] == ["Q: 2+2", "4", "Q: 3+3"]
    calc("4+4")
    assert [p.text for m in r.requests[1].messages for p in m.parts] == ["Q: 2+2", "4", "Q: 3+3", "6", "Q: 4+4"]


def test_a_prefill_is_sent_to_models_that_continue_it(fake):
    @ai(template=[system("{instruction}\nReply exactly as:\n<answer>{result}</answer>"), user("{text}"),
                  assistant("<answer>")])
    def f(text: str) -> str:
        """Answer."""

    r = fake("Paris</answer>", provider="anthropic", lm="claude-3-5-haiku-latest")   # no extended thinking
    assert f("capital of France") == "Paris"
    assert r.requests[0].messages[-1].role == "assistant"
    assert r.requests[0].messages[-1].parts[0].text == "<answer>"


# ------------------------------------------------------------------ shipped adapters


def test_the_chat_adapter_uses_dspy_sections(fake):
    @ai(adapter="chat")
    def f(question: str) -> int:
        """Add."""
        reasoning: str = _ai
        return _ai

    r = fake("[[ ## reasoning ## ]]\n2+2\n\n[[ ## result ## ]]\n4\n\n[[ ## completed ## ]]")
    assert f("2+2") == 4
    assert "[[ ## question ## ]]\n2+2" in r.user()


def test_the_json_adapter_asks_the_provider_for_a_schema(fake):
    @dataclasses.dataclass
    class Person:
        name: str
        age: int

    @ai(adapter="json")
    def person(text: str) -> Person:
        """Extract the person."""

    r = fake('{"result": {"name": "Ann", "age": 41}}')
    assert person("Ann, 41") == Person("Ann", 41)
    schema = r.requests[0].config.response_format
    assert schema is not None


def test_the_json_adapter_refuses_a_model_without_structured_output(fake):
    @ai(adapter="json")
    def f(x: str) -> str: ...
    fake(provider="groq", lm="groq:llama")
    with pytest.raises(lmcc.Refusal) as err:
        f("x")
    assert err.value.code == "capability-missing"


def test_dspy_adapters_and_modules_are_refused_with_a_way_forward():
    class FakeDspyAdapter:
        pass
    FakeDspyAdapter.__module__ = "dspy.adapters.chat_adapter"

    @ai(adapter=FakeDspyAdapter())
    def f(x: str) -> str: ...
    functai.configure(lm="gpt-4.1-mini")
    with pytest.raises(TypeError, match="adapter='chat'"):
        f.plan()
    with pytest.raises(TypeError, match="module"):
        @ai(module="ProgramOfThought")
        def g(x: str) -> str: ...


# ------------------------------------------------------------------ reasoning


def test_cot_writes_a_reasoning_section_on_models_without_thinking(fake):
    @ai(module="cot")
    def solve(problem: str) -> int:
        """Solve it."""

    r = fake("<reasoning>\n3*7=21, 50-21=29\n</reasoning>\n<result>\n29\n</result>", provider="groq", lm="groq:x")
    pred = solve("change from 50 after 7 pens at 3?", all=True)
    assert pred.result == 29 and pred.reasoning.startswith("3*7")
    assert "Reason step by step" in r.system()


def test_cot_uses_the_native_channel_where_the_model_thinks(fake):
    @ai(module="cot")
    def solve(problem: str) -> int:
        """Solve it."""

    r = fake([lm15.ThinkingPart("7*3=21"), lm15.TextPart("<result>\n29\n</result>")], provider="anthropic",
             lm="claude-sonnet-4-5")
    pred = solve("x", all=True)
    assert (pred.result, pred.reasoning) == (29, "7*3=21")
    assert r.requests[0].config.reasoning is not None
    assert "<reasoning>" not in r.system()


# ------------------------------------------------------------------ tools


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"Sunny and 22C in {city}."


def test_the_tool_loop_with_native_calls(fake):
    @ai(tools=[get_weather])
    def assistant_(question: str) -> str:
        """Answer; use a tool when you need facts."""

    call = lm15.ToolCallPart(id="c1", name="get_weather", input={"city": "Montreal"})
    r = fake([call], "<result>\nSunny, 22C.\n</result>")
    pred = assistant_("Weather in Montreal?", all=True)
    assert pred.result == "Sunny, 22C."
    assert [t.name for t in r.requests[0].tools] == ["get_weather"]
    kinds = [type(p).__name__ for m in r.requests[1].messages for p in m.parts]
    assert "ToolCallPart" in kinds and "ToolResultPart" in kinds
    assert len(pred.turn.steps) == 3          # model call, tool result, model answer


def test_the_tool_loop_with_text_calls(fake):
    @ai(tools=[get_weather])
    def ask(question: str) -> str:
        """Answer."""

    r = fake('```tool\n{"name": "get_weather", "input": {"city": "Oslo"}}\n```',
             "<result>\nSunny.\n</result>", provider="ollama", lm="ollama:llama3")
    assert ask("Weather in Oslo?") == "Sunny."
    assert not r.requests[0].tools and "get_weather" in r.system(0)
    assert "Sunny and 22C in Oslo." in r.user(1)


def test_a_failing_tool_is_reported_to_the_model(fake):
    def boom(x: str) -> str:
        """Fails."""
        raise ValueError("nope")

    @ai(tools=[boom])
    def f(q: str) -> str: ...

    r = fake([lm15.ToolCallPart(id="c1", name="boom", input={"x": "1"})], "<result>\nok\n</result>")
    assert f("q") == "ok"
    results = [p for m in r.requests[1].messages for p in m.parts if type(p).__name__ == "ToolResultPart"]
    assert "ValueError: nope" in str(results[0])


def test_the_step_limit(fake):
    @ai(tools=[get_weather], max_steps=2)
    def f(q: str) -> str: ...
    call = lm15.ToolCallPart(id="c", name="get_weather", input={"city": "X"})
    fake(responder=lambda req: [call])
    with pytest.raises(functai.StepLimit) as err:
        f("q")
    assert err.value.turn is not None


# ------------------------------------------------------------------ memory


def test_stateful_functions_remember_within_a_window(fake):
    @ai(stateful=True, state_window=2)
    def chat(message: str) -> str:
        """A friendly assistant."""

    r = fake(responder=lambda req: "<result>\nok\n</result>")
    chat("Hello, I am Alex."); chat("What is my name?"); chat("And again?")
    assert r.roles(1) == ["user", "assistant", "user"]
    assert "Hello, I am Alex." in r.requests[1].messages[0].parts[0].text
    assert len(r.requests[2].messages) == 5 and len(chat.history) == 2
    chat.reset()
    assert chat.history == []
