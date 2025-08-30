
# FunctAI: The Function-is-the-Prompt Paradigm

Welcome to FunctAI. This library reimagines how Python developers integrate Large Language Models (LLMs) into their applications. FunctAI allows you to treat AI models as reliable, typed Python functions, abstracting away the complexities of prompt engineering and output parsing.

The core philosophy of FunctAI is simple yet powerful:

> **The function definition *is* the prompt, and the function body *is* the program definition.**

By leveraging Python’s native features—docstrings for instructions, type hints for structure, and variable assignments for logic flow—you can define sophisticated AI behaviors with minimal boilerplate.

FunctAI is built on the powerful [DSPy](https://github.com/stanfordnlp/dspy) framework, unlocking advanced strategies like Chain-of-Thought, automatic optimization, and agentic tool usage through an ergonomic, decorator-based API.

-----

## Table of Contents

- [FunctAI: The Function-is-the-Prompt Paradigm](#functai-the-function-is-the-prompt-paradigm)
  - [Table of Contents](#table-of-contents)
  - [1. Getting Started](#1-getting-started)
    - [1.1. Installation](#11-installation)
    - [1.2. Configuration](#12-configuration)
    - [1.3. Your First AI Function](#13-your-first-ai-function)
  - [2. Core Concepts](#2-core-concepts)
    - [2.1. The `@ai` Decorator](#21-the-ai-decorator)
    - [2.2. The `_ai` Sentinel](#22-the-_ai-sentinel)
    - [2.3. Post-processing and Validation](#23-post-processing-and-validation)
  - [3. Structured Output and Type System](#3-structured-output-and-type-system)
    - [3.1. Basic Types](#31-basic-types)
    - [3.2. Dataclasses and Complex Structures](#32-dataclasses-and-complex-structures)
    - [3.3. Enums and Restricted Choices](#33-enums-and-restricted-choices)
  - [4. Configuration and Flexibility](#4-configuration-and-flexibility)
    - [4.1. The Configuration Cascade](#41-the-configuration-cascade)
    - [4.2. Global Configuration](#42-global-configuration)
    - [4.3. Per-Function Configuration](#43-per-function-configuration)
    - [4.4. Contextual Overrides](#44-contextual-overrides)
    - [4.5. Personas and Live Modification](#45-personas-and-live-modification)
  - [5. Advanced Execution Strategies](#5-advanced-execution-strategies)
    - [5.1. Chain of Thought (CoT) Reasoning](#51-chain-of-thought-cot-reasoning)
    - [5.2. Accessing Intermediate Steps](#52-accessing-intermediate-steps)
    - [5.3. Multiple Explicit Outputs](#53-multiple-explicit-outputs)
    - [5.4. Tool Usage (ReAct Agents)](#54-tool-usage-react-agents)
  - [6. Stateful Interactions (Memory)](#6-stateful-interactions-memory)
  - [7. Optimization (In-place Compilation)](#7-optimization-in-place-compilation)
    - [7.1. The Optimization Workflow](#71-the-optimization-workflow)
    - [7.2. Reverting Optimization](#72-reverting-optimization)
  - [8. Inspection and Debugging](#8-inspection-and-debugging)
    - [8.1. Previewing the Prompt](#81-previewing-the-prompt)
    - [8.2. Inspecting the Signature](#82-inspecting-the-signature)
    - [8.3. Viewing History](#83-viewing-history)
  - [9. Best Practices](#9-best-practices)
  - [10. Real-World Examples](#10-real-world-examples)
    - [10.1. Data Extraction Pipeline](#101-data-extraction-pipeline)
    - [10.2. Research Assistant Agent](#102-research-assistant-agent)
  - [11. Migration Guide](#11-migration-guide)
    - [11.1. From OpenAI Functions/Chat API](#111-from-openai-functionschat-api)
    - [11.2. From LangChain](#112-from-langchain)
  - [12. API Reference](#12-api-reference)
    - [Core](#core)
    - [Configuration](#configuration)
    - [`FunctAIFunc` Object](#functaifunc-object)
    - [Utilities](#utilities)

-----

## 1\. Getting Started

### 1.1. Installation

Install FunctAI and its core dependency, DSPy.

```bash
pip install functai
pip install dspy-ai
```

If you plan to use providers like OpenAI, install their respective libraries:

```bash
pip install openai
```

### 1.2. Configuration

Before using FunctAI, you must configure a default Language Model (LM). This requires initializing a DSPy LM provider.

```python
import dspy
from functai import configure

# Initialize a DSPy LM provider (e.g., OpenAI GPT-4o)
# Ensure your OPENAI_API_KEY environment variable is set
gpt4o = dspy.OpenAI(model='gpt-4o')

# Configure FunctAI globally
configure(
    lm=gpt4o,
    temperature=0.0,
    adapter="json" # Highly recommended for robust, structured output
)
```

Setting `adapter="json"` instructs the LM to return structured JSON, which FunctAI uses to ensure outputs strictly adhere to your type hints.

### 1.3. Your First AI Function

Creating an AI function is as simple as defining a standard Python function with type hints and a docstring, then decorating it with `@ai`.

```python
from functai import ai, _ai

@ai
def summarize(text: str, focus: str = "key points") -> str:
    """Summarize the text in one concise sentence,
    concentrating on the specified focus area."""
    # The _ai sentinel represents the LLM output
    return _ai

# Call it exactly like a normal Python function
long_text = "FunctAI bridges the gap between Python's expressive syntax and the dynamic capabilities of LLMs. It allows developers to focus on logic rather than boilerplate."
summary = summarize(long_text, focus="developer benefits")
print(summary)
```

**Example Output:**

```
FunctAI benefits developers by enabling them to integrate LLMs using intuitive Python syntax, prioritizing logic over implementation boilerplate.
```

**What happens when you call `summarize`?**

1.  FunctAI intercepts the call.
2.  It constructs a prompt using the docstring and the inputs (`text`, `focus`).
3.  It invokes the configured LM (GPT-4o).
4.  It parses the LM's response and returns the result, ensuring it matches the return type (`str`).

## 2\. Core Concepts

### 2.1. The `@ai` Decorator

The `@ai` decorator is the magic wand. It transforms a Python function into an LLM-powered program. It analyzes the function's signature (parameters, return type, and docstring) to understand the task requirements.

```python
@ai
def sentiment(text: str) -> str:
    """Analyze the sentiment of the given text.
    Return 'positive', 'negative', or 'neutral'."""
    ... # An empty body or Ellipsis also works like returning _ai
```

### 2.2. The `_ai` Sentinel

The `_ai` object is a special sentinel used within an `@ai` function. It represents the value(s) that will be generated by the AI. It acts as a proxy, deferring the actual LM execution.

Returning `_ai` directly indicates that the LLM's output is the function's final result.

```python
@ai
def extract_price(description: str) -> float:
    """Extract the price from a product description."""
    return _ai
```

### 2.3. Post-processing and Validation

FunctAI encourages writing robust code. You can assign `_ai` to a variable and apply standard Python operations before returning. `_ai` behaves dynamically as if it were the expected return type.

This allows you to combine the power of AI with the reliability of Python code for validation, cleaning, or transformation.

```python
@ai
def sentiment_score(text: str) -> float:
    """Returns a sentiment score between 0.0 (negative) and 1.0 (positive)."""

    # _ai behaves like a float here due to the return type hint
    score = _ai

    # Post-processing: ensure the score is strictly within bounds
    return max(0.0, min(1.0, float(score)))

score = sentiment_score("I think that FunctAI is amazing!")
# Example Output: 1.0
```

## 3\. Structured Output and Type System

FunctAI excels at extracting structured data. Python type hints serve as the contract between your code and the LLM.

### 3.1. Basic Types

FunctAI handles standard Python types (`int`, `float`, `bool`, `str`, `list`, `dict`).

```python
@ai
def calculate(expression: str) -> int:
    """Evaluate the mathematical expression."""
    return _ai

result = calculate("What is 15 times 23?")
# Output: 345 (as an integer)

@ai
def get_keywords(article: str) -> list[str]:
    """Extract 5 key terms from the article."""
    keywords = _ai
    # Post-processing example: ensure lowercase
    return [k.lower() for k in keywords]
```

### 3.2. Dataclasses and Complex Structures

For complex data extraction, define a `dataclass` (or Pydantic model) and use it as the return type. This is highly reliable when combined with `adapter="json"`.

```python
from dataclasses import dataclass
from typing import List

@dataclass
class ProductInfo:
    name: str
    price: float
    features: List[str]
    in_stock: bool

@ai(adapter="json") # Explicitly ensuring JSON adapter for robustness
def extract_product(description: str) -> ProductInfo:
    """Extract product information from the description."""
    return _ai

info = extract_product("iPhone 15 Pro - $999, 5G, titanium design, available now")

# The output is a validated ProductInfo instance
print(info)
# Output: ProductInfo(name='iPhone 15 Pro', price=999.0, features=['5G', 'titanium design'], in_stock=True)
```

### 3.3. Enums and Restricted Choices

Use `Enum` to restrict the LM's output to a predefined set of values, increasing reliability for classification tasks.

```python
from enum import Enum

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@ai
def classify_priority(issue_description: str) -> TicketPriority:
    """Analyzes the issue and classifies its priority level."""
    return _ai

result = classify_priority(text="The main database is unresponsive.")
# Output: <TicketPriority.HIGH: 'high'>
```

## 4\. Configuration and Flexibility

### 4.1. The Configuration Cascade

FunctAI uses a flexible, cascading configuration system. Settings are applied in the following order of precedence (highest to lowest):

1.  **Function-Level** (e.g., `@ai(temperature=0.1)`)
2.  **Contextual** (e.g., `with defaults(temperature=0.1):`)
3.  **Global** (e.g., `configure(temperature=0.1)`)

### 4.2. Global Configuration

Use `functai.configure()` for project-wide defaults.

```python
import functai
import dspy

# Example using GPT-3.5-Turbo as a default
functai.configure(
    lm=dspy.OpenAI(model="gpt-3.5-turbo"),
    temperature=0.5,
    adapter="json",       # "json" (structured) or "chat" (conversational)
    module="predict",     # Default execution strategy: "predict", "cot", "react"
    stateful=False
)
```

### 4.3. Per-Function Configuration

Override defaults for specific functions directly in the decorator. This is useful when different tasks require different models or creativity levels.

```python
# Deterministic task requiring a powerful model
@ai(temperature=0.0, lm="gpt-4o")
def legal_analysis(document: str) -> str:
    """Provide precise legal analysis of the document."""
    return _ai

# Creative task using a different provider
@ai(temperature=0.9, lm="claude-3.5-sonnet")
def creative_story(prompt: str) -> str:
    """Write a creative story based on the prompt."""
    return _ai
```

### 4.4. Contextual Overrides

Use the `functai.defaults()` context manager to temporarily override defaults for a block of code.

```python
from functai import defaults

@ai
def analyze(data): return _ai

analyze("data1") # Uses global defaults

# Temporarily switch model and temperature
with defaults(temperature=0.0, lm="gpt-4o"):
    analyze("data2") # Uses GPT-4o, Temp 0.0

analyze("data3") # Back to global defaults
```

### 4.5. Personas and Live Modification

You can define a `persona` to set the AI's role, which is prepended to the function's instructions (docstring). You can also modify configuration properties live.

```python
@ai
def translator(text: str) -> str:
    """Translate text."""
    return _ai

# Set a specific persona
translator.persona = "You are an expert literary translator specializing in French literature."

# Change the temperature on the fly
translator.temperature = 0.2

# Now the translation will be influenced by the persona and new temperature
french_text = translator("To be or not to be, that is the question.")
```

## 5\. Advanced Execution Strategies

FunctAI truly shines by allowing developers to define complex execution strategies directly within the function body, adhering to the "function body is the program definition" philosophy.

### 5.1. Chain of Thought (CoT) Reasoning

Eliciting step-by-step reasoning (Chain of Thought) often significantly improves the quality and accuracy of the final answer, especially for complex tasks.

In FunctAI, you define CoT by declaring intermediate reasoning steps within the function body using `_ai` assignments with descriptions.

```python
@ai
def solve_math_problem(question: str) -> float:
    """Solves a math word problem and returns the numerical answer."""

    # Define the reasoning step:
    # 1. The variable name ('reasoning') becomes the field name.
    # 2. The type hint (str) defines the output type for this step.
    # 3. The subscript _ai["..."] provides specific instructions for the LLM.
    reasoning: str = _ai["Step-by-step thinking process to reach the solution."]

    # The final return value (float) is the main output
    return _ai
```

**Behavior:** FunctAI analyzes the function body, detects the intermediate `reasoning` assignment, and automatically configures the execution to generate the reasoning *before* attempting to generate the final result.

*(Note: You can also enable a generic CoT by setting `@ai(module="cot")`, but the explicit definition above offers more control.)*

### 5.2. Accessing Intermediate Steps

While the function call normally returns only the final result, you can access the intermediate steps (like `reasoning`) by adding the special argument `_prediction=True` to the function call.

This returns the raw prediction object containing all generated fields.

```python
question = "If a train travels 120 miles in 2 hours, what is its speed?"
prediction = solve_math_problem(question, _prediction=True)

print("--- Reasoning ---")
print(prediction.reasoning)
# Example Output: 1. Identify distance (120 miles) and time (2 hours). 2. Use formula speed = distance/time...

print("\n--- Answer ---")
# When _ai is returned directly, the main output is stored in the 'result' attribute
print(prediction.result)
# Example Output: 60.0
```

### 5.3. Multiple Explicit Outputs

You can define and return multiple distinct outputs from a single function call by declaring them inline, similar to the CoT pattern.

```python
from typing import Tuple

@ai
def critique_and_improve(text: str) -> Tuple[str, str]:
    """
    Analyze the text, provide constructive criticism, and suggest an improved version.
    """
    # Define explicit output fields using _ai[...]
    critique: str = _ai["Constructive criticism focusing on clarity and tone."]
    improved_text: str = _ai["The improved version of the text."]

    # Return the materialized fields (Python handles the Tuple structure)
    return critique, improved_text

critique, improved = critique_and_improve(text="U should fix this asap, it's broken.")
```

### 5.4. Tool Usage (ReAct Agents)

FunctAI supports the ReAct (Reasoning + Acting) pattern for creating agents that can interact with external tools. Tools are standard, typed Python functions.

When the `tools` argument is provided to the `@ai` decorator, the execution strategy automatically upgrades to an agentic loop (using `dspy.ReAct`).

```python
# 1. Define tools
def search_web(query: str) -> str:
    """Searches the web for information. (Mock implementation)"""
    print(f"[Tool executing: Searching for '{query}']")
    # In a real scenario, this would call a search API
    return f"Mock search results for {query}."

def calculate(expression: str) -> float:
     """Performs mathematical calculations. (Mock implementation)"""
     print(f"[Tool executing: Calculating '{expression}']")
     # WARNING: eval() is unsafe in production. Use a safe math library.
     return eval(expression)

# 2. Define the AI function with access to the tools
@ai(tools=[search_web, calculate])
def research_assistant(question: str) -> str:
    """Answer questions using available tools to gather data and perform calculations."""
    return _ai

# 3. Execute the agent
# The AI will potentially use search_web and then calculate.
answer = research_assistant("What is the result of (15 * 23) + 10?")
```

**Behavior:** The function will iteratively think about the task, decide which tool to use, execute the tool, observe the results, and repeat until it can provide the final answer.

## 6\. Stateful Interactions (Memory)

By default, `@ai` functions are stateless; each call is independent. To maintain context across calls (e.g., in a chatbot scenario), set `stateful=True`.

```python
@ai(stateful=True)
def assistant(message: str) -> str:
    """A friendly AI assistant that remembers the conversation history."""
    return _ai

response1 = assistant("Hello, my name is Alex.")
# Output: Nice to meet you, Alex!

response2 = assistant("What is my name?")
# Output: Your name is Alex.
```

**Behavior:** When `stateful=True`, FunctAI automatically includes the history of previous inputs and outputs in the context of the next call.

You can manage the state explicitly using the `.state` attribute:

```python
assistant.state.clear() # Reset the conversation history
```

## 7\. Optimization (In-place Compilation)

FunctAI integrates seamlessly with DSPy's optimization capabilities (Teleprompters). Optimization (often called compilation in DSPy) improves the quality and reliability of your AI functions by using a dataset of examples.

The optimizer can automatically generate effective few-shot examples or refine instructions. This happens *in place* using the `.opt()` method on the function object.

### 7.1. The Optimization Workflow

```python
from dspy import Example
from functai import ai, _ai

# 1. Define the function
@ai
def classify_intent(user_query: str) -> str:
    """Classify user intent as 'booking', 'cancelation', or 'information'."""
    return _ai

# 2. Define the training data (List of DSPy Examples)
# .with_inputs() specifies which keys are inputs to the function
trainset = [
    Example(user_query="I need to reserve a room.", result="booking").with_inputs("user_query"),
    Example(user_query="How do I get there?", result="information").with_inputs("user_query"),
    Example(user_query="I want to cancel my reservation.", result="cancelation").with_inputs("user_query"),
]

# 3. Optimize the function in place
# strategy="launch" typically uses a default like BootstrapFewShot
print("Optimizing...")
classify_intent.opt(trainset=trainset, strategy="launch")
print("Optimization complete.")

# 4. The function is now optimized (it includes generated few-shot examples in its prompt)
result = classify_intent("Can I book a suite for next Tuesday?")
# Output: "booking"
```

### 7.2. Reverting Optimization

FunctAI tracks optimization steps. If the results are not satisfactory, you can revert using `.undo_opt()`.

```python
# Revert the last optimization step
classify_intent.undo_opt(steps=1)
```

## 8\. Inspection and Debugging

Understanding what happens under the hood is crucial. FunctAI provides utilities to inspect the prompts being sent to the LM and the execution history.

### 8.1. Previewing the Prompt

Use `functai.format_prompt` to preview the exact messages that will be sent to the LM for a given set of inputs, without actually calling the LM.

```python
from functai import format_prompt

preview = format_prompt(summarize, text="A long document...", focus="technical details")
print(preview['render'])
```

This output will show you how the adapter, module, persona, and inputs combine to form the final prompt.

### 8.2. Inspecting the Signature

FunctAI automatically computes a DSPy Signature based on your Python function definition. You can inspect this derived signature using `functai.signature_text`.

```python
from functai import signature_text

# Inspect the signature of the CoT function from section 5.1
print(signature_text(solve_math_problem))
```

**Example Output:**

```
Signature: SolveMathProblemSig
Doc: Solves a math word problem and returns the numerical answer.
Inputs:
- question: <class 'str'>
Outputs:
- reasoning: <class 'str'> (Description: Step-by-step thinking process...)
- result: <class 'float'> (primary)
```

### 8.3. Viewing History

To see the history of recent LM calls, including the actual prompts sent and the raw responses received, use `functai.inspect_history_text()`.

```python
from functai import inspect_history_text

# After running some AI functions...
print(inspect_history_text(n=1)) # Show the last call
```

## 9\. Best Practices

1.  **Clear and Specific Docstrings:** The docstring is the primary instruction. Be descriptive. If you need a specific format, tone, or constraint, mention it explicitly in the docstring.
2.  **Type Hints Are Contracts:** Always provide comprehensive type hints. They guide the AI, validate outputs, and make your code robust. Use `dataclasses` for complex structures and `Enum` for classifications.
3.  **Use `adapter="json"` for Structure:** When extracting data (lists, dataclasses, dicts), configure the adapter to "json" for maximum reliability.
4.  **Post-Process for Reliability:** Don't trust the AI blindly for critical applications. Use the function body to validate and clean the AI's output (e.g., checking bounds, validating email formats).
5.  **Use Lower Temperatures for Facts:** For deterministic tasks (like data extraction or calculation), use `temperature=0.0` or `0.1`. Use higher temperatures (e.g., `0.7`) for creative tasks.
6.  **Leverage Explicit Chain of Thought:** For any non-trivial task, explicitly defining a `reasoning: str = _ai["..."]` step will almost always improve the result and make debugging easier.
7.  **Optimize for Production:** Before deploying a critical AI function, create a small dataset and use `.opt()` to improve its performance with few-shot examples.

## 10\. Real-World Examples

### 10.1. Data Extraction Pipeline

This example demonstrates chaining multiple AI functions to process unstructured data reliably.

```python
from dataclasses import dataclass
from typing import List
from functai import ai, _ai, configure

# Ensure deterministic extraction
configure(temperature=0.0, adapter="json")

# 1. Define the target structure
@dataclass
class Invoice:
    invoice_number: str
    vendor_name: str
    total: float
    items: List[str]

# 2. Define the extraction function with CoT for accuracy
@ai
def extract_invoice(document_text: str) -> Invoice:
    """Extract invoice information from the document text.
    Parse all relevant fields accurately. Convert amounts to float.
    """
    thought_process: str = _ai["Analyze the document layout and identify the location of each field before extracting."]
    return _ai

# 3. Define a validation function
@ai
def validate_invoice(invoice: Invoice) -> bool:
    """Validate if the invoice data is complete and reasonable.
    Check if the total is positive and required fields are present.
    """
    return _ai

# 4. Define a summarization function
@ai
def summarize_invoice(invoice: Invoice) -> str:
    """Create a brief, human-readable summary of the invoice."""
    return _ai

# 5. Execute the pipeline
document = """
INVOICE
Vendor: TechCorp Inc.
Invoice #: INV-2025-101
Items: 5x Laptops, 2x Monitors
Total: $5600.00
"""

invoice = extract_invoice(document)

if validate_invoice(invoice):
    summary = summarize_invoice(invoice)
    print("Invoice Validated Successfully!")
    print(summary)
else:
    print("Invoice Validation Failed.")
```

### 10.2. Research Assistant Agent

This example builds a sophisticated agent using tools and structured internal outputs.

```python
# Define Tools (Placeholders)
def search_web(query: str) -> str:
    """Search the web for information."""
    print(f"[Searching: {query}]")
    return f"Mock search results for {query}."

def read_paper(paper_id: str) -> str:
    """Read the content of a specific research paper."""
    print(f"[Reading: {paper_id}]")
    return f"Mock content of paper {paper_id}."

# Define the Agent
@ai(tools=[search_web, read_paper])
def research_assistant(query: str) -> str:
    """Advanced research assistant.
    Use available tools to gather information. Synthesize findings.
    """
    # Define intermediate outputs for better structure and inspection
    research_notes: list[str] = _ai["Key findings gathered during the ReAct process."]
    confidence: str = _ai["Confidence level: high/medium/low."]
    sources: list[str] = _ai["Sources consulted during the ReAct process."]

    answer = _ai

    # Post-processing: Add metadata to the final response
    return f"{answer}\n\nConfidence: {confidence}\nSources: {', '.join(sources)}"

# Execution
result = research_assistant("What are the latest breakthroughs in quantum computing?")
```

## 11\. Migration Guide

FunctAI simplifies patterns commonly used in other frameworks by removing boilerplate.

### 11.1. From OpenAI Functions/Chat API

Traditional interaction requires explicit prompt construction and response handling.

```python
# Before (Conceptual)
import openai
client = openai.OpenAI()

def summarize_text(text):
    prompt = f"Summarize the following text in one sentence:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# After (FunctAI)
from functai import ai, _ai

@ai(lm="gpt-4o")
def summarize(text: str) -> str:
    """Summarize the text in one sentence."""
    return _ai
```

### 11.2. From LangChain

LangChain often requires assembling Chains and PromptTemplates explicitly.

```python
# Before (Conceptual LangChain)
# Requires defining LLM, PromptTemplate, and LLMChain...
# chain = LLMChain(llm=llm, prompt=prompt_template)
# result = chain.run(input=input_text)

# After (FunctAI)
@ai
def my_chain(input: str) -> str:
    """Template instructions here."""
    return _ai
```

## 12\. API Reference

### Core

  * `@ai(_fn=None, **cfg)`: Decorator to create an AI function (`FunctAIFunc`).
      * Config options: `lm`, `temperature`, `adapter` ("json", "chat"), `module` ("predict", "cot", "react"), `tools` (list of functions), `stateful` (bool), `persona` (str).
  * `_ai`: Sentinel object representing the LM output.
      * `_ai["description"]`: Used within the function body (e.g., `var: type = _ai["desc"]`) to define intermediate (e.g., CoT) or explicit output fields with specific instructions.

### Configuration

  * `configure(**cfg)`: Set global defaults.
  * `defaults(**overrides)`: Context manager for temporary default overrides.
  * `settings`: Direct access to the global configuration object.

### `FunctAIFunc` Object

The callable object returned by `@ai`.

  * `__call__(*args, _prediction=False, **kwargs)`: Executes the AI program.
      * If `_prediction=True`, returns the full `dspy.Prediction` object containing all intermediate and final outputs.
  * `.opt(trainset, strategy="launch", **opts)`: Optimize the function in place using a dataset (list of `dspy.Example`).
  * `.undo_opt(steps=1)`: Revert optimization steps.
  * `.state`: Access state management (for `stateful=True` functions).
      * `.state.clear()`: Resets the conversation history.
      * `.state.history`: Access the raw history data.
  * Mutable properties: `lm`, `temperature`, `adapter`, `module`, `tools`, `persona` can be modified after the function is defined.

### Utilities

  * `format_prompt(fn_or_prog, /, **inputs)`: Preview the formatted prompt for given inputs without calling the LM. Returns a dictionary including the rendered prompt string.
  * `compute_signature(fn_or_prog)`: Compute and return the underlying `dspy.Signature`.
  * `signature_text(fn_or_prog)`: Return a human-readable summary of the signature.
  * `inspect_history_text(n=5)`: Return the recent LM call history (prompts and responses) as text.