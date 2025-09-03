# FunctAI Specification: The Function-is-the-Prompt Paradigm


- [FunctAI Specification: The Function-is-the-Prompt Paradigm](#functai-specification-the-function-is-the-prompt-paradigm)
- [FunctAI: The Function-is-the-Prompt Paradigm](#functai-the-function-is-the-prompt-paradigm)
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
    - [3.3. Restricted Choices](#33-restricted-choices)
  - [4. Configuration and Flexibility](#4-configuration-and-flexibility)
    - [4.1. The Configuration Cascade](#41-the-configuration-cascade)
    - [4.2. Global Configuration](#42-global-configuration)
    - [4.3. Per-Function Configuration](#43-per-function-configuration)
    - [4.4. Contextual Overrides](#44-contextual-overrides)
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
  - [10. Real-World Examples](#10-real-world-examples)
    - [10.1. Data Extraction Pipeline](#101-data-extraction-pipeline)
    - [10.2. Research Assistant Agent](#102-research-assistant-agent)

# FunctAI: The Function-is-the-Prompt Paradigm

Welcome to FunctAI. This library reimagines how Python developers
integrate Large Language Models (LLMs) into their applications. FunctAI
allows you to treat AI models as reliable, typed Python functions,
abstracting away the complexities of prompt engineering and output
parsing.

The core philosophy of FunctAI is simple yet powerful:

> **The function definition *is* the prompt, and the function body *is*
> the program definition.**

By leveraging Python’s native features—docstrings for instructions, type
hints for structure, and variable assignments for logic flow—you can
define sophisticated AI behaviors with minimal boilerplate.

FunctAI is built on the powerful
[DSPy](https://github.com/stanfordnlp/dspy) framework, unlocking
advanced strategies like Chain-of-Thought, automatic optimization, and
agentic tool usage through an ergonomic, decorator-based API.

------------------------------------------------------------------------

## 1. Getting Started

### 1.1. Installation

Install FunctAI and its core dependency, DSPy.

``` bash
pip install functai
```

### 1.2. Configuration

Before using FunctAI, you must configure a default Language Model (LM).
This requires initializing a DSPy LM provider.

``` python
from functai import configure

# Configure FunctAI globally
configure(lm="gpt-4.1", temperature=1.0, api_key="<YOUR_API_KEY>")
```

### 1.3. Your First AI Function

Creating an AI function is as simple as defining a standard Python
function with type hints and a docstring, then decorating it with `@ai`.

``` python
from functai import ai, _ai

@ai
def summarize(text: str, focus: str = "key points") -> str:
    """Summarize the text in one concise sentence,
    concentrating on the specified focus area."""
    # The _ai sentinel represents the LLM output
    return _ai

# Call it exactly like a normal Python function
long_text = "FunctAI bridges the gap between Python's expressive syntax and the dynamic capabilities of LLMs. It allows developers to focus on logic rather than boilerplate."
summarize(long_text, focus="developer benefits")
```

    "FunctAI enables developers to concentrate on core logic by reducing boilerplate and leveraging Python's syntax with LLM capabilities."

**What happens when you call `summarize`?**

1.  FunctAI intercepts the call.
2.  It constructs a prompt using the docstring and the inputs (`text`,
    `focus`).
3.  It invokes the configured LM (GPT-4.1).
4.  It parses the LM’s response and returns the result, ensuring it
    matches the return type (`str`).

## 2. Core Concepts

### 2.1. The `@ai` Decorator

The `@ai` decorator is the magic wand. It transforms a Python function
into an LLM-powered program. It analyzes the function’s signature
(parameters, return type, and docstring) to understand the task
requirements and a prompt is automatically constructed out of that for
you.

``` python
@ai
def sentiment(text: str) -> str:
    """Analyze the sentiment of the given text.
    Return 'positive', 'negative', or 'neutral'."""
    ... # An empty body or Ellipsis also works like returning _ai
```

### 2.2. The `_ai` Sentinel

The `_ai` object is a special sentinel used within an `@ai` function. It
represents the value(s) that will be generated by the AI. It acts as a
proxy, deferring the actual LM execution.

Returning `_ai` directly indicates that the LLM’s output is the
function’s final result.

``` python
@ai
def extract_price(description: str) -> float:
    """Extract the price from a product description."""
    return _ai
```

### 2.3. Post-processing and Validation

FunctAI encourages writing robust code. You can assign `_ai` to a
variable and apply standard Python operations before returning. `_ai`
behaves dynamically as if it were the expected return type.

This allows you to combine the power of AI with the reliability of code
for validation, cleaning, or transformation.

``` python
@ai
def sentiment_score(text: str) -> float:
    """Returns a sentiment score between 0.0 (negative) and 1.0 (positive)."""

    # _ai behaves like a float here due to the return type hint
    score = _ai

    # Post-processing: ensure the score is strictly within bounds
    return max(0.0, min(1.0, float(score)))

sentiment_score("I think that FunctAI is amazing!")
```

    0.95

## 3. Structured Output and Type System

FunctAI excels at extracting structured data. Python type hints serve as
the contract between your code and the LLM.

### 3.1. Basic Types

FunctAI handles standard Python types (`int`, `float`, `bool`, `str`,
`list`, `dict`).

``` python
@ai
def calculate(expression) -> int:
    """Evaluate the mathematical expression."""
    return _ai

result = calculate("What is 15 times 23?")
print(result)
```

    345

``` python
@ai
def get_keywords(article) -> list[str]:
    """Extract 5 key terms from the article."""
    keywords: list[str] = _ai
    # Post-processing example: ensure lowercase
    return [k.lower() for k in keywords]

get_keywords("FunctAI excels at extracting structured data. Python type hints serve as the contract between your code and the LLM.")
```

    ['functai', 'structured data', 'python type hints', 'contract', 'llm']

### 3.2. Dataclasses and Complex Structures

For complex data extraction, define a `dataclass` (or Pydantic model)
and use it as the return type.

``` python
from dataclasses import dataclass
from typing import List

@dataclass
class ProductInfo:
    name: str
    price: float
    features: List[str]
    in_stock: bool

@ai
def extract_product(description: str) -> ProductInfo:
    """Extract product information from the description."""
    return _ai

info = extract_product("iPhone 15 Pro - $999, 5G, titanium design, available now")

# The output is a validated ProductInfo instance
print(info)
```

    ProductInfo(name='iPhone 15 Pro', price=999.0, features=['5G', 'titanium design'], in_stock=True)

### 3.3. Restricted Choices

Use `Enum` to restrict the LM’s output to a predefined set of values,
increasing reliability for classification tasks.

``` python
from enum import Enum

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@ai
def classify_priority(issue_description: str) -> TicketPriority:
    """Analyzes the issue and classifies its priority level."""
    return _ai

result = classify_priority(issue_description="The main database is unresponsive.")
result
```

    <TicketPriority.HIGH: 'high'>

You can also use typing `Literal`:

``` python
from typing import Literal

@ai
def categorize(text: str) -> Literal["sport", "fashion"]:
    ...

categorize("Vibe Coding as been declared a sport")
```

    'sport'

## 4. Configuration and Flexibility

### 4.1. The Configuration Cascade

FunctAI uses a flexible, cascading configuration system. Settings are
applied in the following order of precedence (highest to lowest):

1.  **Function-Level** (e.g., `@ai(temperature=0.1)`)
2.  **Contextual** (e.g., `with configure(temperature=0.1):`)
3.  **Global** (e.g., `configure(temperature=0.1)`)

### 4.2. Global Configuration

Use `functai.configure()` for project-wide defaults.

``` python
import functai

# Example using GPT-3.5-Turbo as a default
functai.configure(
    lm="groq/openai/gpt-oss-120b",
    temperature=0.5
)

@ai
def math(formal):
    ...

math("2+2")
```

    2025/08/30 13:32:38 WARNING dspy.adapters.json_adapter: Failed to use structured output format, falling back to JSON mode.

    4

### 4.3. Per-Function Configuration

Override defaults for specific functions directly in the decorator. This
is useful when different tasks require different models or creativity
levels.

``` python
# Deterministic task requiring a powerful model
@ai(temperature=0.0, lm="openai/gpt-4o")
def legal_analysis(document):
    """Provide precise legal analysis of the document."""
    return _ai

# Creative task using a different provider
@ai(temperature=0.9, lm="anthropic/claude-sonnet-4-20250514")
def creative_story(prompt):
    """Write a creative story based on the prompt."""
    return _ai
```

### 4.4. Contextual Overrides

Use the `functai.configure()` context manager to temporarily override
defaults for a block of code.

``` python
from functai import configure

@ai
def analyze(data): return _ai

analyze("data1") # Uses global defaults

# Temporarily switch model and temperature
with configure(temperature=0.0, lm="gpt-4o"):
    print(analyze("data2")) # Uses GPT-4.1, Temp 0.0

analyze("data3") # Back to global defaults
```

    2025/08/30 13:32:43 WARNING dspy.adapters.json_adapter: Failed to use structured output format, falling back to JSON mode.
    2025/08/30 13:32:49 WARNING dspy.adapters.json_adapter: Failed to use structured output format, falling back to JSON mode.

    data2

    'data3'

## 5. Advanced Execution Strategies

FunctAI truly shines by allowing developers to define complex execution
strategies directly within the function body, adhering to the “function
body is the program definition” philosophy.

### 5.1. Chain of Thought (CoT) Reasoning

Eliciting step-by-step reasoning (Chain of Thought) often significantly
improves the quality and accuracy of the final answer, especially for
complex tasks.

In FunctAI, you define CoT by declaring intermediate reasoning steps
within the function body using `_ai` assignments with descriptions.

``` python
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

**Behavior:** FunctAI analyzes the function body, detects the
intermediate `reasoning` assignment, and automatically configures the
execution to generate the reasoning *before* attempting to generate the
final result.

*(Note: You can also enable a generic CoT by setting
`@ai(module="cot")`, but the explicit definition above offers more
control.)*

### 5.2. Accessing Intermediate Steps

While the function call normally returns only the final result, you can
access the intermediate steps (like `reasoning`) by adding the special
argument `all=True` to the function call.

This returns the raw prediction object containing all generated fields.

``` python
question = "If a train travels 120 miles in 2 hours, what is its speed?"
prediction = solve_math_problem(question, all=True)

print("--- Reasoning ---")
print(prediction.reasoning)

print("\n--- Answer ---")
# When _ai is returned directly, the main output is stored in the 'result' attribute
print(prediction.result)
```

    --- Reasoning ---
    Speed is calculated by dividing distance by time. The train travels 120 miles in 2 hours, so speed = 120 miles / 2 hours = 60 miles per hour.

    --- Answer ---
    60.0

### 5.3. Multiple Explicit Outputs

You can define and return multiple distinct outputs from a single
function call by declaring them inline, similar to the CoT pattern.

``` python
from typing import Tuple

@ai()
def critique_and_improve(text: str) -> Tuple[str, str, int]:
    """
    Analyze the text, provide constructive criticism, and suggest an improved version.
    """
    # Define explicit output fields using _ai[...]
    critique: str = _ai["Constructive criticism focusing on clarity and tone."]
    improved_text: str = _ai["The improved version of the text."]

    # Return the materialized fields (Python handles the Tuple structure)
    return critique, improved_text, 1.0

critique, improved, number = critique_and_improve(text="U should fix this asap, it's broken.")


print("--- Critique ---")
print(critique)
print("--- Improved ---")
print(improved)
```

    --- Critique ---
    The original message is overly informal and uses shorthand ('U') and vague urgency ('asap') that may come across as unprofessional. It lacks specific details about what is broken, which can make it harder for the recipient to address the issue efficiently. A clearer, more courteous tone with a brief description of the problem would improve communication.
    --- Improved ---
    Please address this issue as soon as possible; the current functionality appears to be broken.

### 5.4. Tool Usage (ReAct Agents)

FunctAI supports the ReAct (Reasoning + Acting) pattern for creating
agents that can interact with external tools. Tools are standard, typed
Python functions.

When the `tools` argument is provided to the `@ai` decorator, the
execution strategy automatically upgrades to an agentic loop (using
`dspy.ReAct`).

``` python
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

    [Tool executing: Calculating '(15 * 23) + 10']

    2025/08/30 13:32:58 WARNING dspy.adapters.json_adapter: Failed to use structured output format, falling back to JSON mode.

``` python
import functai
#| echo: false
print(functai.phistory())
```





    [2025-08-30T13:32:58.957207]

    System message:

    Your input fields are:
    1. `question` (str): 
    2. `trajectory` (str):
    Your output fields are:
    1. `reasoning` (str): 
    2. `result` (str):
    All interactions will be structured in the following way, with the appropriate values filled in.

    Inputs will have the following structure:

    [[ ## question ## ]]
    {question}

    [[ ## trajectory ## ]]
    {trajectory}

    Outputs will be a JSON object with the following fields.

    {
      "reasoning": "{reasoning}",
      "result": "{result}"
    }
    In adhering to this structure, your objective is: 
            Function: research_assistant
            
            Answer questions using available tools to gather data and perform calculations.


    User message:

    [[ ## question ## ]]
    What is the result of (15 * 23) + 10?

    [[ ## trajectory ## ]]
    [[ ## thought_0 ## ]]
    I need to compute the arithmetic expression (15 * 23) + 10.

    [[ ## tool_name_0 ## ]]
    calculate

    [[ ## tool_args_0 ## ]]
    {"expression": "(15 * 23) + 10"}

    [[ ## observation_0 ## ]]
    355

    [[ ## thought_1 ## ]]
    The calculation is complete; the result of (15 * 23) + 10 is 355.

    [[ ## tool_name_1 ## ]]
    finish

    [[ ## tool_args_1 ## ]]
    {}

    [[ ## observation_1 ## ]]
    Completed.

    Respond with a JSON object in the following order of fields: `reasoning`, then `result`.


    Response:

    {"reasoning":"I used the calculate tool to evaluate the expression (15 * 23) + 10, which gave 355. No further steps are needed.","result":"355"}




**Behavior:** The function will iteratively think about the task, decide
which tool to use, execute the tool, observe the results, and repeat
until it can provide the final answer.

## 6. Stateful Interactions (Memory)

By default, `@ai` functions are stateless; each call is independent. To
maintain context across calls (e.g., in a chatbot scenario), set
`stateful=True`.

``` python
from functai import ai, _ai

@ai(lm="gpt-4.1", stateful=True)
def assistant(message):
    """A friendly AI assistant that remembers the conversation history."""
    return _ai

response1 = assistant("Hello, my name is Alex.")
print(f"Output1: {response1}")

response2 = assistant("What is my name?")
print(f"Output2: {response2}")
```

    Output1: Hello Alex! It's nice to meet you. How can I assist you today?
    Output2: Your name is Alex.





    [2025-08-30T13:32:59.675591]

    System message:

    Your input fields are:
    1. `message` (str): 
    2. `history` (History):
    Your output fields are:
    1. `result` (Any):
    All interactions will be structured in the following way, with the appropriate values filled in.

    Inputs will have the following structure:

    [[ ## message ## ]]
    {message}

    [[ ## history ## ]]
    {history}

    Outputs will be a JSON object with the following fields.

    {
      "result": "{result}        # note: the value you produce must adhere to the JSON schema: {}"
    }
    In adhering to this structure, your objective is: 
            Function: assistant
            
            A friendly AI assistant that remembers the conversation history.


    User message:

    [[ ## message ## ]]
    Hello, my name is Alex.


    Assistant message:

    {
      "result": "Hello Alex! It's nice to meet you. How can I assist you today?"
    }


    User message:

    [[ ## message ## ]]
    What is my name?

    Respond with a JSON object in the following order of fields: `result` (must be formatted as a valid Python Any).


    Response:

    {
      "result": "Your name is Alex."
    }




**Behavior:** When `stateful=True`, FunctAI automatically includes the
history of previous inputs and outputs in the context of the next call.

## 7. Optimization (In-place Compilation)

**WARNING: optimization is still a work in progress**

FunctAI integrates seamlessly with DSPy’s optimization capabilities
(Teleprompters). Optimization (often called compilation in DSPy)
improves the quality and reliability of your AI functions by using a
dataset of examples.

The optimizer can automatically generate effective few-shot examples or
refine instructions. This happens *in place* using the `.opt()` method
on the function object.

### 7.1. The Optimization Workflow

``` python
import dspy
from dspy import Example
from functai import ai, _ai

dspy.configure(lm = dspy.LM("gpt-4.1"))

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
classify_intent.opt(trainset=trainset)
print("Optimization complete.")

# 4. The function is now optimized (it includes generated few-shot examples in its prompt)
result = classify_intent("Can I book a suite for next Tuesday?")
# Output: "booking"
```

    Optimizing...

      0%|          | 0/3 [00:00<?, ?it/s]100%|██████████| 3/3 [00:00<00:00, 36.98it/s]

    Bootstrapped 3 full traces after 2 examples for up to 1 rounds, amounting to 3 attempts.
    Optimization complete.


    2025/08/30 13:33:04 WARNING dspy.adapters.json_adapter: Failed to use structured output format, falling back to JSON mode.

### 7.2. Reverting Optimization

FunctAI tracks optimization steps. If the results are not satisfactory,
you can revert using `.undo_opt()`.

``` python
# Revert the last optimization step
classify_intent.undo_opt(steps=1)
```

## 8. Inspection and Debugging

**More to come**

To see the last call, use `functai.phistory()`.

``` python
from functai import phistory

# After running some AI functions...
print(phistory()) # Show the last call
```





    [2025-08-30T13:33:06.742620]

    System message:

    Your input fields are:
    1. `user_query` (str):
    Your output fields are:
    1. `result` (str):
    All interactions will be structured in the following way, with the appropriate values filled in.

    Inputs will have the following structure:

    [[ ## user_query ## ]]
    {user_query}

    Outputs will be a JSON object with the following fields.

    {
      "result": "{result}"
    }
    In adhering to this structure, your objective is: 
            Function: classify_intent
            
            Classify user intent as 'booking', 'cancelation', or 'information'.


    User message:

    [[ ## user_query ## ]]
    I need to reserve a room.


    Assistant message:

    {
      "result": "booking"
    }


    User message:

    [[ ## user_query ## ]]
    How do I get there?


    Assistant message:

    {
      "result": "information"
    }


    User message:

    [[ ## user_query ## ]]
    I want to cancel my reservation.


    Assistant message:

    {
      "result": "cancelation"
    }


    User message:

    [[ ## user_query ## ]]
    Can I book a suite for next Tuesday?

    Respond with a JSON object in the following order of fields: `result`.


    Response:

    {
      "result": "booking"
    }




## 10. Real-World Examples

### 10.1. Data Extraction Pipeline

This example demonstrates chaining multiple AI functions to process
unstructured data reliably.

``` python
from dataclasses import dataclass
from typing import List
from functai import ai, _ai, configure

# Ensure deterministic extraction
configure(lm = "gpt-4.1", temperature=0.0, adapter="json")

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

    Invoice Validated Successfully!
    Invoice INV-2025-101 from TechCorp Inc. totals $5,600.00 and includes 5 laptops and 2 monitors.

### 10.2. Research Assistant Agent

This example builds a sophisticated agent using tools and structured
internal outputs.

``` python
from functai import ai, _ai, configure

configure(lm = "gpt-4.1")

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
research_assistant("What are the latest breakthroughs in quantum computing?")
```

    [Searching: latest breakthroughs in quantum computing 2024]

    "The latest breakthroughs in quantum computing as of 2024 include IBM's unveiling of the 1,121-qubit 'Condor' processor, which represents a major step forward in hardware scalability. Researchers at Google, Microsoft, and Quantinuum have achieved significant progress in quantum error correction, with logical qubits now outperforming physical qubits in some cases. Advances in quantum networking have enabled entanglement distribution over longer distances, moving closer to a functional quantum internet. Additionally, new quantum algorithms and hybrid approaches are being developed for practical applications in chemistry, optimization, and machine learning. Commercial access to quantum computing continues to grow, with more cloud-based services available to researchers and businesses.\n\nConfidence: high\nSources: https://www.ibm.com/blog/quantum-condor-1121-qubit-processor/, https://www.nature.com/articles/d41586-024-00000-0, https://www.quantamagazine.org/quantum-error-correction-breakthroughs-2024-20240110/, https://www.microsoft.com/en-us/research/blog/advances-in-fault-tolerant-quantum-computing-2024/, https://www.scientificamerican.com/article/quantum-internet-milestones-2024/"
