# Getting Started with Engram

This guide walks you from zero to a working Engram integration in under 5 minutes.

---

## 1. Install

```bash
pip install engdbram
```

For LLM-powered reflection (optional):

```bash
pip install 'engdbram[anthropic]'   # Claude
pip install 'engdbram[openai]'      # OpenAI / Ollama
```

For integrations:

```bash
pip install 'engdbram[mcp]'         # MCP server
pip install 'engdbram[langchain]'   # LangChain
pip install 'engdbram[llamaindex]'  # LlamaIndex
```

---

## 2. First memory store

```python
from engram import Engram

# Single-file store (like SQLite)
mem = Engram(path="./my_agent.engram")

# Record observations — instant, no LLM needed
mem.observe("Alice joined the team as lead engineer")
mem.observe("Ivan moved from Acme to Globex last month", actors=["Ivan"])
mem.observe("The payment service shipped on Friday", tags=["shipping"])

# Retrieve by semantic similarity
for r in mem.recall("Ivan job", k=3):
    print(f"[{r.score:.2f}] {r.episode.content}")

mem.close()
```

---

## 3. Add LLM reflection (fact extraction)

Reflection extracts structured facts from episodes and runs asynchronously — it never blocks your write path.

```python
import os
from engram import Engram, AnthropicAdapter

mem = Engram(
    path="./my_agent.engram",
    llm=AnthropicAdapter(),  # reads ANTHROPIC_API_KEY from env
)

mem.observe("Ivan said he's now the CTO at Globex")
mem.observe("Alice is leading the new infrastructure project")

# Trigger reflection in the background
thread = mem.reflect_async()
thread.join()

run = thread.result
print(f"Extracted {run.facts_extracted} facts from {run.episodes_processed} episodes")
print(f"Resolved {run.contradictions_resolved} contradictions")
```

---

## 4. Time travel

```python
from datetime import datetime, UTC

# What did we know about Ivan in March 2024?
results = mem.recall(
    "Ivan employer",
    k=5,
    as_of=datetime(2024, 3, 1, tzinfo=UTC),
)

# Full history of facts about Ivan
for fact in mem.timeline("Ivan"):
    end = fact.valid_to.date() if fact.valid_to else "present"
    print(f"[{fact.valid_from.date()} → {end}]  Ivan {fact.predicate} {fact.object}")
```

---

## 5. Graph-based spreading activation

Surfaces contextually connected episodes beyond pure semantic similarity:

```python
results = mem.recall(
    "what do I know about Ivan?",
    mode="spreading",
    depth=2,    # graph traversal hops
    decay=0.5,  # activation decay per hop
    k=10,
)
```

---

## 6. MCP server (Claude Desktop / Cursor)

```bash
python -m engram.mcp_server --path ./my_agent.engram
```

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "python",
      "args": ["-m", "engram.mcp_server", "--path", "/absolute/path/to/my_agent.engram"]
    }
  }
}
```

Available tools in Claude Desktop: `observe`, `recall`, `assert_fact`, `timeline`, `why`, `reflect`.

---

## 7. LangChain integration

```python
from engram import Engram
from engram.adapters.langchain import EngramRetriever, EngramChatMessageHistory

mem = Engram(path="./my_agent.engram")

# Use as a retriever in RAG chains
retriever = EngramRetriever(engram=mem, k=5)
docs = retriever.invoke("Ivan Globex project")

# Persistent chat history across sessions
history = EngramChatMessageHistory(engram=mem)
history.add_user_message("What did Ivan say about Globex?")
history.add_ai_message("Ivan mentioned he joined Globex last week.")
```

---

## 8. LlamaIndex integration

```python
from engram.adapters.llamaindex import EngramMemory
from llama_index.core.llms import ChatMessage, MessageRole

memory = EngramMemory.from_defaults(engram_path="./my_agent.engram", k=5)
memory.put(ChatMessage(role=MessageRole.USER, content="Ivan joined Globex"))
msgs = memory.get("Ivan employer")
```

---

## 9. Provenance and trust

```python
# Assert a fact directly (no LLM)
fact_id = mem.assert_fact("Ivan", "works_at", "Globex", confidence=0.95)

# Explain where any fact came from
info = mem.why(fact_id)
print(info)
# {
#   "fact": "Ivan works_at Globex",
#   "extracted_from": ["ep-uuid-1"],
#   "extracted_by": "reflection-run-uuid",
#   "confidence": 0.95,
#   "model": "direct assertion"
# }

# Surface contradictions
for a, b in mem.contradictions():
    print(f"CONFLICT: {a.subject} {a.predicate} '{a.object}' vs '{b.object}'")
```

---

## 10. Configuration

```python
from engram import Engram, DecayConfig

mem = Engram(
    path="./my_agent.engram",
    decay_config=DecayConfig(
        lambda_=0.1,    # Forgetting rate. 0.1 ≈ half-life ~7 days.
        alpha=0.2,      # Reinforcement per access.
        beta=0.1,       # Emotional valence weight.
        threshold=0.1,  # Prune below this importance during reflect().
    ),
)
```

---

## Next steps

- Read the full [API reference in README.md](README.md)
- Explore the [architecture and design decisions in DESIGN.md](DESIGN.md)
- Run benchmarks: `python -m engram.benchmarks all`
- Browse [tests/](tests/) for usage examples of every feature
