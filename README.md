# Vardex

**An open-source framework for the AI apps companies ask for: analysts, knowledge assistants,
support desks, document processors, back-office agents and monitors, built from the same blocks.**

Vardex takes its name from the *varde*, the stone cairn that marks a trail across the Norwegian
mountains. It marks a proven path for companies adopting AI, in three layers:

- **Blocks** (the engine): data loading, a warehouse, a document index, extraction, search,
  text-to-SQL, classification, an agent, scheduled workflows, **governed actions** (permission
  levels, an approval inbox, dry-run and undo, an audit log), channels (web app, Teams, email, MCP),
  evals, tracing and guardrails.
- **Recipes**: six ready-made app types built from the blocks, each with its own evals and
  guardrails: *analyst*, *knowledge assistant*, *support desk*, *document processor*,
  *back-office agent* and *monitor*.
- **Packs**: everything specific to one domain (sources, definitions, prompts, glossary, eval
  questions, scheduled briefs), plus the recipes it uses.

```yaml
# packs/norwegian-companies/pack.yaml
name: norwegian-companies
recipes: [analyst, monitor, back-office-agent]
```

Packs can be private. A company keeps its pack in its own repository and points Vardex at it;
nothing about the company enters the engine.

The first pack, **Norwegian companies**, uses only open data: the Brønnøysund registers (NLOD 2.0),
Statistics Norway's Statbank API (CC BY 4.0) and annual reports that listed companies publish
themselves. It both answers and acts: it proposes credit-limit changes in a demo CRM, which a person
approves and can undo.

## Status

Vardex is in early development. Each minor release adds one layer; see [CHANGELOG.md](CHANGELOG.md)
once the first release is out.

| Version | Adds |
|---|---|
| 0.1 | The engine skeleton, the `vardex` command-line tool, pack format v0 |
| 0.2 | Tokens, time and cost for every model call; `vardex tokens` and `vardex cost` |
| 0.3 | Prompts, conversations and structured outputs |
| 0.4 | The warehouse and ingestion for the first pack |
| 0.5 | Text-to-SQL: the analyst recipe, first demo |
| 0.6 | Evals and `vardex eval` |
| 0.7 | Documents and retrieval, with citations |
| 0.8 | The agent, tools, an MCP server and permission levels |
| 0.9 | Scheduled workflows, the approval inbox, dry-run and undo |
| 0.10 | Tracing, cost and quality monitoring |
| 0.11 | Open-weight models and model routing |
| 0.12 | Web app, API, deployment, and `pip install vardex` |
| 0.13 | Guardrails, security tests and the audit log |
| 0.14 | The knowledge-assistant recipe, `vardex new pack`, packs as separate packages |
| 1.0 | A stable pack format and complete documentation |

## Data

Raw data is never committed. The ingest scripts download it.

## License

MIT. Companies can use Vardex with private packs without sharing them.
