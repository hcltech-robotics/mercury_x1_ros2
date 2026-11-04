# Knowledge files for voice chat

Every `*.md` here (except this README) and in `~/.config/mercury_x1/knowledge/` on the robot is searched for each
spoken question (`src/visualization/mercury_x1_web/mercury_x1_web/knowledge.py`, BM25). The best paragraphs go to the
language model with the question. Files are re-read automatically when they change, so no restart is needed.

| File | Content |
|---|---|
| `hcltech_2026.md` | HCLTech facts as of November 2026 (Q2 FY27 results, leadership, news), with sources |
| `hcltech_q3fy25.md` | older Q&A (Q3 FY25) imported from proj-cec-doggo-ai; kept for clients, locations, history |
| `mercury_robot.md` | what the robot is and can do |
| `lab.md` (to add) | the lab itself: rooms, demos, people to ask, opening hours, visitor rules |

Writing facts:
- **Title line:** the first line is `# <title, with the "as of" date>`. The model sees it as the source.
- **Sections:** use `## <section>` headings. A blank line separates paragraphs, and a paragraph is what search returns.
- **One fact per sentence, written to be spoken.** Give every number its period and unit, for example
  "revenue was 3.8 billion dollars in Q2 FY27".
- **Local facts:** put facts that should not go to git (names, phone numbers) in `~/.config/mercury_x1/knowledge/`
  on the robot.
