# Figures Directory

Place the following figure files here. The `main.tex` file references them by these exact names.

| Filename | What it should show | Source |
|---|---|---|
| `tu_logo.png` | Tribhuvan University Star logo (same one used in the mid-term proposal) | Re-use from mid-term proposal repo / IOE official site |
| `lte_arch.png` | LTE architecture (UE -> eNB -> MME / SGW / PGW -> Internet) | Re-use from mid-term proposal |
| `a3_event.png` | Event-A3 trigger curve + handover signalling sequence | Re-use from mid-term proposal |
| `gnn_dqn_arch.png` | GNN-DQN architecture overview | Re-use from mid-term proposal (figure 2.4 / 3.4) |
| `system_block.png` | Overall system block diagram with the SON layer added | Extend mid-term figure 3.1 with the SON translation block on the right |

## Tips

- Keep figures at 300 DPI for print quality. PNG is fine; vector PDF is better if you can export it.
- If a figure is missing at compile time, the build still succeeds but a placeholder box appears in the PDF.

## Compile

```bash
cd final_report
pdflatex main.tex
pdflatex main.tex      # second pass to settle ToC, LoF, LoT
pdflatex main.tex      # third pass to settle page numbers

# Standalone engagement brief for the NTC meeting
pdflatex ntc_engagement_brief.tex
```
