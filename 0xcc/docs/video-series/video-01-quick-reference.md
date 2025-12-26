# Video 1 Quick Reference Card

*Print this or keep it on a second monitor while recording*

---

## Scene Flow (6:30 total)

| Time | Scene | Action | Key Points |
|------|-------|--------|------------|
| 0:00 | Hook | Show full interface | "Train SAEs locally, no cloud" |
| 0:30 | What is miStudio | Pan across tabs | "Complete interpretability workbench" |
| 1:30 | Tour | Click each tab | Datasets → Models → SAEs → Training → Features → Steering |
| 2:30 | System Monitor | Open GPU stats | Show utilization, memory, temp |
| 3:30 | Workflow | Hover tabs as mentioned | 6 steps: Data → Train → Discover → Label → Steer → Share |
| 5:00 | Requirements | Brief mention | GPU, miStudio, HuggingFace, optional OpenAI |
| 6:00 | Close | Return to Datasets | "Next: download model & dataset" |

---

## Panel Click Order

```
1. Datasets   (2 sec pause)
2. Models     (2 sec pause)
3. SAEs       (2 sec pause)
4. Training   (2 sec pause)
5. Features   (2 sec pause)
6. Steering   (2 sec pause)
```

---

## Terms to Define

- **SAE** = learns interpretable features from activations
- **Tokenization** = text → numbers for model
- **Feature** = interpretable direction in activation space
- **Steering** = amplify/suppress features to change behavior

---

## 6-Step Workflow (Scene 5)

1. **Data Prep** - Download & tokenize
2. **Train SAE** - Configure & monitor
3. **Discovery** - Extract features
4. **Labeling** - Auto-label with LLM
5. **Steering** - Test understanding
6. **Sharing** - Export to HuggingFace

---
Suggested Video Series Structure
## Don't Forget

- [ ] Move mouse SLOWLY
- [ ] Pause after each click
- [ ] Say what you'll do BEFORE doing it
- [ ] Check GPU stats are visible
- [ ] End on Datasets panel (setup for Video 2)
