import re
from pathlib import Path
js = Path("frontend/dist/assets/app.547295de.js").read_text(encoding="utf-8")
idx = js.find('"SD-Trainer"')
start = js.rfind("[", 0, idx)
# find matching ]
depth = 0
for i in range(start, len(js)):
    if js[i] == "[":
        depth += 1
    elif js[i] == "]":
        depth -= 1
        if depth == 0:
            print(js[start : i + 1])
            break
