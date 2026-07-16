# D-DIN source and figure-use contract

These files are pinned from `amcchord/datto-d-din` commit
`e199c8441e758d6e492cd01ef52c3c67ba4bae26`, which mirrors the open D-DIN
release commissioned by Datto from Monotype. The font software is distributed
under the included SIL Open Font License 1.1; the repository also includes the
upstream copying, attribution, font log, README, and CC BY-SA text with only
whitespace normalization for repository linting.

Included figure faces:

- `D-DIN.ttf`, `D-DIN-Italic.ttf`, and `D-DIN-Bold.ttf` for ordinary figures;
- `D-DINCondensed.ttf` and `D-DINCondensed-Bold.ttf` for exceptional short labels;
- `D-DINExp.ttf`, `D-DINExp-Italic.ttf`, and `D-DINExp-Bold.ttf` for sparse display accents.

Pinned SHA-256 values:

| File | SHA-256 |
|---|---|
| `D-DIN.ttf` | `705bece88e33c8f86d0ace0c7d93ee143b745cba7a99643753a4f91c3c22e204` |
| `D-DIN-Italic.ttf` | `22100d8442e310a1add840c3cbb717b64b5b59f176c0061e440b753c09ed3a26` |
| `D-DIN-Bold.ttf` | `69cc46d24509802693a2a5f6e1b18bcad65f8baf5fd3c08b9bd364beb2a8bdd5` |
| `D-DINCondensed.ttf` | `724b48d534dbcde7a9f039bddc3a7344d4913de43726f3b7d7a56f0770a8ea6b` |
| `D-DINCondensed-Bold.ttf` | `664e694799db84a910f08edc717916763a2e3f23ee44b4530769968768b34293` |
| `D-DINExp.ttf` | `ebb595323d0af86931cccc35ba232bee564ab034f62a8b41ef7c1617dd5111c1` |
| `D-DINExp-Italic.ttf` | `bbab81e71d8707d0ebbe9f19f7be4d2ef7e04b0d983df36effb8d2cd3e2e887e` |
| `D-DINExp-Bold.ttf` | `ede3a43f2ed4c5658a607fe179f3ddc497dfe6645f29bfbb8b5fbc5b8831e0ad` |

Do not modify or rename the font files. `src/dpsae/plot_style.py` registers them
from these exact paths before loading the Matplotlib style.
