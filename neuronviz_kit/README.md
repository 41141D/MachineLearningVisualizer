# neuronviz

Live view of a small PyTorch model while it trains: decision map, network with weight colors, loss and accuracy, and a scrubber to replay training.

## Install
Copy the `neuronviz/` folder next to your script (or into your project root). Needs `torch`; nothing else.

## Use
```python
import neuronviz as nv
model = nn.Sequential(...)
nv.attach(model, X, y)      # after building the model and data
# your training loop, unchanged
```
Run it in PyCharm. The console prints `http://127.0.0.1:8765`. The first run opens a browser tab; leave that tab open beside PyCharm. Every later run resets it and streams live.

## Options
`nv.attach(model, X, y, every=10, slow=0.05, loss_fn=None)`
- `every`: snapshot every N optimizer steps.
- `slow`: seconds to pause per snapshot so fast runs are watchable. `0` for full speed.
- `loss_fn`: only if your loss is not BCE, cross-entropy or MSE.

## Limits
Draws models made of `nn.Linear` layers. The decision map needs 2 input features; with more, you still get the network, loss and accuracy. Layers wider than 16 neurons show the first 16.
