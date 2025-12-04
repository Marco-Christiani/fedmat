

```sh
uv sync
uv run pytest
```


```sh
# Disable wandb with `wandb disabled` while developing
fedmat-train epochs=1 --cfg=job # remove --cfg when ready to launch
```

Choose a matcher (none by default):

```sh
fedmat-train matcher=greedy
```

Use a custom config by copying `experiment.yaml`, modifying as needed, and running:

```sh
fedmat-train --config-name=myconfig --cfg=job # remove --cfg when ready to launch
```

