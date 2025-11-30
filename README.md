
```sh
# Disable wandb with WANDB_MODE=disabled while developing
WANDB_MODE=disabled fedmat-train epochs=1 --cfg=job # remove --cfg when ready to launch
```

Choose a matcher (none by default):

```sh
fedmat-train matcher=greedy
```

Use a custom config by copying `experiment.yaml`, modifying as needed, and running:

```sh
fedmat-train --config-name=myconfig --cfg=job # remove --cfg when ready to launch
```

