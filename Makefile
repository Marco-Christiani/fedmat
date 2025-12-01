.PHONY: default

default:
	export WANDB_DISABLE_MMAP=1 WANDB_IGNORE_GLOBS="*.pt" WANDB_DISABLE_CODE=true WANDB_MODE=disabled fedmat-train matcher=greedy num_clients=2 mode=federated