.PHONY: default

default:
	WANDB_DISABLE_MMAP=1 WANDB_IGNORE_GLOBS="*.pt" WANDB_DISABLE_CODE=true fedmat-train matcher=greedy num_clients=2 mode=federated
