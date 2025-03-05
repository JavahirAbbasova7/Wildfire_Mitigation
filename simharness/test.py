import pickle

ckpt_path = "/nfs/lslab2/fireline/simharness/experiments/2025-03-03_18-41-42/DQN_2025-03-03_18-41-44/DQN_simharness2.environments.ReactiveHarness_27e39_00000_0_gamma=0.8609,lr=0.0033,train_batch_size=128_2025-03-03_18-41-44/checkpoint_000000/algorithm_state.pkl"

with open(ckpt_path, "rb") as f:
    state = pickle.load(f)

# print(state.keys())  # See what keys exist
# print(state.get("config", {}).keys())  # Check what's inside config
print(state["config"])  # Check what's inside config