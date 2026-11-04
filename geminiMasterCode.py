import pickle
import time
import numpy as np
import redis

# 1. Connect to Redis (running locally on the Master Pi)
r = redis.Redis(host="localhost", port=6379, db=0)

# Q-Learning Hyperparameters
ALPHA = 0.1  # Learning rate (how much to trust new info vs old info)
GAMMA = 0.99  # Discount factor (importance of future rewards)

# Initialize CartPole Q-table shape: 4 state variables binned into 10 buckets each, 2 actions (Left/Right)
STATE_BINS = 10
Q_TABLE_SHAPE = (STATE_BINS, STATE_BINS, STATE_BINS, STATE_BINS, 2)

# Check if a Q-table already exists in Redis, otherwise create a new one of zeros
if not r.exists("q_table"):
  print("Initializing new Q-table in Redis...")
  initial_q = np.zeros(Q_TABLE_SHAPE)
  r.set("q_table", pickle.dumps(initial_q))
else:
  print("Found existing Q-table in Redis.")


def discretize_state(state):
  """Helper to convert continuous CartPole numbers into discrete grid bins."""
  # CartPole bounds: position (-2.4 to 2.4), velocity (-3 to 3), angle (-0.2 to 0.2), ang_velocity (-3 to 3)
  cart_pos, cart_vel, pole_ang, pole_vel = state

  # Binning logic
  bins_pos = np.digitize(
      cart_pos, np.linspace(-2.4, 2.4, STATE_BINS - 1)
  ) - 1
  bins_vel = np.digitize(
      cart_vel, np.linspace(-3.0, 3.0, STATE_BINS - 1)
  ) - 1
  bins_ang = np.digitize(
      pole_ang, np.linspace(-0.2, 0.2, STATE_BINS - 1)
  ) - 1
  bins_pvel = np.digitize(
      pole_vel, np.linspace(-3.0, 3.0, STATE_BINS - 1)
  ) - 1

  # Ensure indices stay within bounds [0, STATE_BINS-1]
  return (
      max(0, min(STATE_BINS - 1, bins_pos)),
      max(0, min(STATE_BINS - 1, bins_vel)),
      max(0, min(STATE_BINS - 1, bins_ang)),
      max(0, min(STATE_BINS - 1, bins_pvel)),
  )


print("Master learning loop started. Waiting for experiences from slaves...")

# Main Master Learning Loop
while True:
  # Block and wait for an experience tuple pushed by any slave into the Redis queue 'experience_queue'
  packed_data = r.blpop("experience_queue", timeout=1)

  if packed_data:
    # Unpack the Redis queue tuple (queue_name, serialized_data)
    _, data_bytes = packed_data
    experience = pickle.loads(data_bytes)
    state, action, reward, next_state, done = experience

    # Load current Q-table from Redis
    q_table = pickle.loads(r.get("q_table"))

    # Convert continuous states to discrete grid indexes
    s = discretize_state(state)
    ns = discretize_state(next_state)

    # Apply the Q-learning update formula (Bellman Equation)
    current_q = q_table[s + (action,)]

    if done:
      max_future_q = 0
    else:
      max_future_q = np.max(q_table[ns])

    # New Q-value calculation
    new_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
    q_table[s + (action,)] = new_q

    # Save updated Q-table back to Redis so slaves can fetch it
    r.set("q_table", pickle.dumps(q_table))