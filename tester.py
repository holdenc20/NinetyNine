import torch
from DQN import DQN
from NinetyNineEnv import NinetyNineEnv
import tkinter as tk
import numpy as np
import time
from PIL import Image, ImageTk

# Initialize environment
env = NinetyNineEnv()
state = env.reset_game()
state_size = len(state)

# Load the checkpoint
try:
    checkpoint = torch.load(f'model_{3}.pth', map_location=torch.device('cpu'))
except FileNotFoundError:
    print("Checkpoint file not found. Please check the file path.")
    exit(1)

# Initialize models
dqn_models_loaded = {}
for key, state in checkpoint.items():
    dqn = DQN(state_size, 52, num_layers=3, hidden_dim=128)  # Match architecture
    if "state_dict" in state:
        dqn.load_state_dict(state["state_dict"])  # Extract the correct part
    else:
        print(f"Invalid checkpoint format for key {key}")
        continue
    dqn_models_loaded[key] = dqn

# Confirm loaded models
print("Loaded models:", dqn_models_loaded.keys())


Suits = ["\u2663", "\u2665", "\u2666", "\u2660"]  # Clubs, Hearts, Diamonds, Spades
Ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']  # Rank names
class NinetyNineGUI:
    def __init__(self, root, env, model):
        self.root = root
        self.env = env
        self.model = model

        self.done = False
        self.total_reward = 0

        self.root.geometry("1200x800")
        self.root.title("NinetyNine Game")

        self.update_gui()

        self.play_auto()

    def update_gui(self):
        pass

    def play_auto(self):
        """Automatically play the game using the model."""
        if self.done:
            self.status_label.config(text=f"Game Over! Final Reward: {self.total_reward}")
            return

        state = self.env.get_state()

        q_values = self.model(torch.tensor(state, dtype=torch.float32))

        possible_actions = self.env.possible_actions()
        if len(possible_actions) == 0:
            self.done = True
            self.update_gui()
            return

        num_actions = q_values.size(0)
        mask = torch.full((num_actions,), float('-inf'))
        mask[possible_actions] = 0
        masked_q_values = q_values + mask
        action = torch.argmax(masked_q_values).item()

        next_state, reward, self.done, _ = self.env.step(action)
        self.total_reward += reward

        self.update_gui()

        if not self.done:
            self.root.after(10000, self.play_auto)


root = tk.Tk()
root.title("NinetyNine Game")

env = NinetyNineEnv()


game_gui = NinetyNineGUI(root, env, dqn_models_loaded["100_0"])

root.mainloop()
