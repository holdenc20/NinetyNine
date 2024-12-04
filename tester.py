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
    checkpoint = torch.load(f'model_finalish.pth', map_location=torch.device('cpu'))
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


Suits = ["\u2666", "\u2660", "\u2665", "\u2663"]  # Diamond, Spade, Heart, Club
Ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']  # Rank names
class NinetyNineGUI:
    def __init__(self, root, env, model):
        self.root = root
        self.env = env
        self.model = model

        self.done = False
        self.total_reward = 0

        # Initialize UI elements
        self.root.geometry("1200x800")
        self.root.title("NinetyNine Game")

        # Frame for the main content area
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=20, pady=20, expand=True, fill=tk.BOTH)

        # Label for game state info
        self.info_label = tk.Label(self.main_frame, text="Game State", font=("Helvetica", 18, "bold"))
        self.info_label.grid(row=0, column=0, columnspan=3, pady=10)

        self.status_label = tk.Label(self.main_frame, text="Game Starting...", font=("Helvetica", 16))
        self.status_label.grid(row=1, column=0, columnspan=3, pady=10)

        self.bids_label = tk.Label(self.main_frame, text="Bids: ", font=("Helvetica", 14))
        self.bids_label.grid(row=2, column=0, columnspan=3, pady=10)

        self.played_cards_label = tk.Label(self.main_frame, text="Played Cards: ", font=("Helvetica", 14))
        self.played_cards_label.grid(row=3, column=0, columnspan=3, pady=10)

        # Create a frame for displaying players' hands
        self.players_frame = tk.Frame(self.main_frame)
        self.players_frame.grid(row=4, column=0, columnspan=3, pady=20, sticky="ew")

        # Create a frame for displaying the current trick (cards in the middle)
        self.trick_frame = tk.Frame(self.main_frame)
        self.trick_frame.grid(row=5, column=0, columnspan=3, pady=10, sticky="ew")

        # Create a frame for displaying the tricks needed
        self.tricks_needed_frame = tk.Frame(self.main_frame)
        self.tricks_needed_frame.grid(row=6, column=0, columnspan=3, pady=10, sticky="ew")

        # Initialize a label to show the current state of the game
        self.update_gui()

        # Start the auto-play process
        self.play_auto()

    def update_gui(self):
        """Update the game state display with card information (suits and ranks)."""
        # Get current hand and other info from the environment
        hand = self.env.get_hand()
        self.status_label.config(text=f"Player {self.env.current_player + 1}'s Turn")

        # Clear previous display elements
        for widget in self.players_frame.winfo_children():
            widget.destroy()

        for widget in self.trick_frame.winfo_children():
            widget.destroy()

        for widget in self.tricks_needed_frame.winfo_children():
            widget.destroy()

        # Display each player's hand with rank and suit
        for i in range(self.env.num_players):
            player_hand = self.env.player_hands[i]
            hand_str = " ".join([f"{Ranks[card % 13]}{Suits[card // 13]}" for card in np.where(player_hand == 1)[0]])
            player_label = tk.Label(self.players_frame, text=f"Player {i + 1}'s Hand: {hand_str}", font=("Helvetica", 14))
            player_label.pack(pady=5)

        # Display the trump suit
        trump_label = tk.Label(self.players_frame, text=f"Trump Suit: {Suits[self.env.trump_suit]}", font=("Helvetica", 14))
        trump_label.pack(pady=10)

        # Display current bids for all players
        bids_str = " ".join([f"Player {i+1}: {bid}" for i, bid in enumerate(self.env.player_bids)])
        self.bids_label.config(text=f"Bids: {bids_str}")

        # Display the cards that have been played in the current trick (cards in the middle)
        played_cards_str = ""

        trick = self.env.current_trick

        if sum(self.env.current_trick) == -3:
            trick = self.env.last_trick
        for i, card in enumerate(trick):
            if card != -1:
                played_cards_str += f"Player {i+1}: {Ranks[card % 13]}{Suits[card // 13]}  "
            else:
                played_cards_str += f"Player {i+1}: None  "

        # Ensure exactly 3 cards are displayed
        while len(self.env.current_trick) < 3:
            played_cards_str += f"Player {len(self.env.current_trick)+1}: None  "
            self.env.current_trick.append(-1)  # Append dummy value for missing cards
        
        self.played_cards_label.config(text=f"Played Cards: {played_cards_str.strip()}")

        # Display the tricks needed for each player
        tricks_needed_str = " ".join([f"Player {i+1}: {self.env.tricks_needed[i]}" for i in range(self.env.num_players)])
        tricks_needed_label = tk.Label(self.tricks_needed_frame, text=f"Tricks Needed: {tricks_needed_str}", font=("Helvetica", 14))
        tricks_needed_label.pack(pady=5)

        if self.done:
            self.status_label.config(text=f"Game Over! Final Reward: {self.total_reward}")
            self.bids_label.config(text="")
            self.played_cards_label.config(text="")

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
