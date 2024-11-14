import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NinetyNineEnv(gym.Env):
    def __init__(self):
        super(NinetyNineEnv, self).__init__()

        self.hand_size = 13
        self.num_players = 3
        self.bidding_cards = 3
        self.num_suites = 4

        self.bidding_phase = 1  #1 for bidding
        
        #Game state variables
        #self.points = np.zeros(self.num_players) FOR LATER

        #Bidding states
        self.player_hands = None #Observable - your hand 
        self.trump_suit = np.random.randint(4)

        #Game play states - includes all the bidding states
        self.current_trick = np.zeros(self.num_players) #maybe 52 - 1,0 for shown cards
        self.current_player = np.random.randint(self.num_players) #Observable - the current player's turn - maybe we can get this from the current_trick
        self.tricks_needed = np.zeros(self.num_players) #

        self.reset_game()


    def reset(self):
        self.reset_game()

        observation = {
            'hand': self.player_hands[0], # change to current player
            'current_trick': self.current_trick,
            'points': self.points,
            'current_player': self.current_player,
            'bidding_phase': self.bidding_phase,
            'trump_suit': self.trump_suit,
            'tricks_taken': self.tricks_taken,
            'current_bid': self.current_bids[0] #change to curernt player
        }

        return observation


    def reset_game(self):
        self.reset_hand()

        self.points = np.zeros(self.num_players)
        self.current_player = np.random.randint(self.num_players)
        self.trump_suit = 0 #Initial trump suit is diamonds

    def reset_hand(self):
        #Randomly select 13 cards from a full deck no replacement
        deck = np.random.choice(52, self.hand_size * self.num_players, replace=False)
        self.player_hands[0 : 3] = np.split(deck, self.num_players + 1) # 3 players + dead cards
        self.current_trick = np.zeros(self.num_players, dtype=int)
        self.current_player = np.random.randint(self.num_players)
        self.bidding_phase = 1
        self.trump_suit = self.contracts_met #Next round trump suit is the number of players who met contract the last round
        self.tricks_taken = np.zeros(self.num_players, dtype=int)
        self.tricks_remaining = 13
        self.current_bids = np.zeros(self.num_players, dtype=int)


    def step(self, action):
        if self.bidding_phase == 1:
            #in bidding phase
            pass
        elif self.bidding_phase == 0:
            #in playing phase
            pass
        else:
            raise ValueError('Invalid phase')

    def valid_actions(self):
        return [card for card in self.player_hands[self.current_player]]


    def score_hand(self):
        self.contracts_met = [0, 0, 0]
        for i in range(self.num_players):
            if self.observation_space['tricks_taken'][i] == self.observation_space['current_bids'][i]:
                #contract met
                self.contract_met[i] = 1

        for i in range(self.num_players):
            if sum(self.contracts_met) == 1:
                self.points[i] += 30 + self.observation_space['tricks_taken'][i]
            elif sum(self.contracts_met) == 2:
                self.points[i] += 20 + self.observation_space['tricks_taken'][i]
            elif sum(self.contracts_met) == 3:
                self.points[i] += 10 + self.observation_space['tricks_taken'][i]
            else:
                self.points[i] += self.observation_space['tricks_taken'][i]

        return self.points

    def score_trick(self, trick):
        pass
