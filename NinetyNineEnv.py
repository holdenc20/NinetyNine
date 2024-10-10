import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NinetyNineEnv(gym.Env):
    def __init__(self):
        super(NinetyNineEnv, self).__init__()

        self.hand_size = 9
        self.num_players = 3
        self.bidding_cards = 2
        self.num_suites = 4

        #Game state variables
        self.player_hands = None #Observable - your hand 
        self.current_trick = np.zeros(self.num_players) #Observable - the currently shown cards
        self.points = np.zeros(self.num_players) #Observable - the current points of all players
        self.current_player = np.random.randint(self.num_players) #Observable - the current player's turn - maybe we can get this from the current_trick
        self.bidding_phase = 1  #Observable
        self.trump_suit = np.random.randint(4) #Observable
        self.tricks_taken = np.zeros(self.num_players) #Observable
        self.tricks_remaining = self.hand_size #You can get this from player_hands - kinda
        self.current_bids = np.zeros(self.num_players) #Observable only for your 

        self.contracts_met = np.zeros(self.num_players)

        #action space
        self.action_space = spaces.Discrete(self.hand_size)

        #TODO: Update to actual observation space
        self.observation_space = spaces.Dict(
            {
            'hand' : spaces.MultiDiscrete([self.num_suites, self.hand_size]),
            'current_trick' : spaces.MultiDiscrete([self.num_suites, self.hand_size] * self.num_players),
            'points' : spaces.MultiDiscrete([139] * self.num_players),# this seems like a place to reduce
            'current_player' : spaces.Discrete(self.num_players),
            'bidding_phase' : spaces.Discrete(2),
            'trump_suit' : spaces.Discrete(self.num_suites),
            'tricks_taken' : spaces.MultiDiscrete([self.hand_size] * self.num_players),
            'currnet_bid': spaces.MultiDiscrete([self.hand_size - self.bidding_cards]),
            }
        )

        #SIZE OF OBSERVATION SPACE:
        # 4 * 9
        # 4 * 9 * 3
        # 139 * 3
        # 3
        # 2
        # 4
        # 9 * 3
        # 7

        #THINGS TO ABSTRACT:
        #hand -> maybe to hand quality / suite quality - meaning the number of tricks they can take
        #current_trick -> cards player and (strength of cards played - or number of cards that you can win will)
        #points -> todo - possible arrangements of points


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

    def render(self, mode='human'):
        pass

    def calculate_reward(self):
        pass

    def check_winner(self):
        pass

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
