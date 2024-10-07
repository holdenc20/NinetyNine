import gymnasium as gym
from gymnasium import spaces
import numpy as np

class NinetyNineEnv(gym.Env):
    def __init__(self):
        super(NinetyNineEnv, self).__init__()

        self.hand_size = 13
        self.num_players = 3

        #Game state variables
        self.player_hands = None
        self.current_trick = np.zeros(self.num_players)
        self.points = np.zeros(self.num_players)
        self.current_player = np.random.randint(self.num_players)
        self.bidding_phase = 1
        self.trump_suit = np.random.randint(4)
        self.tricks_taken = np.zeros(self.num_players)
        self.tricks_remaining = 13
        self.current_bids = np.zeros(self.num_players)

        self.contracts_met = np.zeros(self.num_players)

        #action space
        self.action_space = spaces.Discrete(self.hand_size)

        #TODO: Update to actual observation space
        self.observation_space = spaces.Dict(
            {
            'hand' : spaces.MultiDiscrete([4, 13] * self.hand_size),
            'current_trick' : spaces.MultiDiscrete([4, 13] * self.num_players),
            'points' : spaces.MultiDiscrete([100] * self.num_players),
            'current_player' : spaces.Discrete(self.num_players),
            'bidding_phase' : spaces.Discrete(2),
            'trump_suit' : spaces.Discrete(4),
            'tricks_taken' : spaces.MultiDiscrete([13] * self.num_players),
            'tricks_remaining' : spaces.Discrete(13),
            'currnet_bids': spaces.MultiDiscrete([10] * self.num_players),
            }
        )


        self.reset_game()


    def reset(self):
        self.reset_game()

        observation = {
            'hand': self.player_hands[0],
            'current_trick': self.current_trick,
            'points': self.points,
            'current_player': self.current_player,
            'bidding_phase': self.bidding_phase,
            'trump_suit': self.trump_suit,
            'tricks_taken': self.tricks_taken,
            'tricks_remaining': self.tricks_remaining,
            'current_bids': self.current_bids
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
