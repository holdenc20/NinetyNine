import numpy as np
import gymnasium as gym
import random

class NinetyNineEnv(gym.Env):
    def __init__(self):
        super(NinetyNineEnv, self).__init__()

        self.hand_size = 13
        self.num_players = 3
        self.bidding_cards = 3
        self.num_suites = 4

        self.bidding_phase = 1  # 1 for bidding

        self.points = [0, 0, 0]

        # Game state variables
        self.player_hands = [np.zeros(52, dtype=int) for _ in range(self.num_players)] # 0 for bid cards -1 for other cards 1 for in hand player cards
        self.burned_cards = np.zeros(52, dtype=int)

        self.trump_suit = np.random.randint(4)

        # Gameplay state
        self.current_trick = -1 * np.ones(self.num_players, dtype=int)
        self.current_player = np.random.randint(self.num_players)  # 0, 1, 2


        self.player_bids = np.zeros(self.num_players, dtype=int) #bid
        self.tricks_needed = np.zeros(self.num_players, dtype=int)  # bid - tricks taken

        self.reset_game()

    def reset_game(self):
        """
        Resets the game state, including card distribution and bidding phase.
        """
        # Randomly select `hand_size * num_players` cards from the deck
        deck = np.random.choice(52, self.hand_size * self.num_players, replace=False)

        # Initialize all player hands as arrays of -1 (no cards in hand)
        self.player_hands = [np.full(52, -1, dtype=int) for _ in range(self.num_players)]

        # Assign cards to players
        for i in range(self.num_players):
            self.player_hands[i][deck[i * self.hand_size : (i + 1) * self.hand_size]] = 1  # Cards in hand

        # Remaining cards (not assigned to players) are set to 0 for bidding
        remaining_cards = np.setdiff1d(np.arange(52), deck)
        for player_hand in self.player_hands:
            player_hand[remaining_cards] = -1  # Mark as bidding cards

        # Initialize other game state variables
        self.trump_suit = np.random.randint(self.num_suites)
        self.bidding_phase = 1

        self.player_bids = np.zeros(self.num_players, dtype=int)
        self.tricks_needed = np.zeros(self.num_players, dtype=int)
        self.current_player = np.random.randint(self.num_players)  # Randomize starting player

        self.points = [0, 0, 0]

        #REMOVE LATER
        #get to playing phase randomly
        for r in range(3):
            for p in range(3):
                actions = self.possible_actions()
                action = random.choice(actions)
                self.step(action)
        #print("Playing phase started")

        return self.get_state()

    def get_hand(self):
        """
        Returns the observation for the current player.
        """
        return self.player_hands[self.current_player]

    def get_bidding_observation(self):
        """
        Returns the observation for the current player during the bidding phase.
        """
        hand = self.get_hand()
        trump = [self.trump_suit]
        current_bid_count = [self.tricks_needed[self.current_player]]

        observation = np.concatenate((hand, trump, current_bid_count))
        return observation

    def get_playing_observation(self):
        """
        Returns the observation for the current player during the playing phase.
        """
        hand = self.get_hand()
        current_trick = self.current_trick
        trump = [self.trump_suit]
        current_bid_count = [self.tricks_needed[self.current_player]]

        observation = np.concatenate((hand, current_trick, trump, current_bid_count))
        return observation

    def get_state(self):
        if self.bidding_phase == 1:
            return self.get_bidding_observation()
        elif self.bidding_phase == 0:
            return self.get_playing_observation()
        else:
            raise ValueError("Invalid phase.")


    def step(self, card: int):#card aka action
        """
        Allows the current player to play a card.
        """
        #print(f"Player {self.current_player} plays {card}.")
        if self.player_hands[self.current_player][card] != 1:
            return self.get_state(), 0, False, {}
        if card > 51 or card < 0:
            return self.get_state(), 0, False, {}

        #TAKING ACTION
        trickover = False
        if self.bidding_phase == 1:
            #need to add to the current bid
            self.player_bids[self.current_player] += card // 13
            self.tricks_needed[self.current_player] += card // 13

            #print(f"Player {self.current_player} bids {card // 13} tricks.")
            self.player_hands[self.current_player][card] = 0
        elif self.bidding_phase == 0:
            #need to play a card
            self.current_trick[self.current_player] = card
            for i in range(self.num_players):
                self.player_hands[i][card] = 0
            if self.current_trick[0] != -1 and self.current_trick[1] != -1 and self.current_trick[2] != -1:
                #trick is over
                trickover = True
                winner = self.best_card(self.current_trick, self.trump_suit, (self.current_player + 1) % self.num_players)
                self.tricks_needed[winner] -= 1
                self.current_trick = -1 * np.ones(self.num_players, dtype=int)

        #CHECKING IF BIDDING ROUND IS OVER
        if self.bidding_phase == 1 and all(np.sum(hand == 0) == self.bidding_cards for hand in self.player_hands):
            self.bidding_phase = 0

        #REWARD / GAME OVER
        reward = 0
        done = False
        if self.bidding_phase == 0 and len(self.possible_actions(0)) == 0 and len(self.possible_actions(1)) == 0 and len(self.possible_actions(2)) == 0:
            scores = self.score_hand()
            reward = scores[0] - scores[1] - scores[2]
            done = True

        #UPDATING GAME STATE
        if trickover:
            self.current_player = winner
        else:
            self.current_player = (self.current_player + 1) % self.num_players
        next_state = self.get_state()

        #next_state, reward, done, info
        return next_state, reward, done, {}

    def best_card(self, trick, trump, first_player):
        lead_card = trick[first_player]
        lead_suit = lead_card // 13

        best_card = -1
        best_player = -1

        for player, card in enumerate(trick):

            suit = card // 13
            rank = card % 13

            if best_card == -1:
                # First valid card, automatically the best
                best_card = card
                best_player = player
            else:
                best_suit = best_card // 13
                best_rank = best_card % 13

                if suit == trump and best_suit != trump:
                    # Trump beats non-trump
                    best_card = card
                    best_player = player
                elif suit == best_suit and rank > best_rank:
                    # Higher rank within the same suit
                    best_card = card
                    best_player = player
                elif suit == lead_suit and best_suit != trump and best_suit != lead_suit:
                    # Lead suit beats other non-trump suits
                    best_card = card
                    best_player = player

        return best_player

    #index of all possible actions
    def possible_actions(self, player = None):
        if player is not None:
            return np.where(self.player_hands[player] == 1)[0]
        return np.where(self.player_hands[self.current_player] == 1)[0]

    #TODO
    def score_hand(self):
        self.contracts_met = [0, 0, 0]
        for i in range(self.num_players):
            if self.tricks_needed[i] == 0:
                #contract met
                self.contracts_met[i] = 1

        for i in range(self.num_players):
            if sum(self.contracts_met) == 1:
                self.points[i] += 30 + self.player_bids[i]
            elif sum(self.contracts_met) == 2:
                self.points[i] += 20 + self.player_bids[i]
            elif sum(self.contracts_met) == 3:
                self.points[i] += 10 + self.player_bids[i]
            else:
                self.points[i] += self.player_bids[i]

        #print("Points", self.points)

        return self.points

'''
env = NinetyNineEnv()

for i in range(30):
    print("------------")
    print(env.get_state())
    print(env.possible_actions())
    print(env.current_trick)
    x = input("Card to play: ")
    print(env.step(int(x)))
'''

#THINGS TO ADD LATER
# NEED TO FOLLOW SUIT 
# POINTS SYSTEM