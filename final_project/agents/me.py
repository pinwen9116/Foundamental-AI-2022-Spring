from game.players import BasePokerPlayer
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import random as rand
import copy 

total_state = []
total_loss = []
state = []
next_cards = 0                 # change to next street, update community card
do_it = 0
left_round = 21
put_in = 0
class MyPlayer(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info] 
        # Use RL for poker game
        # 0 fold, 1 call, 2 raise
        # Not smart intellegence aka ME!
        print("this is my card:", hole_card)
        if do_it == 0:
            print("do it myself")
            card1 = hole_card[0][1]
            card2 = hole_card[1][1]
            if card1.isnumeric() and card2.isnumeric():
                if abs(int(card1) - int(card2)) >= 1:
                    act, action, amount = 0, "fold", 0
            elif card1.isnumeric() or card2.isnumeric():
                if card1.isnumeric() and int(card1) < 6:
                    act, action, amount = 0, "fold", 0
                elif card2.isnumeric() and int(card2) < 6:
                    act, action, amount = 0, "fold", 0
                else:
                    call_action_info = valid_actions[1]
                    act, action, amount = 1, call_action_info["action"], call_action_info["amount"]
                    if amount > 50:
                        act, action = 0, "fold"
            return action, amount
       
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass  

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def setup_ai():
    return MyPlayer()
