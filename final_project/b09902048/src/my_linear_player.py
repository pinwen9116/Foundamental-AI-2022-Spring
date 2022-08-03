from sklearn.linear_model import LinearRegression as LR
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

network = LR()
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
        global state, next_cards, do_it, left_round, total_loss, amount, put_in
        print("this is loss amount:", sum(total_loss), total_loss)
        if sum(total_loss) < left_round * 20 * (-1):
            print("expect", sum(total_loss), left_round * 20 * (-1))
            return "fold", 0
        if next_cards == 0:
            next_cards = 1
            round_card = round_state["community_card"]
            the_card = []
            for card in round_card:
                if card[1].isnumeric():
                    cardi = int(card[1]) - 2
                elif card[1] == "T":
                    cardi = 8
                elif card[1] == "J":
                    cardi = 9
                elif card[1] == "Q":
                    cardi = 10
                elif card[1] == "K":
                    cardi = 11
                elif card[1] == "A":
                    cardi = 12

                if card[0] == "H":
                    cardj = 0
                elif card[0] == "S":
                    cardj = 1
                elif card[0] == "C":
                    cardj = 2
                elif card[0] == "D":
                    cardj = 3

                the_card.append(cardi * 4 + cardj)
            
            state.extend(the_card)
            state.append(round_state["pot"]["main"]["amount"])
        else:
            state.pop()
            state.pop()
            state.pop()
            state.append(round_state["pot"]["main"]["amount"])
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
            else:
                
                action_info= valid_actions[2]
                act, action, amount = 2, action_info["action"], action_info["amount"]["max"]
                print("DO IT AMOUNT", amount)
                divide = 1
                if card1 == "K":
                    divide += 0.5
                elif card1 == "Q":
                    divide += 1
                elif card1 == "T":
                    divide += 2

                if card2 == "K":
                    divide += 0.5
                elif card2 == "Q":
                    divide += 1
                elif card2 == "T":
                    divide += 2
                print("DO IT DIVIDE", divide)
                amount /= divide
            state.append(act)
            state.append(amount)
            put_in += amount
            print("afdsafdsafsafdsafdsaf: ", amount, put_in)
            return action, amount
        do_it = 1
        best_act = 0
        amount = 0
        best_loss = np.inf
        for action_info in valid_actions:
            if action_info["action"] == "fold":
                cstate = copy.deepcopy(state)
                cstate.append(0)
                cstate.append(action_info["amount"])
                while(len(cstate) < 26):
                    cstate.append(-1)
                loss = network.predict(np.array(cstate).reshape(1, -1))
                print("fold loss:", loss)
                if loss <= best_loss:
                    best_act = 0
                    best_loss = loss
                    amount = action_info["amount"]
            elif action_info["action"] == "call":
                cstate = copy.deepcopy(state)
                cstate.append(1)
                cstate.append(action_info["amount"])
                while(len(cstate) < 26):
                    cstate.append(-1)
                loss = network.predict(np.array(cstate).reshape(1, -1))
                print("call loss:", loss)
                if loss <= best_loss:
                    best_act = 1
                    best_loss = loss
                    amount = action_info["amount"]
            elif action_info["action"] == "raise":
                cstate = copy.deepcopy(state)
                cstate.append(2)
                t_amount = action_info["amount"]
                t_amount = rand.randrange(t_amount["min"], t_amount["max"])
                cstate.append(t_amount)
                while(len(cstate) < 26):
                    cstate.append(-1)
                loss = network.predict(np.array(cstate).reshape(1, -1))
                print("raise loss:", loss)
                print("RAISE")
                if loss <= best_loss:
                    best_act = 2
                    best_loss = loss
                    amount = t_amount

        if best_act == 0:
            action = "fold"
            state.append(0)
        elif best_act == 1:
            action = "call"
            state.append(1)
        elif best_act == 2:
            action = "raise"
            state.append(2)
        state.append(amount)
        put_in += amount
        return action, amount  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        global do_it, left_round, total_loss, total_state, put_in
        put_in = 5
        left_round -= 1
        if round_count % 2 == 1:
            do_it = 1
        else:
            do_it =0 
        if round_count == 1:
            left_round = 20
            total_state = []
            total_loss = []

        global state
        state = []
        my_card = []
        for card in hole_card:
            if card[1].isnumeric():
                cardi = int(card[1]) - 2
            elif card[1] == "T":
                cardi = 8
            elif card[1] == "J":
                cardi = 9
            elif card[1] == "Q":
                cardi = 10
            elif card[1] == "K":
                cardi = 11
            elif card[1] == "A":
                cardi = 12

            if card[0] == "H":
                cardj = 0
            elif card[0] == "S":
                cardj = 1
            elif card[0] == "C":
                cardj = 2
            elif card[0] == "D":
                cardj = 3

            my_card.append(cardi * 4 + cardj)
        state.extend(my_card)
        #original_state.extend(hole_card)
        pass

    def receive_street_start_message(self, street, round_state):
        global next_cards
        next_cards = 0
        pass  

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        
        global total_state, total_loss, state, do_it, left_round, put_in
        # train
        winner = winners[0]["name"]
        if "me" not in winner:
            loss = 0
        else:
            loss = -1
        print("pot and put_in", round_state["pot"]["main"]["amount"], put_in)
        amount = round_state["pot"]["main"]["amount"] * loss
        amount += put_in
        print("hihihihihi loss:", amount)
        while len(state) != 26:
            state.append(-1)
        total_state.append(state)
        total_loss.append(amount)
        if not sum(total_loss) >=  left_round * (-20):
            network.fit(total_state, total_loss)
        # this is for Reinforce learnings
        #agent.forward(state)

        # save
        #pickle.dump(network, open("./agents/network_LR.pkl", "wb"))

        pass

def setup_ai():
    # load trained data
    global network
    network = pickle.load(open("./agents/network_LR.pkl", "rb"))

    return MyPlayer()
