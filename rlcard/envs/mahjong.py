import numpy as np
from collections import OrderedDict

from rlcard.envs import Env
from rlcard.games.mahjong import Game
from rlcard.games.mahjong import Card
from rlcard.games.mahjong.utils import card_encoding_dict, encode_cards, pile2list

class MahjongEnv(Env):
    ''' Mahjong Environment
    '''

    def __init__(self, config):
        self.name = 'mahjong'
        self.game = Game()
        super().__init__(config)
        self.action_id = card_encoding_dict
        self.de_action_id = {self.action_id[key]: key for key in self.action_id.keys()}
        self.state_shape = [[6, 34, 4] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]
        self.reward_config = {
            'base_win': 1.0,
            'home_win': 2.0,
            'self_draw': 1.5,
        }       

    def _extract_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state

        Returns:
            numpy array: 6*5*15 array
                         6 : current hand
                             the union of the other two players' hand
                             the recent three actions
                             the union of all played cards
        '''
        players_pile = state['players_pile']
        hand_rep = encode_cards(state['current_hand'])
        piles_rep = []
        for p in players_pile.keys():
            piles_rep.append(encode_cards(pile2list(players_pile[p])))
        piles_rep = np.array(piles_rep)
        table_rep = encode_cards(state['table'])
        rep = [hand_rep, table_rep]
        rep.extend(piles_rep)     
        obs = np.array(rep)

        extracted_state = {
            'obs': obs,
            'legal_actions': self._get_legal_actions(),
            'raw_obs': state,
            'raw_legal_actions': [a for a in state['action_cards']],
            'action_record': self.action_recorder
        }

        return extracted_state

    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''

        _, player, win_info = self.game.judger.judge_game(self.game)
            
        payoffs = [0.0] * self.num_players
        
        if player == -1:  # Draw
            return np.array(payoffs)
            
        base_reward = self.reward_config['base_win']

        if win_info['method'] == 'self_draw':
            base_reward += self.reward_config['self_draw']

        if player == self.game.home_player:
            base_reward += self.reward_config['home_win']
            
        payoffs[player] = base_reward

        # If won from discard, penalize the player who discarded
        if win_info['method'] == 'discard':
            discard_player = win_info['from_player']
            if discard_player != None:
                payoffs[discard_player] -= 0.5
                
        else:
            for i in range(self.num_players):
                if i!=player:
                    payoffs[i] -= 0.5

        return np.array(payoffs)

    def _decode_action(self, action_id):
        ''' Action id -> the action in the game. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action (string): the action that will be passed to the game engine.
        '''
        action = self.de_action_id[action_id]
        if action_id < 34:
            candidates = self.game.get_legal_actions(self.game.get_state(self.game.round.current_player))
            for card in candidates:
                if card.get_str() == action:
                    action = card
                    break
        return action

    def _get_legal_actions(self):
        ''' Get all legal actions for current state

        Returns:
        if type(legal_actions[0]) == Card:
            print("GET:", [c.get_str() for c in legal_actions])
        else:
            print(legal_actions)
            legal_actions (list): a list of legal actions' id
        '''
        legal_action_id = {}
        legal_actions = self.game.get_legal_actions(self.game.get_state(self.game.round.current_player))
        if legal_actions:
            for action in legal_actions:
                if isinstance(action, Card):
                    action = action.get_str()
                action_id = self.action_id[action]
                legal_action_id[action_id] = None
        else:
            print("##########################")
            print("No Legal Actions")
            print(self.game.judger.judge_game(self.game))
            print(self.game.is_over())
            print([len(p.pile) for p in self.game.players])
            #print(self.game.get_state(self.game.round.current_player))
            #exit()
        return OrderedDict(legal_action_id)
