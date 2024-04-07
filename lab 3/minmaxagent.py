from exceptions import GameplayException
from connect4 import Connect4
from exceptions import AgentException

from copy import deepcopy
from typing import Tuple
import random

class MinMaxAgent:
    def __init__(self, my_token):
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        _, move = self.minmax(connect4, 1, 5)   # looking for move which wins ("1") and predict exactly "5" moves
        print(move)
        return move
    
    # resultOfMove: 1 - I win, -1 - opponent wins, 0 - tie
    # d - max recursion depth, how many moves to predict, should be > 0
    # returns [best score, best possible move: number from 0...maxSlot, from connect4]
    def minmax(self, connect4: 'Connect4', resultOfMove: int, d: int) -> Tuple[int, int]:
        if connect4.game_over and connect4.wins == self.my_token:
            return 1, -1    # slot -1 is invalid, because game stops and some value must be returned

        if connect4.game_over and connect4.wins != self.my_token:
            return -1, -1   # I lose, so best score is -1
        
        if connect4.game_over and connect4.wins == None:
            return 0, -1    # Its a tie, no one wins.
        
        if d == 0:
            return self.heuristic(connect4), -1 # 0 moves to predict, we are nobrainers and opponent always wins
        
        best_score = -1 if resultOfMove == 1 else 1
        best_move = 0
        for moveToTest in connect4.possible_drops():
            game = deepcopy(connect4)
            game.drop_token(moveToTest)

            if resultOfMove == 1:   # if move is winning, maximize result
                score, _ = self.minmax(game, 0, d-1)
                if score > best_score:
                    score = best_score
                    best_move = moveToTest
            
            else:                   # we are loosing, minimize loss
                score, _ = self.minmax(game, 1, d-1)
                if score < best_score:
                    score = best_score
                    best_move = moveToTest
        
        return [best_score, best_move]
    

    def heuristic(self, connect4: 'Connect4') -> int:
        weights = [0, 5, 50, 500]     # the more we have valid fours, the better
        totalFours = 0
        score = 0
        for four in connect4.iter_fours():
            totalFours += 1
            ours = four.count(self.my_token)
            empty = four.count('_')
            theirs = 4 - ours - empty

            if theirs != 0:
                continue

            score += weights[ours]
        return score / (totalFours * weights[-1])