from collections import defaultdict
from enum import Enum
import random
from typing import Tuple, Sequence, Callable

import numpy as np
import pandas as pd
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from tqdm import trange


class PortfolioAction(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2


def convert_prices_to_discrete_state(prev_data_df: pd.DataFrame, current_data_df: pd.DataFrame) -> Tuple:
    # if prev_data_df['tic'].tolist() == current_data_df['tic'].tolist():
    # for x, y in zip(prev_data_df['tic'].tolist(), current_data_df['tic'].tolist()):
    #    print(x == y)
    # raise IndexError("Dfs are wrong")
    percent_diffs = ((current_data_df['open'] - prev_data_df['open']) / prev_data_df['open']) * 100

    discrete = []
    for dif in percent_diffs.values:
        if dif > 5:
            discrete.append(0)
        elif 5 > dif > 0:
            discrete.append(1)
        elif 0 > dif > -5:
            discrete.append(2)
        elif -5 > dif:
            discrete.append(3)
        else:
            discrete.append(4)
    return tuple(discrete)


def argmax(arr: Sequence[float]) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    largest = max(arr)
    in_case_of_ties = []
    for i in range(len(arr)):
        if arr[i] == largest:
            in_case_of_ties.append(i)
    return random.choice(in_case_of_ties)


def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        a_star = argmax(Q[state])
        if np.random.random() < epsilon:
            action = random.choice(list(PortfolioAction))
        else:
            probabilities = [(1 - epsilon + (epsilon / num_actions)) if a.value == a_star else epsilon / num_actions for
                             a in list(PortfolioAction)]
            return np.random.choice(list(PortfolioAction), p=probabilities)
        return action

    return get_action


def update_single_stock_percent(perc, act):
    if act == PortfolioAction.BUY.value:
        perc += 0.1
    elif act == PortfolioAction.SELL.value:
        perc -= 0.1
    elif act == PortfolioAction.HOLD.value:
        perc = perc

    if perc > 1:
        perc = 1
    if perc < 0:
        perc = 0
    return perc


def softmax_normalization(actions):
    numerator = np.exp(actions)
    denominator = np.sum(np.exp(actions))
    softmax_output = numerator / denominator
    return softmax_output


def sarsa_single_stock(env: StockPortfolioEnv, num_episodes: int, gamma: float, epsilon: float, step_size: float,
                       stock: int = 0):
    """SARSA algorithm."""
    Q = defaultdict(lambda: np.zeros(len(PortfolioAction)))
    policy = create_epsilon_policy(Q, epsilon)
    episodes = []
    for _ in trange(num_episodes, desc="Episode", leave=False):
        env.reset()
        # take initial step without investing anything to get a price to compare with
        env.step(np.zeros(28))

        previous_data = env.data
        env.step(np.zeros(28))
        current_data = env.data
        # create discrete state
        S = convert_prices_to_discrete_state(prev_data_df=previous_data, current_data_df=current_data)
        # given state which includes all stocks return buy sell hold for the single stock we are looking at
        A = policy(S)
        A = A.value

        episode = []
        percent = 0
        while True:
            # need to convert buy sell hold into a percentage for env step
            percent = update_single_stock_percent(percent, A)

            # convert our action into the portfolio percentages
            portfolio_breakdown = np.zeros(28)
            portfolio_breakdown[stock] = percent
            # print(portfolio_breakdown)
            # print(softmax_normalization(portfolio_breakdown))
            # TODO THE ENVIRONMENT AUTO NORMALIZES ACTION. CHANGE TO UPDATE REFLECT PORTFOLIO BREAKDOWN???
            # JUST HAVE 28 SARSA's and have softmax handle it in the environment
            next_state, reward, done, _, _ = env.step(portfolio_breakdown)

            # update ticker prices
            previous_data = current_data
            current_data = env.data

            # create discrete next state
            next_state = convert_prices_to_discrete_state(prev_data_df=previous_data, current_data_df=current_data)

            # record information
            episode.append((S, A, reward, percent))

            # get next action
            A_star = policy(next_state)
            # update
            A_star = A_star.value

            Q[S][A] = Q[S][A] + (step_size * (reward + (gamma * Q[next_state][A_star]) - Q[S][A]))
            S = next_state
            A = A_star

            if done:
                break

        episodes.append(episode)
    return env, episodes