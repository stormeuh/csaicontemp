import argparse
import importlib
import signal
import time
import threading
import os
from copy import deepcopy
import numpy as np
import matplotlib as plt
from RLAgent import SarsaAgent
from Sapientino import Sapientino
from Minecraft import Minecraft

optimalPolicyFound = False

def handler(signum, frame):
    global userquit
    print('User quit (CTRL-C) [signal: %d]' %signum)
    userquit = True

def get_args():
    parser = argparse.ArgumentParser(description='RL games')
    parser.add_argument('name', type=str)
    parser.add_argument('game', type=str)
    parser.add_argument('-gamma', type=float, help='discount factor [default: 0.9]', default=0.9)
    parser.add_argument('-epsilon', type=float, help='epsilon greedy factor [default: -1 = adaptive]', default=-2)
    parser.add_argument('-alpha', type=float, help='alpha factor (-1 = based on visits) [default: -1]', default=-1)
    parser.add_argument('-nstep', type=int, help='n-steps updates [default: 1]', default=10)
    parser.add_argument('-lambdae', type=float, help='lambda eligibility factor [default: -1 (no eligibility)]',
                        default=-1)
    parser.add_argument('--debug', help='debug flag', action='store_true')
    parser.add_argument('--shielding', help='enable for shielding, disable for restraining bolt', action='store_true')
    return parser.parse_args()

def execution_step(game, agent):
    x = game.getstate() # current state
    if (game.isAuto):  # agent choice
        a = agent.decision(x) # current action
    else: # otherwise command is set by user input
        a = game.getUserAction() # action selected by user
    game.update(a)
    x2 = game.getstate() # new state
    r = game.getreward() # reward
    agent.notify(x,a,r,x2)

# learning process
def learn(game, agent, maxtime=-1, stopongoal=False, logging=True):
    global optimalPolicyFound, userquit

    run = True
    userquit = False
    last_goalreached = False
    next_optimal = False
    rewards = np.zeros(args.niter)
    iter_goal = 0  # iteration in which first goal policy if found

    # timing the experiment
    exstart = time.time()
    elapsedtime0 = game.elapsedtime

    if (maxtime > 0 and game.elapsedtime >= maxtime):
        run = False
    # elif (game.iteration>0 and game.iteration<100 and not game.debug): # try an optimal run  ???
    #    next_optimal = True
    #    game.iteration -= 1

    while (run and (args.niter < 0 or game.iteration < args.niter)):
        if (game.iteration % 100 == 0): print(game.iteration)
        game.reset()  # increment game.iteration
        game.draw()
        time.sleep(game.sleeptime)
        if ((last_goalreached and agent.gamma == 1) or next_optimal):
            agent.optimal = True
            next_optimal = False
        while (run and not game.finished):
            grun = game.input()
            if (not grun):
                userquit = True
            if game.pause:
                time.sleep(1)
                continue

            execution_step(game, agent)

            if (agent.error):
                game.pause = True
                agent.debug = True
                agent.error = False
            game.draw()
            time.sleep(game.sleeptime)

        # episode finished
        if (game.finished):
            rewards[game.iteration - 1] = game.cumreward
            agent.notify_endofepisode(game.iteration)
            game.elapsedtime = (time.time() - exstart) + elapsedtime0
            if(logging): game.print_report()
            time.sleep(game.sleeptime)

        # end of experiment
        if (agent.optimal and game.goal_reached()):
            optimalPolicyFound = True
            if (agent.gamma == 1 or stopongoal):
                run = False
            # elif (iter_goal==0):
            #    iter_goal = game.iteration
            # elif (game.iteration>int(1.5*iter_goal)):
            #    run = False
        elif (maxtime > 0 and game.elapsedtime >= maxtime):
            run = False
        elif (userquit or game.userquit):
            run = False

        last_goalreached = game.goal_reached()

    if optimalPolicyFound and logging:
        print("\n***************************")
        print("***  Goal policy found  ***")
        print("***************************\n")
        if (agent.Qapproximation):
            for a in range(0, game.nactions):
                print("Q[%d]" % a)
                print("       ", agent.Q[a].get_weights())
    return rewards

# evaluation process
def evaluate(game, agent, n, logging=True):  # evaluate best policy n times (no updates)
    i = 0
    run = True
    rewards = np.zeros(n)
    actions = np.zeros(n)
    # game.sleeptime = 0.00001
    # if (game.gui_visible):
    #     game.sleeptime = 0.1
    #     game.pause = True

    while (i < n and run):
        game.reset()
        game.draw()
        time.sleep(game.sleeptime)

        agent.optimal = True
        while (run and not game.finished):
            run = game.input()
            if game.pause:
                time.sleep(1)
                continue
            execution_step(game, agent)
            game.draw()
            time.sleep(game.sleeptime)
        if logging: game.print_report(printall=True)
        rewards[i] = game.cumreward
        actions[i] = game.numactions
        if (game.gui_visible):
            n = 3
            j = 0
            while (j < n):
                time.sleep(1)
                game.input()
                if game.pause:
                    time.sleep(1)
                j += 1
            time.sleep(3)
        i += 1
    agent.optimal = False
    return (rewards.mean(), actions.mean())

def run_test(game, agent):
    run_game = deepcopy(game)
    run_agent = deepcopy(agent)

    run_game.init(run_agent)
    learning_rewards = learn(run_game, run_agent, logging=True)
    policy_rewards, policy_actions = evaluate(run_game, run_agent, 10, logging=False)
    return learning_rewards, policy_rewards, policy_actions

if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # Set the signal handler
    signal.signal(signal.SIGINT, handler)
    args = get_args()
    if (args.shielding):
        print "Shielding"
    else:
        print "Bolts"

    n = 50
    args.niter = 1000
    args.rows = 5
    args.cols = 7
    learning_rewards = np.zeros([args.niter,n])
    policy_rewards = np.zeros(n)
    policy_actions = np.zeros(n)
    for i in range(0, n):
        print i
        agent = SarsaAgent()
        game = Sapientino(args.rows, args.cols)
        game.nvisitpercol = 3
        game.debug = args.debug
        game.shielding = args.shielding
        game.gui_visible = False

        agent.gamma = args.gamma
        agent.epsilon = args.epsilon
        agent.alpha = args.alpha
        agent.nstepsupdates = args.nstep
        agent.lambdae = args.lambdae
        agent.debug = args.debug
        game.init(agent)
        learning_rewards[:,i] = learn(game, agent, logging=False)
        policy_rewards[i], policy_actions[i] = evaluate(game, agent, 10, logging=False)
    np.savetxt(args.name + "_learning_rewards.dat", np.mean(learning_rewards, axis=1), delimiter=',')
    np.savetxt(args.name + "_policy_rewards.dat", policy_rewards, delimiter=',')
    np.savetxt(args.name + "_policy_actions.dat", policy_actions, delimiter=',')
