import argparse
import importlib
import signal
import time

def handler(signum, frame):
    global userquit
    print('User quit (CTRL-C) [signal: %d]' %signum)
    userquit = True

def get_args():
    parser = argparse.ArgumentParser(description='RL games')
    parser.add_argument('-gamma', type=float, help='discount factor [default: 1.0]', default=.9)
    parser.add_argument('-epsilon', type=float, help='epsilon greedy factor [default: -1 = adaptive]', default=-2)
    parser.add_argument('-alpha', type=float, help='alpha factor (-1 = based on visits) [default: -1]', default=-1)
    parser.add_argument('-nstep', type=int, help='n-steps updates [default: 1]', default=1)
    parser.add_argument('-lambdae', type=float, help='lambda eligibility factor [default: -1 (no eligibility)]',
                        default=-1)
    parser.add_argument('--debug', help='debug flag', action='store_true')
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
def learn(game, agent, maxtime=-1, stopongoal=False):
    global optimalPolicyFound, userquit

    run = True
    userquit = False
    last_goalreached = False
    next_optimal = False
    iter_goal = 0  # iteration in which first goal policy if found

    # timing the experiment
    exstart = time.time()
    elapsedtime0 = game.elapsedtime

    if (maxtime > 0 and game.elapsedtime >= maxtime):
        run = False
    # elif (game.iteration>0 and game.iteration<100 and not game.debug): # try an optimal run  ???
    #    next_optimal = True
    #    game.iteration -= 1

    while (run and (args.niter < 0 or game.iteration <= args.niter)):
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
            agent.notify_endofepisode(game.iteration)
            game.elapsedtime = (time.time() - exstart) + elapsedtime0
            game.print_report()
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

    if optimalPolicyFound:
        print("\n***************************")
        print("***  Goal policy found  ***")
        print("***************************\n")
        if (agent.Qapproximation):
            for a in range(0, game.nactions):
                print("Q[%d]" % a)
                print("       ", agent.Q[a].get_weights())

if __name__ == "__main__":

    # Set the signal handler
    signal.signal(signal.SIGINT, handler)
    args = get_args()

    args.niter = 1000
    args.rows = 5
    args.cols = 7
    agent = importlib.import_module('RLAgent').SarsaAgent()
    game = importlib.import_module('Sapientino').Sapientino(args.rows, args.cols)
    game.nvisitpercol = 3
    game.debug = args.debug

    agent.gamma = args.gamma
    agent.epsilon = args.epsilon
    agent.alpha = args.alpha
    agent.nstepsupdates = args.nstep
    agent.lambdae = args.lambdae
    agent.debug = args.debug

    game.init(agent)
    learn(game, agent)