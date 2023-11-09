# NGLMORL
Code repository for the thesis Normative Compliance in Lexicographic Multi-Objective Reinforcement Learning Agents by Bernhard Schiehl.

Some code inspired by:
* [Joar Skalse and Lewis Hammond](https://github.com/lrhammond/lmorl) (lexicographic agents)
* [Jamie Spacco](https://github.com/jspacco/pac3man) (Berkeley Pac-Man implementation for python 3)
* [Tycho van der Ouderaa](https://github.com/tychovdo/PacmanDQN) and [Andrei C.](https://github.com/applied-ai-collective/Pacman-Deep-Q-Network) (Pac-Man DQNs)

To test a lexicographic agent, start the normative supervisor (see https://github.com/lexeree/normative-player-characters), navigate to pac3man/reinforcement and run:

`python pacman.py -p [agent name] -l [layout name] -x [number of training episodes] -n [number of total episodes] --norm=[name of normative system] --reason=[name of reasoner] --lex`

