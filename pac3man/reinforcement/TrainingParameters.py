import inspect

from constants import env_names
from constants import agent_names
from dataclasses import dataclass, field

# Doesn't do anything yet
parameters_used_by_all_agents = ["env_name", "agent_name", "num_episodes", "num_interacts",
                                 "test_group_label", "save_every_n"]

inconsistent_parameters = {
    "LDQN": ["epsilon", "buffer_size", "no_cuda", "update_every", "slack", "reward_size", "network"]
}


@dataclass
class TrainingParameters:
    # dataclass attributes created in __init__ and address by self.
    # TODO - perhaps add validation to check when unnecessary parameters are specified e.g. batch_size for tabular

    width: int # The width of the environment
    height: int # The height of the environment
    num_training: int
    epsilon_decay: float

    env_name: str = "Pacman" # the str representing the environment, found in src.constants.env_names
    agent_name: str =  "Agent" # the str representing the agent, found in src.constants.agent_names
    network: str = "DNN"  # The type of network to use within an agent ("DNN" or "CNN")

    
    num_interacts: int = None  # The number of interactions between agent and environment during training

    test_group_label: str = None  # A label used to identify a batch of experiments
    save_every_n: int = None  # How frequently should copies of the model be saved during training?

    buffer_size: int = 10000 # PyTorch buffer size to use during training
    batch_size: int = 64  # PyTorch batch size to use during training
    update_every: int = 1  # After how many interacts should we update the model?
    update_every_eps = 1  # Deprecated
    update_steps: int = 1  # Used by LDQN

    epsilon_start: float = 0.9  # Hyperparameter used in epsilon-greedy algorithms (and others)
    epsilon: float = 0.05
    
    slack: list = field(default_factory= lambda: [1.0, 0.1]) # Hyperparameter used by lexicographic algorithms
    slack_start: list = field(default_factory= lambda: [0.1, 0.5]) # Hyperparameter used by lexicographic algorithms with dynamic slack
    additive_slack: bool = True # Hyperparameter used by lexicographic algorithms 

    learning_rate: float = 1e-3 # Hyperparameter used by the optimiser
    tau = 0.005 # update rate for target network

    trained: bool = False # load a trained model
    model_path: str = None # path to a trained model
    largeEnv: bool = False

    # AproPo
    lambda_lr_2: float = 0.05
    alpha: float = 1
    beta: float = 0.95

    no_cuda: bool = False

    reward_size: int = 2
    constraint: int = 0.1

    constraints = [(0.3, 0.5),
                   (0.0, 0.1)]

    lextab_on_policy: bool = False

    # After dataclass attributes are initialised, validate the training parameters
    def __post_init__(self):
        self.buffer_size = int(self.buffer_size)
        self.batch_size = int(self.batch_size)
        self.update_every = int(self.update_every)
        self.update_steps = int(self.update_steps)
        self.epsilon = float(self.epsilon)
        self.epsilon_start = float(self.epsilon_start)
        for i in range(len(self.slack)):
            self.slack[i] = float(self.slack[i])
            self.slack_start[i] = float(self.slack_start[i])
        self.learning_rate = float(self.learning_rate)
        self.lambda_lr_2 = float(self.lambda_lr_2)
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.reward_size = int(self.reward_size)
        self.constraint = float(self.constraint)
        if type(self.no_cuda) == str:
            self.no_cuda = self.no_cuda == "True"
        if type(self.largeEnv) == str:
            self.largeEnv = self.largeEnv == "True"
        if type(self.trained) == str:
            self.trained = self.trained == "True"
        self.lextab_on_policy = self.lextab_on_policy == "True"
        # assert (self.agent_name in agent_names)
        # assert (self.env_name in env_names)
        # assert (self.network in ["CNN", "DNN"])

        # if self.num_episodes is not None:
        #     raise NotImplementedError("num_episodes has been deprecated")

        # assert (not (self.num_interacts is None and self.num_episodes is None))
        # assert (self.num_interacts is None or self.num_episodes is None)
        # if self.num_interacts is not None:
        #     self.is_interact_mode = True
        # else:
        #     self.is_interact_mode = False

    def render_and_print(self):
        print(self.render_to_string())

    def render_to_string(self):
        x = ""
        for atr_name, atr in inspect.getmembers(self):
            if not atr_name.startswith("_") and not inspect.ismethod(atr):
                x += f" < {atr_name}: {str(atr)} >, "
        return x

    def render_to_file(self, dir):
        x = self.render_to_string()
        with open(dir, "w") as f: f.write(x)
