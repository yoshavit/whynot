"""Implementation of the Perdomo et. al model of strategic classification.

The data is from the Kaggle Give Me Some Credit dataset:

    https://www.kaggle.com/c/GiveMeSomeCredit/data,

"""
import copy
import dataclasses
from typing import Any

import sklearn.linear_model
import sklearn.decomposition

import whynot as wn
import whynot.traceable_numpy as np
from whynot.dynamics import BaseConfig, BaseIntervention, BaseState
from whynot.simulators.responsive_credit.dataloader import CreditData


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the Credit model."""

    #: Matrix of agent features (e.g. https://www.kaggle.com/c/GiveMeSomeCredit/data)
    features: np.ndarray = CreditData.features

    #: Vector proportional to the probability a person experiences financial distress
    labels: np.ndarray = CreditData.labels

    def values(self):
        """Return the state as a dictionary of numpy arrays."""
        return {name: getattr(self, name) for name in self.variable_names()}

def create_true_outcome_theta():
    """Trains and computes a sparse true outcome-generating model vector

    Returns
    -------
    np.ndarray
        The true outcome-generating theta
    """

    # load credit data
    features, labels, info = CreditData.load_data()

    # train classifier using L1 regularization to create a sparse model
    lr = sklearn.linear_model.LogisticRegression(
        solver='liblinear', 
        fit_intercept=False, 
        penalty='l1', 
        C=10**(-2.4)
        ).fit(features, labels)

    # let the linear model be the parameters of that classifier 
    # WITHOUT the sigmoid final layer
    true_theta = lr.coef_
    assert true_theta.shape == (1, features.shape[1])
    return true_theta


def get_visible_feature_projector(config, x):
    """Create matrix that masks away invisible features
    """
    vf_mask = np.zeros(11)
    vf_mask[config.visible_features] = 1
    return np.eye(vf_mask)
    

def get_changeable_feature_inds():
    """Return the manipulable feature indices
    """
    changeable_feature_names = ['age', 'NumberOfDependents']
    return [i for (i, fname) in enumerate(CreditData.feature_names) 
            if not (fname in changeable_feature_names)]


def create_pca_action_matrix(d=4):
    """Return the dimensions along which agents can change their features
    by running PCA on the data to find the dimensions of usual variation

    Parameters
    ----------
    d : int, optional
        dimensionality of action space, by default 4
    """

    features, *_ = CreditData.load_data()
    changeable_feature_inds = get_changeable_feature_inds()
    # let's mask away the unchangeable features
    cf_mask = np.zeros(features.shape[1])
    cf_mask[changeable_feature_inds] = 1
    cf_mask = np.eye(cf_mask)
    features_masked = features @ cf_mask

    # compute directions of large variance in the original data
    pca = sklearn.decomposition.PCA(n_components=d)
    pca.fit(features_masked)

    action_matrix = pca.explained_variance_

    # just to double-check that the unchangeable directions are zeroed
    action_matrix = action_matrix @ cf_mask
    assert action_matrix.shape == (features.shape[1], d)

    return action_matrix


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of Credit simulator dynamics.

    Examples
    --------
    >>> # Configure simulator for run for 10 iterations
    >>> config = Config(start_time=0, end_time=10, delta_t=1)

    """

    # Dynamics parameters

    # Features changeable by the agents
    changeable_features: list = get_changeable_feature_inds()

    #: Model how much the agent adapt her features in response to a classifier
    epsilon: float = 0.1

    #: Parameters for logistic regression classifier used by the institution
    theta: np.ndarray = np.ones((11, 1))

    #: L2 penalty on the logistic regression loss
    l2_penalty: float = 0.0

    #: Whether or not dynamics have memory
    memory: bool = False

    #: State systems resets to if no memory.
    base_state: Any = State()

    # Responsive-credit-specific attributes
    # True outcome-generating linear model parameters
    true_theta: np.ndarray = create_true_outcome_theta()

    # Action matrix for computing feature changes
    action_matrix: np.ndarray = create_pca_action_matrix()

    # The indices of the features visible to the decision-maker
    visible_features: list = list(range(11))

    # Simulator book-keeping
    #: Start time of the simulator
    start_time: int = 0
    #: End time of the simulator
    end_time: int = 5
    #: Spacing of the evaluation grid
    delta_t: int = 1



class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the Credit model.

    An intervention changes a subset of the configuration variables in the
    specified year. The remaining variables are unchanged.

    Examples
    --------
    >>> # Starting at time 25, update the classifier to random chance.
    >>> config = Config()
    >>> Intervention(time=25, theta=np.zeros_like(config.theta))

    """

    def __init__(self, time=30, **kwargs):
        """Specify an intervention in credit.

        Parameters
        ----------
            time: int
                Time of intervention in simulator dynamics.
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


def squared_loss(config, features, labels, theta):
    """Evaluate the performative loss for the regressor."""

    config = config.update(Intervention(theta=theta))

    # compute squared loss
    preds = features @ config.theta
    loss = (preds - labels) ** 2
    return loss


def agent_model(features, config):
    """Compute agent reponse to the regressor and adapt features accordingly.
    """
    # Move everything by epsilon in the direction towards better classification
    strategic_features = np.copy(features)

    # the actions resulting from quadratic cost
    actions = config.action_matrix.T @ config.theta.T

    # convert those action weights to feature effects, and add to original features
    strategic_features += config.action_matrix @ actions
    return strategic_features


def dynamics(state, time, config, intervention=None):
    """Perform one round of interaction between the agents and the credit scorer.

    Parameters
    ----------
        state: whynot.simulators.responsive_credit.State
            Agent state at time TIME
        time: int
            Current round of interaction
        config: whynot.simulators.responsive_credit.Config
            Configuration object controlling the interaction, e.g. classifier
            and agent model
        intervention: whynot.simulators.responsive_credit.Intervention
            Intervention object specifying when and how to update the dynamics.

    Returns
    -------
        state: whynot.simulators.responsive_credit.State
            Agent state after one step of strategic interaction.

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    # Only use the current state if the dynamics have memory.
    # Otherwise, agents "reset" to the base dataset. The latter
    # case is the one treated in the performative prediction paper.
    if config.memory:
        features, labels = state.features, state.labels
    else:
        features, labels = config.base_state.features, config.base_state.labels

    # Update features in response to classifier. Labels are fixed.
    strategic_features = agent_model(features, config)
    strategic_labels = config.true_theta @ strategic_features
    return strategic_features, strategic_labels


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the Credit model.

    Parameters
    ----------
        initial_state: whynot.responsive_credit.State
        config: whynot.responsive_credit.Config
            Base parameters for the simulator run
        intervention: whynot.responsive_credit.Intervention
            (Optional) Parameters specifying a change in dynamics
        seed: int
            Unused since the simulator is deterministic.

    Returns
    -------
        run: whynot.dynamics.Run
            Simulator rollout

    """
    # Iterate the discrete dynamics
    times = [config.start_time]
    states = [initial_state]
    state = copy.deepcopy(initial_state)

    for step in range(config.start_time, config.end_time):
        next_state = dynamics(state, step, config, intervention)
        state = State(*next_state)
        states.append(state)
        times.append(step + 1)

    return wn.dynamics.Run(states=states, times=times)


if __name__ == "__main__":
    print(simulate(State(), Config(end_time=2)))
