"""Credit simulator initialization."""

from whynot.simulators.responsive_credit.simulator import (
    agent_model,
    Config,
    dynamics,
    Intervention,
    simulate,
    squared_loss,
    get_visible_feature_projection,
    State,
)

from whynot.simulators.responsive_credit.dataloader import CreditData

from whynot.simulators.responsive_credit.environments import *

SUPPORTS_CAUSAL_GRAPHS = True
