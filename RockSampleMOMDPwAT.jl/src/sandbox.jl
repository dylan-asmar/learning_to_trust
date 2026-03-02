using POMDPs
using POMDPTools
using MOMDPs
using RockSampleMOMDPProblemAT
using Printf
using Compose

discount_factor = 0.95
exit_reward = 10.0
good_rock_reward = 10.0
bad_rock_penalty = -10.0
step_penalty = 0.0

map_size = (8, 8)
sensor_efficiency = 10.0
sensor_use_penalty = -1.0
rocks_positions = [(1,1),
                (2,7),
                (6,2),
                (7,8)]
init_pos = (3,4)

num_asks = 0

momdp84 = RockSampleMOMDPAT(;
    map_size=map_size,
    rocks_positions=rocks_positions,
    init_pos=init_pos,
    sensor_efficiency=sensor_efficiency,
    bad_rock_penalty=bad_rock_penalty,
    good_rock_reward=good_rock_reward,
    step_penalty=step_penalty,
    sensor_use_penalty=sensor_use_penalty,
    exit_reward=exit_reward,
    discount_factor=discount_factor,
    num_asks=num_asks
)
    

map_size = (7, 8)
sensor_efficiency = 20.0
sensor_use_penalty = 0.0
rocks_positions = [
    (1,2),
    (2,7),
    (3,1),
    (3,5),
    (4,2),
    (4,5),
    (6,6),
    (7,4)
]
init_pos = (1,4)

num_asks = 0

momdp78 = RockSampleMOMDPAT(;
    map_size=map_size,
    rocks_positions=rocks_positions,
    init_pos=init_pos,
    sensor_efficiency=sensor_efficiency,
    bad_rock_penalty=bad_rock_penalty,
    good_rock_reward=good_rock_reward,
    step_penalty=step_penalty,
    sensor_use_penalty=sensor_use_penalty,
    exit_reward=exit_reward,
    discount_factor=discount_factor,
    num_asks=num_asks
)


b = initialstate(momdp84)
s84 = (s=(RockSampleMOMDPXState((3,4), 0), RockSampleMOMDPYState{4}((true, true, true, false), 1.0)), b=b, a=6)



render(momdp84, s84; viz_rock_state=false, viz_types=true, text_below=true)
render(momdp84, s84; viz_rock_state=false, viz_types=true, text_below=false)
render(momdp84, s84; viz_rock_state=false, viz_types=false, text_below=true)
p = render(momdp84, s84; viz_rock_state=false, viz_types=false, text_below=false)



# render(momdp84, s84; viz_rock_state=false, viz_types=false)

s78 = (s=(RockSampleMOMDPXState((1,4), 0), RockSampleMOMDPYState{4}((true, true, true, false), 1.0)), )
render(momdp78, s78; viz_rock_state=false, viz_types=false, text_below=false)
