from RevisedNumericalSolver import torchstate
import torch
import time



# Perturbs input vector using normal distribution
# takes in float standard deviation
# Requires floats
def perturb(vec, std):
    return torch.tensor([torch.normal(mean=vec[0], std=torch.tensor(std)),
                         torch.normal(mean=vec[1], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[3], std=torch.tensor(std)),
                         torch.normal(mean=vec[4], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[6], std=torch.tensor(std)),
                         torch.normal(mean=vec[7], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[9], std=torch.tensor(std)),
                         torch.normal(mean=vec[10], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[12], std=torch.tensor(std)),
                         torch.normal(mean=vec[13], std=torch.tensor(std)),
                         0.0,
                         torch.normal(mean=vec[15], std=torch.tensor(std)),
                         torch.normal(mean=vec[16], std=torch.tensor(std)),
                         0.0,], requires_grad = True)

# Compares two states and returns a numerical value rating how far apart the two states in the three
# body problem are to each other. Takes in two tensor states and returns tensor of value of distance rating
# The higher the score, the less similar they are
def nearest_position(particle, state1, state2):
    mse = torch.nn.L1Loss()
    if particle == 1:
        return mse(state1[:3], state2[:3]) + mse(state1[9:12], state2[9:12])
    elif particle == 2:
        return mse(state1[3:6], state2[3:6]) + mse(state1[12:15], state2[12:15])
    elif particle == 3:
        return mse(state1[6:9], state2[6:9]) + mse(state1[15:18], state2[15:18])
    else:
        print("bad input")


# Finds the most similar state to the initial position in a data set
def nearest_position_state(particle, state, data_set, min, max, time_step):
    i = min
    max_val = torch.tensor([100000000])
    index = -1
    while i < max:
        if nearest_position(particle, state, data_set[i]).item() < max_val.item():
            index = i
            max_val = nearest_position(particle, state, data_set[i])

        i += 1
    #print(f"Time: {index*time_step}")
    return index


# Optimization Process of the Algorithm

def loss_values(identity, vec, m_1, m_2, m_3, lr, time_step, num_epochs, max_period, opt_func):
    initial_vec = vec
    optimizer = opt_func([vec], lr=lr)
    losses = []
    #result = {}
    #loss_values = []
    i = 0
    print("start")
    while i < num_epochs:
        print(i)
        input_vec = torch.cat((vec, torch.tensor([m_1,m_2,m_3])))

        data_set = torchstate(input_vec, time_step, max_period, "rk4")

        # In place in case the loss values ever start repeating themselves-in that case switch optimizer for one iteration
        if len(losses) > 10:
            if losses[-1] == losses[-3]:
                #print("Repeated")
                optimizer = torch.optim.SGD([vec], lr=.00001)
                
        optimizer.zero_grad()
      
        first_index = nearest_position_state(1, data_set[0], data_set, 300, len(data_set), time_step)
        first_particle_state = data_set[first_index]
        second_index = nearest_position_state(2, data_set[0], data_set, 300, len(data_set), time_step)
        second_particle_state = data_set[second_index]
        third_index = nearest_position_state(3, data_set[0], data_set, 300, len(data_set), time_step)
        third_particle_state = data_set[third_index]
        loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0],
                                                                                         second_particle_state) + nearest_position(
            3, data_set[0], third_particle_state)
        # Tracks loss then optimizes vector
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        i += 1



