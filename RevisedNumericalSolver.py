import torch
from torchdiffeq import odeint_adjoint as odeint
import torch.nn as nn



# Differential Equations Governing Three Bodies
# w is flattened input torch tensor with position vector followed by velocity vector followed by
# the three masses
def ThreeBodyDiffEq(w):
    # Unpacks flattened array
    r_1 = w[:3]
    r_2 = w[3:6]
    r_3 = w[6:9]
    v_1 = w[9:12]
    v_2 = w[12:15]
    v_3 = w[15:18]
    m_1 = w[18]
    m_2 = w[19]
    m_3 = w[20]

    # Torch calculated displacement vector magnitudes
    r_12 = torch.sqrt((w[3:6][0] - w[:3][0]) ** 2 + (w[3:6][1] - w[:3][1]) ** 2 + (w[3:6][2] - w[:3][2]) ** 2)
    r_13 = torch.sqrt((r_3[0] - r_1[0]) ** 2 + (r_3[1] - r_1[1]) ** 2 + (r_3[2] - r_1[2]) ** 2)
    r_23 = torch.sqrt((r_3[0] - r_2[0]) ** 2 + (r_3[1] - r_2[1]) ** 2 + (r_3[2] - r_2[2]) ** 2)

    # The derivatives of the velocities. Returns torch tensor
    # G is assumed to be 1
    dv_1bydt = m_2 * (r_2 - r_1) / r_12 ** 3 + m_3 * (r_3 - r_1) / r_13 ** 3
    dv_2bydt = m_1 * (r_1 - r_2) / r_12 ** 3 + m_3 * (r_3 - r_2) / r_23 ** 3
    dv_3bydt = m_1 * (r_1 - r_3) / r_13 ** 3 + m_2 * (r_2 - r_3) / r_23 ** 3

    # The derivatives of the positions
    dr_1bydt = v_1
    dr_2bydt = v_2
    dr_3bydt = v_3

    # Vector in form [position derivatives, velocity derivatives]
    derivatives = torch.stack([dr_1bydt, dr_2bydt, dr_3bydt, dv_1bydt, dv_2bydt, dv_3bydt]).flatten()
    # Includes mass derivatives of 0
    derivatives = torch.cat((derivatives, torch.tensor([0,0,0])))

    # Flattens into 1d array for use
    return derivatives


# v = torch.tensor([-1.0089e+00, -5.6278e-03,  0.0000e+00,  1.0043e+00, -2.1155e-06,
#          0.0000e+00,  4.5193e-03,  5.6299e-03,  0.0000e+00,  3.4643e-01,
#          5.2202e-01,  0.0000e+00,  3.3615e-01,  5.3153e-01,  0.0000e+00,
#         -6.9759e-01, -1.0589e+00,  0.0000e+00,  3.5708e+01,  3.5706e+01,
#          3.5706e+01])
# print(ThreeBodyDiffEq(v))


# Uses the Runge Kutta 4 method to numerically approximate and return
# the entire trajectory of positions and velocities of the particles
# w is flattened input tensor of 21 dimension
def get_full_state(w, dt, time_span):

    results = w


    # Number of times we iterate over
    number_of_points = int(time_span / dt)
    i = 0

    while (i < number_of_points):
        # Calculates the k values for the Runge Kutta 4 method
        k_1 = ThreeBodyDiffEq(w)
        k_2 = ThreeBodyDiffEq(w+dt/2*k_1)
        k_3 = ThreeBodyDiffEq(w+dt/2*k_2)
        k_4 = ThreeBodyDiffEq(w+dt*k_3)

        # Iterates to next value using Runge Kutta 4 formula
        w = w + 1 / 6 * dt * (k_1 + 2 * k_2 + 2 * k_2 + k_4)

        # Adds new inpt vector step to total trajectory
        if i == 0:
            results = torch.stack((results,w))
        else:
            results = torch.vstack((results, w))

        #print(i)
        i += 1

    return results


class ThreeBody(nn.Module):
    def forward(self, t, y):
        # Unpacks flattened array
        # r_1 = y[:3]
        # r_2 = y[3:6]
        # r_3 = y[6:9]
        # v_1 = y[9:12]
        # v_2 = y[12:15]
        # v_3 = y[15:18]
        # m_1 = y[18]
        # m_2 = y[19]
        # m_3 = y[20]

        # Torch calculated displacement vector magnitudes
        r_12 = torch.sqrt((y[3:6][0] - y[:3][0]) ** 2 + (y[3:6][1] - y[:3][1]) ** 2 + (y[3:6][2] - y[:3][2]) ** 2)
        r_13 = torch.sqrt((y[6:9][0] - y[:3][0]) ** 2 + (y[6:9][1] - y[:3][1]) ** 2 + (y[6:9][2] - y[:3][2]) ** 2)
        r_23 = torch.sqrt((y[6:9][0] - y[3:6][0]) ** 2 + (y[6:9][1] - y[3:6][1]) ** 2 + (y[6:9][2] - y[3:6][2]) ** 2)

        # The derivatives of the velocities. Returns torch tensor
        # G is assumed to be 1
        dv_1bydt = y[19] * (y[3:6] - y[:3]) / r_12 ** 3 + y[20] * (y[6:9] - y[:3]) / r_13 ** 3
        dv_2bydt = y[18] * (y[:3] - y[3:6]) / r_12 ** 3 + y[20] * (y[6:9] - y[3:6]) / r_23 ** 3
        dv_3bydt = y[18] * (y[:3] - y[6:9]) / r_13 ** 3 + y[19] * (y[3:6] - y[6:9]) / r_23 ** 3



        # Vector in form [position derivatives, velocity derivatives]
        derivatives = torch.stack([y[9:12], y[12:15], y[15:18], dv_1bydt, dv_2bydt, dv_3bydt]).flatten()
        # Includes mass derivatives of 0
        derivatives = torch.cat((derivatives, torch.tensor([0, 0, 0])))

        # Flattens into 1d array for use
        return derivatives


def torchstate(y, dt, time_span, method):
    t = torch.linspace(0., time_span, steps=int(time_span/dt))
    return odeint(ThreeBody(), y, t, method=method)


if __name__ == "__main__":
    w = torch.tensor(
        [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2 * 0.347111, -2 * 0.532728, 0, 35,
         35, 35])
