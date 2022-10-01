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



class ThreeBody(nn.Module):
    def forward(self, t, y):

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

    
# Calculates full trajecotry through fully differentiable diff eq solver
def torchstate(y, dt, time_span, method):
    t = torch.linspace(0., time_span, steps=int(time_span/dt))
    return odeint(ThreeBody(), y, t, method=method)


