'''
CDPSloss.Py

Implementation of CDPS loss as described in the paper
Amouri, Hussein El, et al. "CDPS: Constrained DTW-Preserving Shapelets."
Joint European Conference on Machine Learning and Knowledge Discovery in
Databases. Cham: Springer International Publishing, 2022.
'''

import numpy
import torch
import torch.nn as nn


class CDPSloss(nn.Module):
    """
    Proposed loss based on DTW approximation and Contrastive Learning

    Args:
        nn ([type]): [description]
    """

    def __init__(self,
                 dtw_max=20,
                 gamma=1.,
                 alpha=1.,
                 fr=None,
                 saveloss=False,
                 device='cpu',
                 scaled=1,
                 ):
        super(CDPSloss, self).__init__()

        self.device = device

        self.fr = fr
        self.gamma = gamma
        self.alpha = alpha


        self.scaled = scaled
        self.dtw_max = dtw_max

        self.saveloss = saveloss
        self.lossmltrack = []
        self.losscltrack = []

        self.ml, self.cl = numpy.array([]), numpy.array([])

    def constraintmatrices(self, ml, cl, alpha=None, gamma=None):
        """
        Get constraints matrices mask
        Args:
            ml (ndarray): list of must-link constraints
            cl (ndarray): list of cannot-link constraints
            alpha (ndarry optional): ml regularize.
                                    Defaults to None.
            gamma (ndarry, optional): cl regularize.
                                    Defaults to None.
        """

        self.ml = ml
        self.cl = cl
        if alpha is not None:
            self.alpha = torch.tensor(alpha)
        if gamma is not None:
            self.gamma = torch.tensor(gamma)

    def _mij(self, DTW):
        """
        setting mij
        """

        return self.dtw_max + torch.log(DTW / self.dtw_max)

    def forward(self, dtwhat, DTW, citer=None):
        """
        calculating loss
        """

        zero_ = torch.tensor(0, device=self.device)
        ls = nn.MSELoss()
        if self.fr is not None:
            loss = 0.0
            lnc = ls(dtwhat[self.scaled], DTW)
            dtwt_hat_ij = self.ml * dtwhat[0]
            lml = torch.mean(self.alpha * torch.pow(dtwt_hat_ij, 2))
            m_ij = self._mij(self.cl * DTW)
            dtwt_hat_ij = self.cl * dtwhat[0]
            diff = torch.nan_to_num(m_ij, neginf=0.0, posinf=0.0) - dtwt_hat_ij
            lcl = torch.mean(self.gamma * torch.pow(torch.max(zero_, diff), 2))
            loss = lnc + lml + lcl
            if self.saveloss:
                self.lossmltrack.append(lml.cpu().detach().numpy())
                self.losscltrack.append(lcl.cpu().detach().numpy())
        else:
            loss = ls(dtwhat[self.scaled], DTW)
        return loss
