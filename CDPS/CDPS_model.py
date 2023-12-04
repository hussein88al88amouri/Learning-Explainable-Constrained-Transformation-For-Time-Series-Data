"""

CDPS_model.py

This module implements the "Constrained DTW Preserving Shapelets: CDPS" model,
as presented in the ECML Conference 2022.

For more details, refer to the original article.

The CDPS model aims to learn a shapelet transform for time series data,
incorporating two key factors:
- Expert knowledge, represented as must and cannot-link constraints
- Distortions, addressed by approximating the DTW pseudo-distance metric

Usage:
1. Import the module: `import CDPS_model`
2. Instantiate the model: `model = CDPS_model.CDPSModel(param1, param2, etc)`
3. Train the model: `model.fit(training_data)`
4. Make predictions: `predictions = model.predict(test_data)`

Example:
```python
import CDPS_model

# Instantiate the model
model = CDPS_model.CDPSModel(param1=1, param2=2)

# Train the model
training_data = [...]  # Your training data here
model.fit(training_data)

# Make predictions
test_data = [...]  # Your test data here
predictions = model.predict(test_data)

# The following code is inspired by rtavenar's work

"""
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy
from torch.autograd import Variable
from sklearn.linear_model import LinearRegression
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw


def generate_random_indices(nsamples, batch_size, idx_exclude):
    """
    Generate Random indices for the input batch.
    The indices will be gerenated such that they are not repeated in A.
    Args:
        nsamples: total number of samples to generate indices for
        batchsize: size of the batch size
        idx_exclude: set of indices not to choose from
        (These indices will be found in ml or cl sets)
    """

    if nsamples < batch_size:
        raise ValueError("nsamples must >= batch_size.")

    if batch_size > nsamples:
        raise ValueError("batch_size must <= nsamples.")

    size = [batch_size, 2]
    random_indices = numpy.random.choice(
        nsamples, size=size, replace=False)
    f_duplicate = any(numpy.any((random_indices[:, None, :] ==
                                 idx_exclude).all(axis=-1), axis=-1))
    while f_duplicate:
        random_indices = numpy.random.choice(
            nsamples, size=size, replace=False)
        f_duplicate = any(numpy.any((random_indices[:, None, :] ==
                                     idx_exclude).all(axis=-1), axis=-1))
    return random_indices


def compute_dtw_indep_dep(time_series1, time_series2, type_='DEP'):
    """
    Calculate DTW either in depenedent or independet approach.
    check the article:
    Args:
        time_series1: multidimensional time series
        time_series2: multidimensional time series
        type_: "DEP" --> dependent approach
            "INDEP" --> indepent appraoch
    """

    if type_ == 'DEP':
        return dtw(time_series1, time_series2)
    elif type_ == 'INDEP':
        dtwi = 0
        for dim in range(time_series1.shape[-1]):
            dtwi += dtw(time_series1[:, dim], time_series2[:, dim])
        return dtwi
    else:
        raise ValueError("Unsupported value for 'type_'. "
                         "Choose either 'INDEP' or 'DEP'.")


def _kmeans_init_shapelets(x_input, n_shapelets, shp_len, n_draw=10000):
    """
    intialize the shapelets layer weights using k-means cluster
    centers as shapelets.

    Args:
        x_input (ndarry): _description_
        n_shapelets (int): _description_
        shp_len (int): _description_
        n_draw (int, optional): _description_. Defaults to 10000.

    Returns:
        _type_: _description_
    """

    n_ts, sz, d = x_input.shape
    indices_ts = numpy.random.choice(n_ts, size=n_draw, replace=True)
    indices_time = numpy.random.choice(sz - shp_len + 1,
                                       size=n_draw,
                                       replace=True)
    subseries = numpy.zeros((n_draw, shp_len, d))
    for i in range(n_draw):
        t_idx1 = indices_time[i]
        t_idx2 = indices_time[i] + shp_len
        subseries[i] = x_input[indices_ts[i], t_idx1: t_idx2]
    return TimeSeriesKMeans(n_clusters=n_shapelets,
                            metric="euclidean",
                            verbose=False).fit(subseries).cluster_centers_


def tslearn2torch(x_input, device="cpu"):
    """
    numpy array to a torch tensor
    Args:
        x_input (ndarry): input array representing a time series
        device (str, optional): load tensor on "cuda' or "cpu".
    Returns:
        torch.tensor: tensor
    """
    x_tensor = torch.Tensor(numpy.transpose(x_input, (0, 2, 1))).to(device)
    return x_tensor


class MinPool1d(nn.Module):
    """
    Simple Hack for 1D min pooling. Input size = (N, C, L_in)
    Output size = (N, C, L_out) where N = Batch Size, C = No. Channels
    L_in = size of 1D channel, L_out = output size after pooling.
    This implementation does not support custom strides, padding or dilation
    Input shape compatibilty by kernel_size needs to be ensured

    This code comes from:
    https://github.com/reachtarunhere/pytorch-snippets/blob/master/min_pool1d.py
    (under MIT license)
    """

    def __init__(self, kernel_size=3, type_='DEP'):
        super(MinPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.type_ = type_

    def forward(self, x_input):
        """
        minpooling, either dependent or independent of time axis
        Args:
            x_input (tensor): input batch to perform minpoolin on
        """
        _, d, _, _ = [x_input.size(i) for i in range(4)]
        if self.type_ == 'INDEP':
            x_input = torch.stack([x_input[:, i, :, :].min(dim=2)[0]
                                  for i in range(d)])
            return x_input.transpose(1, 0).sum(1)
        elif self.type_ == "DEP":
            x_input = torch.sum(x_input, 1)
            return x_input.min(dim=2)[0]
        else:
            raise ValueError("Unsupported value for 'type_'. "
                             "Choose either 'INDEP' or 'DEP'.")


class ShapeletLayer(nn.Module):
    """
    Shapelet layer.
    [i, 0] == pairs[i, 1]:
    pairs[i, 1] = np.random.choice(Trdata.shape[0])
    Computes sliding window distances between a set of time series and a set
    of shapelets.

    Parameters
    ----------
    in_channels : int
        Number of input channels (modalities of the time series)
    out_channels: int
        Number of output channels (number of shapelets)
    kernel_size: int
        Shapelet length

    Examples
    --------
    >>> time_series = torch.Tensor([[1., 2., 3., 4., 5.], [-4., -5., -6., -7., -8.]]).view(2, 1, 5)
    >>> shapelets = torch.Tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> layer = ShapeletLayer(in_channels=1, out_channels=3, kernel_size=2)
    >>> layer.weight.data = shapelets
    >>> dists = layer.forward(time_series)
    >>> dists.shape
    torch.Size([2, 3, 4])
    >>> dists[0]
    tensor([[ 0.,  1.,  4.,  9.],
            [ 4.,  1.,  0.,  1.],
            [16.,  9.,  4.,  1.]], grad_fn=<SelectBackward>)
    """

    def __init__(self, in_channels, out_channels, kernel_size, device, type_):
        super(ShapeletLayer, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.type_ = type_
        self.false_conv_layer = nn.Conv1d(in_channels=1,
                                          out_channels=1,
                                          kernel_size=kernel_size,
                                          groups=1,
                                          bias=False)

        data = torch.Tensor(numpy.eye(kernel_size)).to(device)
        data_view = data.view(kernel_size, 1, kernel_size)
        self.false_conv_layer.weight.data = data_view
        for p in self.false_conv_layer.parameters():
            p.requires_grad = False
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
                                                kernel_size).to(device))

    def forward(self, x_input):
        """
        calculate the distance between the shapelet candidates and
        the batch time series.

        Args:
            x_input (tensor): batch of time series

        Returns:
            tensor : distance between all possible subsequence
                    and candidate shapelets
        """
        n, _, d = x_input.shape
        reshaped_x = torch.stack([self.false_conv_layer(
                                x_input[:, i, :].view(n, 1, d))
                                    for i in range(self.in_channels)])
        reshaped_x = torch.transpose(torch.transpose(reshaped_x, 0, 1), 2, 3)
        distances = torch.stack([self.pairwise_distances(
                                reshaped_x[:, i, :, :].contiguous().view(
                                   -1, self.kernel_size), self.weight[:, i, :])
                                        for i in range(self.in_channels)])
        distances = distances.view(self.in_channels, n, -1,
                                   self.out_channels).transpose(1, 0)
        return torch.transpose(distances, 2, 3)

    @classmethod
    def pairwise_distances(cls, x, y):
        """Computes pairwise distances between vectors in x and those in y.

        Computed distances are normalized (i.e. divided) by the dimension of
        the space in which vectors lie.
        Assumes x is 2d (n, d) and y is 2d (l, d) and returns
        a tensor of shape (n, l).

        Parameters
        ----------
        x : Tensor of shape=(n, d)
        y : Tensor of shape=(l, d)
        import time
        Returns
        -------
            A 2d Tensor of shape (n, l)
        """
        len_ = y.size(-1)
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, numpy.inf) / len_


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
                 period=5,
                 fr=None,
                 saveloss=False,
                 partitioned_loss=False,
                 device='cpu',
                 scaled=1,
                 ):
        super(CDPSloss, self).__init__()

        self.device = device

        self.fr = fr
        self.gamma = gamma
        self.alpha = alpha
        self.k = 1
        self.period = 5
        self.scaled = scaled
        self.dtw_max = dtw_max

        self.lossmltrack = []
        self.losscltrack = []
        self.saveloss = saveloss
        self.period = period
        self.partitioned = partitioned_loss
        self.ml, self.cl = numpy.array([]), numpy.array([])

    def constraintmatrices(self, ml, cl, alpha=None, gamma=None):
        """
        Generate must-link and cannot-link mask matrices
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

    def sched(self, a, i):
        """
        hyperparameter decay
        """

        return (a / self.k)*numpy.exp(-(i %(self.period)))

    def _f(self, DTW):
        """
        setting mij
        """

        return self.dtw_max + torch.log(DTW/self.dtw_max)

    def forward(self, DTWhat, DTW, citer=None): 
        """
        calculating loss
        """

        zero_ = torch.tensor(0, device=self.device)
        ls = nn.MSELoss()
        if self.fr is not None:
            if citer:
                if citer % 50 == 0:
                    self.k += 0.1
                elif citer % 200 == 0:
                    self.k = 1.0

                gamma = self.sched(self.alpha, citer)
                alpha = self.sched(self.gamma, citer)

            else:
                gamma = self.gamma
                alpha = self.alpha
            loss = 0.0
            lnc = ls(DTWhat[self.scaled], DTW)
            dtwt_hat_ij = self.ml*DTWhat[0]
            lml = torch.mean(alpha*torch.pow(dtwt_hat_ij, 2))
            m_ij = self._f(self.cl*DTW)
            dtwt_hat_ij = self.cl*DTWhat[0]
            diff = torch.nan_to_num(m_ij, neginf=0.0, posinf=0.0) - dtwt_hat_ij
            lcl = torch.mean(gamma*torch.pow(torch.max(zero_, diff), 2))
            loss = lnc + lml + lcl
            if self.saveloss:
                self.lossmltrack.append(lml.cpu().detach().numpy())
                self.losscltrack.append(lcl.cpu().detach().numpy())
        else:
            loss = ls(DTWhat[self.scaled], DTW)
        return loss


class CDPSModel(nn.Module):
    """Learning DTW-Preserving Shapelets (LDPS) model.

    Parameters
    ----------
    n_shapelets_per_size: dict (optional, default: None)
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value)
        None should be used only if `load_from_disk` is set
    ts_dim: int (optional, default: None)
        Dimensionality (number of modalities) of the time series considered
        None should be used only if `load_from_disk` is set
    lr: float (optional, default: 0.01)
        Learning rate
    epochs: int (optional, default: 500)
        Number of training epochs
    batch_size: int (optional, default: 64)
        Batch size for training procedure
    verbose: boolean (optional, default: True)
        Should verbose mode be activated

    Note
    ----
        This implementation requires a dataset of equal-sized time series.
    """

    def __init__(self,
                 n_shapelets_per_size=None,
                 ts_dim=1,
                 lr=.01,
                 epochs=500,
                 batch_size=64,
                 ml=numpy.array([]),
                 cl=numpy.array([]),
                 gamma=None,
                 alpha=None,
                 fr=None,
                 period=5,
                 dtw_max=25,
                 constraints_in_batch=4,
                 device='cpu',
                 type_='INDEP',
                 verbose=True,
                 saveloss=False,
                 citer=None,
                 patience=10,
                 min_delta=0,
                 ple=50,
                 scaled=True,
                 ):
        super(CDPSModel, self).__init__()

        self.device = device
        self.n_shapelets_per_size = n_shapelets_per_size
        self.ts_dim = ts_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.constriants_inbatch = constraints_in_batch
        self.verbose = verbose
        self.period = period
        self.fr = fr
        self.ml = ml
        self.cl = cl
        self.constraints_in_batch = constraints_in_batch
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_, self.gamma_ = 0, 0
        self.mlidx_in_target = []
        self.clidx_in_target = []
        self.alpha_in_target = []
        self.gamma_in_target = []

        self.scaled = scaled
        sc = 1 if self.scaled else 0
        self.losstrack = []
        self.loss_ = CDPSloss(gamma=self.gamma, alpha=self.alpha,
                              period=self.period, fr=self.fr,
                              dtw_max=dtw_max, device=self.device,
                              saveloss=saveloss, scaled=sc,)
        self.citer = citer
        self.type_ = type_
        self.savecheckpoint = False
        self.patience = patience
        self.min_delta = min_delta
        self.ple = ple
        self._set_layers_and_optim()

    def _model_save(self, fname):
        """
        """
        torch.save(self, fname)

    @staticmethod
    def _model_load(fname, device='cpu', usepickle=False):
        """
        """
        if usepickle:
            # used for older models that are saved using pickle
            return pickle.load(open(fname, "rb"))
        return torch.load(fname, map_location=device).to(device)

    def _set_layers_and_optim(self):
        self.shapelet_sizes = sorted(self.n_shapelets_per_size.keys())
        self.shapelet_blocks = self._get_shapelet_blocks()
        self.scaling_layer = nn.Linear(1, 1, bias=False, device=self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_shapelet_blocks(self):
        """
        """
        return nn.ModuleList([
            ShapeletLayer(
                in_channels=self.ts_dim,
                out_channels=self.n_shapelets_per_size[shapelet_size],
                kernel_size=shapelet_size,
                device=self.device,
                type_=self.type_
            ) for shapelet_size in self.shapelet_sizes
        ])

    def _temporal_pooling(self, x):
        """
        """

        pool_size = x.size(-1)
        pooled_x = MinPool1d(kernel_size=pool_size, type_=self.type_)(x)
        return pooled_x.view(pooled_x.size(0), -1)

    def _features(self, x):
        """
        shapelet transform
        """

        features_maxpooled = []
        for _, block in zip(self.shapelet_sizes, self.shapelet_blocks):
            f = block(x)
            f_maxpooled = self._temporal_pooling(f)
            features_maxpooled.append(f_maxpooled)
        return torch.cat(features_maxpooled, dim=-1)

    def _init_params(self, x_input):
        """
        Model intialization
        """

        for m in self.shapelet_blocks:
            self._shapelet_initializer(m.weight, x_input)
        # Initialize scaling using linear regression
        pair, targets = self.get_batch(x_input)
        nn.init.constant_(self.scaling_layer.weight, 1.)  # Start without scale
        output, _ = self(pair)
        reg_model = LinearRegression(fit_intercept=False)
        output = output.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        reg_model.fit(output, targets)
        nn.init.constant_(self.scaling_layer.weight, reg_model.coef_[0, 0])

    def _shapelet_initializer(self, w, x_input):
        """
        intialize the shapelets using kmeans centers
        """

        shapelets_npy = _kmeans_init_shapelets(x_input=x_input,
                                               n_shapelets=int(w.size(0)),
                                               shp_len=w.size(-1))
        shapelet_transposed = shapelets_npy.transpose(0, 2, 1)
        w.data = torch.Tensor(shapelet_transposed).to(self.device)

    def forward(self, x_input):
        """
        forward pass
        """

        xi, xj = x_input
        emb_xi = self._features(xi)
        emb_xj = self._features(xj)
        norm_ij = torch.norm(emb_xi - emb_xj, p=2, dim=1, keepdim=True)
        scaled_norm_ij = self.scaling_layer(norm_ij)
        return norm_ij, scaled_norm_ij

    def get_batch(self, x_input):
        """
        input layer
        """
# remove numpy implmentation and directly use torch,
# as for target tensor try to find a pytorch
# implementation of DTW between time series^

        n_samples = x_input.shape[0]
        if self.fr is not None:
            if self.ml.size == 0 and self.cl.size == 0:
                batch_indices = numpy.sort(numpy.random.choice(
                                    n_samples, size=[self.batch_size, 2]))
            else:
                cib = self.constraints_in_batch//2
                f_mlcl = self.ml.shape[0] != 0 and self.cl.shape[0] != 0
                cib = cib if f_mlcl else self.constraints_in_batch

                if self.ml.shape[0] != 0:
                    choice = numpy.random.choice(self.ml.shape[0], cib)
                    ml_idx = self.ml[choice]
                    self.alpha_ = self.alpha[choice]

                if self.cl.shape[0] != 0:
                    choice = numpy.random.choice(self.cl.shape[0], cib)
                    cl_idx = self.cl[choice]
                    self.gamma_ = self.gamma[choice]

                if self.ml.shape[0] == 0:
                    no_con_s = self.batch_size - self.constraints_in_batch
                    batch_indices = generate_random_indices(n_samples,
                                                            no_con_s,
                                                            self.cl)
                    batch_indices = numpy.vstack([batch_indices, cl_idx])
                    self.alpha_ = 0

                elif self.cl.shape[0] == 0:
                    no_con_s = self.batch_size - self.constraints_in_batch
                    batch_indices = generate_random_indices(n_samples,
                                                            no_con_s,
                                                            self.ml)
                    batch_indices = numpy.vstack([batch_indices, ml_idx])
                    self.gamma_ = 0
                else:
                    no_con_s = self.batch_size - self.constraints_in_batch
                    constraints = numpy.concatenate([self.ml, self.cl])
                    batch_indices = generate_random_indices(n_samples,
                                                            no_con_s,
                                                            constraints)
                    batch_indices = numpy.vstack([batch_indices,
                                                  ml_idx,
                                                  cl_idx])
        else:
            self.alpha_, self.gamma_ = 0, 0
            batch_indices = numpy.random.choice(n_samples,
                                                size=[self.batch_size, 2])
            batch_indices = numpy.sort(batch_indices)

        numpy.random.shuffle(batch_indices)

        x1 = Variable(tslearn2torch(x_input[batch_indices[:, 0]],
                                    device=self.device
                                    ).type(torch.float32), requires_grad=False)
        x2 = Variable(tslearn2torch(x_input[batch_indices[:, 1]],
                                    device=self.device
                                    ).type(torch.float32), requires_grad=False)

        targets_tensor = torch.Tensor([
                            compute_dtw_indep_dep(
                                            x1[i].cpu().T, x2[i].cpu().T,
                                            type_=self.type_
                                            )
                            for i in range(self.batch_size)
                                        ]).to(self.device)

        if self.fr is not None:
            mlidx = numpy.array([j for j, row_B in enumerate(batch_indices)
                                 if tuple(row_B) in {tuple(row)
                                                     for row in self.ml}])
            mlidx = mlidx[~numpy.isnan(mlidx)].astype(numpy.int32)
            self.mlidx_in_target = numpy.zeros(self.batch_size)
            self.mlidx_in_target[mlidx] = 1
            self.mlidx_in_target = torch.from_numpy(
                                        self.mlidx_in_target
                                        ).reshape((-1, 1)).to(self.device)
            self.alpha_in_target = numpy.zeros(self.batch_size)
            self.alpha_in_target[mlidx] = self.alpha_

            clidx = numpy.array([j for j, row_B in enumerate(batch_indices)
                                 if tuple(row_B) in {tuple(row)
                                                     for row in self.cl}])
            clidx = clidx[~numpy.isnan(clidx)].astype(numpy.int32)
            self.clidx_in_target = numpy.zeros(self.batch_size)
            self.clidx_in_target[clidx] = 1
            self.clidx_in_target = torch.from_numpy(
                                        self.clidx_in_target
                                        ).reshape((-1, 1)).to(self.device)
            self.gamma_in_target = numpy.zeros(self.batch_size)
            self.gamma_in_target[clidx] = self.gamma_

            self.loss_.constraintmatrices(self.clidx_in_target,
                                          self.clidx_in_target,
                                          self.alpha,
                                          self.gamma)
        else:
            self.loss_.constraintmatrices(None, None)

        targets = Variable(targets_tensor.view(-1, 1), requires_grad=False)

        return (x1, x2), targets

    def fit(self, x_input=None, init_=True):
        """
        Learn shapelets and weights for a given dataset.

        Args:
            x_input (ndarry): Multivariate time series dataset
                shape=(n_ts, sz, d)
            init_ (bool, optional): learn the weights without initialization
                (If the model was already intialized)
        Returns:
                CDPS model: The fitted model
        """

        if init_:
            self._init_params(x_input)
        n_batch_per_epoch = max(x_input.shape[0] // self.batch_size, 1)
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            running_loss = 0.0
            for _ in range(n_batch_per_epoch):
                # get the training batch
                inputs, targets = self.get_batch(x_input)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                preds = self(inputs)
                loss = self.loss_(preds, targets, citer=epoch
                                  if self.citer else None)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            if self.verbose and (epoch + 1) % self.ple == 0:
                pbar.set_postfix({f'Iteration [{epoch + 1}] loss':
                                  f'{(running_loss / n_batch_per_epoch):.3f}'})
            self.losstrack.append(running_loss/n_batch_per_epoch)

        return self
