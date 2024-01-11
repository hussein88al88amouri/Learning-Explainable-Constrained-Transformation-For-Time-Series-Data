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
from minpool1D import MinPool1d
from utls import generate_random_indices, compute_dtw_indep_dep
from utls import _kmeans_init_shapelets, tslearn2torch
from ShapeletLayer import ShapeletLayer
from CDPSloss import CDPSloss

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
                 dtw_max=25,
                 constraints_in_batch=4,
                 device='cpu',
                 type_='INDEP',
                 verbose=True,
                 saveloss=False,
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
        self.type_ = type_
        self.savecheckpoint = False
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
