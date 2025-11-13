import hamiltorch.util as util
import torch
from beartype import beartype
import torch.nn as nn
from hamiltorch import samplers
from torch.func import jacrev, functional_call


@beartype
class VIHMCTrainer:
    def __init__(
            self,
            det_model: nn.Module,
            vi_model: nn.Module,
    ):
        """
        Prepare to train a Bayesian neural network using hybrid VI-HMC approach
        Parameters
        ----------
        det_model : torch.nn.Module
            A deterministic model with the same architecture as the Bayesian model
        vi_model : torch.nn.Module
            Bayesian model trained with variational inference
        """
        self.model = det_model
        self.params_init = util.flatten(self.model).clone()
        self.vi_model = vi_model
        self.mean_params, self.std_params = self._flatten_mean_std()
        self.sens_indices = None

    def _flatten_mean_std(self):
        mean_params = []
        std_params = []
        for name, param in self.vi_model.named_parameters():
            if "mu" in name:
                mean_params.append(param.flatten())
            elif "rho" in name:
                std_params.append(torch.log1p(torch.exp(param)).flatten())
        return torch.cat(mean_params), torch.cat(std_params)

    def eval_sensitivity(self, valid_data, var_threshold):
        """
        Function to evaluate sensitivity scores
        Parameters
        ----------
        valid_data : torch.DataLoader
            Data to compute sensitivity scores
        var_threshold: float
            Threshold for the captured variance
        Returns
        -------
        numpy.typing.NDArray
            Sensitivity scores of the parameters
        """
        params_unflattened = util.unflatten(self.model, self.mean_params)
        cnt = 0
        for param in self.model.parameters():
            param.data = params_unflattened[cnt]
            cnt = cnt + 1
        grads_list = 0
        num_batches = len(valid_data)
        for i, batch_data in enumerate(valid_data):
            *x, y = batch_data
            grads_list += self.eval_jac(x) / num_batches
        assert i + 1 == num_batches
        sensitivities = grads_list * (self.std_params ** 2)
        tot_var = torch.sum(sensitivities)
        cumilative_sum = torch.cumsum(torch.sort(sensitivities, descending=True)[0], 0)
        return sensitivities, torch.sum(cumilative_sum / tot_var <= var_threshold)

    def functional_model(self, w, inputs):
        """
        Functional call of the model
        Parameters
        ----------
        w : list
            parameters of the model
        inputs : torch.Tensor
            input data to evaluate the model

        Returns
        -------
        torch.Tensor
            predictions for the given inputs

        """
        return functional_call(self.model, w, tuple(inputs))

    def eval_jac(self, x):
        """
        Function to evaluate the gradients to compute sensitivity scores
        Parameters
        ----------
        x : torch.Tensor
            input to the network

        Returns
        -------
        torch.Tensor
            Mean of the square of gradients for various inputs
        """

        params_unflattened = util.unflatten(self.model, self.mean_params)
        params_named_list = [n for n, _ in self.model.named_parameters()]
        params_dict = dict(zip(params_named_list, params_unflattened))

        with torch.no_grad():  # prevents memory leak
            jacobian_output_to_params = jacrev(self.functional_model, argnums=0)(
                params_dict, x
            )
            grads = []
            for jac in jacobian_output_to_params.values():
                grads.append(
                    torch.mean(
                        jac ** 2,
                        dim=tuple(range(x[0].ndim)),
                    ).flatten()
                )
        return torch.cat(grads)

    def define_model_log_prob(
            self,
            model_loss,
            tr_data,
            params_flattened_list,
            params_shape_list,
            prior_list,
            tau_out,
            load_prior=False,
            predict=False,
            prior_scale=1.0,
            device="cpu",
    ):
        """
        This function is built on Hamiltorch, and it defines the `log_prob_func` for torch nn.Modules. This will then be passed into the hamiltorch sampler. This is an important
        function for any work with Bayesian neural networks.
        Parameters
        ----------
        model_loss :{'binary_class_linear_output', 'multi_class_linear_output', 'multi_class_log_softmax_output', 'regression', 'NLL'} or function
            This determines the likelihood to be used for the model. The options correspond to:
            * 'binary_class_linear_output': model has linear output and using binary cross entropy,
            * 'multi_class_linear_output': model has linear output and using cross entropy,
            * 'multi_class_log_softmax_output': model has log softmax output and using cross entropy,
            * 'regression': model has linear output and using Gaussian likelihood (variance fixed),
            * 'NLL': Guassian negative log likelihood (variance learnt),
            * function: function of the form func(y_pred, y_true). It should return a vector (N,), where N is the number of data points.
        tr_data : torch.DataLoader
            Training data
        params_flattened_list : list
            A list containing the total number of parameters (weights/biases) per layer in order of the model.
            E.g. `[weights.nelement() for weights in model.parameters()]`.
        params_shape_list : list
            A list describing the shape of each set of parameters in the model.
            E.g. `[weights.shape for weights in model.parameters()]`.
        prior_list : list
            A list containing the corresponding prior precision for each set of per layer parameters. This is assuming a Gaussian prior.
        tau_out : float
            Only relevant for model_loss = 'regression' or 'NLL' (otherwise leave as 1.0). This corresponds the likelihood output precision.
        load_prior : bool
            If true load the prior distribution from saved file
        predict : bool
            Flag to set equal to `True` when used as part of `hamiltorch.predict_model`, otherwise set to False. This controls the number of objects
            to return.
        prior_scale : float
            Most relevant for splitting (otherwise leave as 1.0). The prior is divided by this value.
        device :

        Returns
        -------

        """

        dist_list = []
        if load_prior:
            # for ind in range(prior_list[0].shape[0]):
            #     dist_list.append(torch.distributions.Normal(prior_list[0][ind], prior_list[1][ind]))
            dist_list.append(torch.distributions.Normal(prior_list[0], prior_list[1]))
        else:
            for tau in prior_list:
                dist_list.append(
                    torch.distributions.Normal(torch.zeros_like(tau), tau ** 0.5)
                )

        if model_loss == "NLL":
            nll_loss = torch.nn.GaussianNLLLoss(reduction="sum")

        def log_prob_func(params):

            l_prior = torch.zeros_like(
                params[0], requires_grad=True
            )  # Set l2_reg to be on the same device as params

            if load_prior:
                # for param, dist in zip(params,dist_list):
                l_prior = dist_list[0].log_prob(params).sum() + l_prior
            else:
                i_prev = 0
                for weights, index, shape, dist in zip(
                        self.model.parameters(),
                        params_flattened_list,
                        params_shape_list,
                        dist_list,
                ):
                    # weights.data = params[i_prev:index+i_prev].reshape(shape)
                    w = params[i_prev: index + i_prev]
                    l_prior = dist.log_prob(w).sum() + l_prior
                    i_prev += index

            # Code for fixing insensitive parameters at means
            weights = self.mean_params.clone()
            weights[self.sens_indices] = params
            params_unflattened = util.unflatten(self.model, weights)
            params_named_list = [n for n, _ in self.model.named_parameters()]
            params_dict = dict(zip(params_named_list, params_unflattened))

            output = []
            y = []
            for *x_batch, y_batch in tr_data:
                output.append(self.functional_model(params_dict, tuple(x_batch)))
                y.append(y_batch)

            output = torch.cat(output)
            y = torch.cat(y)
            y_device = y.to(device)
            output = output.reshape(y_device.shape)
            assert output.shape == y_device.shape
            if model_loss == "regression":
                # crit = nn.MSELoss(reduction='mean')
                ll = -0.5 * tau_out * ((output - y_device) ** 2).sum(0)  # sum(0)
                # print(crit(output,y_device))
            elif model_loss == "binary_class_linear_output":
                crit = nn.BCEWithLogitsLoss(reduction="sum")
                ll = -tau_out * (crit(output, y_device))
            elif model_loss == "multi_class_linear_output":
                #         crit = nn.MSELoss(reduction='mean')
                crit = nn.CrossEntropyLoss(reduction="sum")
                #         crit = nn.BCEWithLogitsLoss(reduction='sum')
                ll = -tau_out * (crit(output, y_device.long().view(-1)))
                # ll = - tau_out *(torch.nn.functional.nll_loss(output, y.long().view(-1)))
            elif model_loss == "multi_class_log_softmax_output":
                ll = -tau_out * (
                    torch.nn.functional.nll_loss(output, y_device.long().view(-1))
                )

            elif model_loss == "NLL":
                ll = -nll_loss(output, y_device, tau_out * torch.ones_like(output))

            elif callable(model_loss):
                # Assume defined custom log-likelihood.
                ll = -model_loss(output, y_device).sum(0)
            else:
                raise NotImplementedError()

            if torch.cuda.is_available():
                # del x_device, y_device
                torch.cuda.empty_cache()

            if predict:
                return (ll + l_prior / prior_scale), output
            else:
                return ll + l_prior / prior_scale

        return log_prob_func

    def predict_model(
            self,
            samples,
            test_loader=None,
            model_loss="multi_class_linear_output",
            tau_out=1.0,
            prior_list=None,
    ):
        """This function is taken from the Hamiltorch library and modified for DeepONets as necessary. Function used to make predictions given model samples. Note that either a data loader can be passed in, or two tensors (x,y) but make sure
        not to pass in both.

        Parameters
        ----------
        samples : list of torch.Tensor
            A list, where each element is a torch.Tensor of shape (D,), where D is the number of parameters of the model.
            The length of the list is given by the number of samples, S.
        test_loader : torch.utils.data.Dataloader, optional
            Data loader to be used for evaluating the samples. This can be set to `None` if `x` and `y` are defined.
        model_loss : {'binary_class_linear_output', 'multi_class_linear_output', 'multi_class_log_softmax_output', 'regression'} or function
            This determines the likelihood to be used for the model. The options correspond to:
            * 'binary_class_linear_output': model has linear output and using binary cross entropy,
            * 'multi_class_linear_output': model has linear output and using cross entropy,
            * 'multi_class_log_softmax_output': model has log softmax output and using cross entropy,
            * 'regression': model has linear output and using Gaussian likelihood,
            * function: function of the form func(y_pred, y_true). It should return a vector (N,), where N is the number of data points.
        tau_out : float
            Only relevant for model_loss = 'regression' (otherwise leave as 1.0). This corresponds the likelihood output precision.
        prior_list : torch.Tensor
            A tensor containing the corresponding prior precision for each set of per layer parameters. This is assuming a Gaussian prior.

        Returns
        -------
        predictions : torch.tensor
            Output of the model of shape (S,N,O), where S is the number of samples, N is the number of data points, and O is the output shape of the model.
        pred_log_prob_list : list
            List of log probability values for each sample. The length of the list is S.

        """
        with torch.no_grad():
            params_shape_list = []
            params_flattened_list = []
            build_tau = False
            if prior_list is None:
                prior_list = []
                build_tau = True
            for weights in self.model.parameters():
                params_shape_list.append(weights.shape)
                params_flattened_list.append(weights.nelement())
                if build_tau:
                    prior_list.append(torch.tensor(1.0))

            log_prob_func = self.define_model_log_prob(
                model_loss,
                test_loader,
                params_flattened_list,
                params_shape_list,
                prior_list,
                tau_out,
                predict=True,
                device=samples[0].device,
            )

            pred_log_prob_list = []
            pred_list = []
            for s in samples:
                lp, pred = log_prob_func(s)
                pred_log_prob_list.append(
                    lp.detach()
                )  # Side effect is to update weights to be s
                pred_list.append(pred.detach())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.stack(pred_list), pred_log_prob_list

    def run(
            self,
            train_data: torch.utils.data.DataLoader,
            valid_data: torch.utils.data.DataLoader,
            variance_threshold: float = 0.9,
            num_samples: int = 1000,
            num_steps: int = 30,
            step_size: float = 1e-4,
            burn: int = 100,
            loss: str = "NLL",
            tau_out: float = 1.0,
            prior_var: float = 1.0,
            load_prior: bool = False,
            init_prior: bool = False,
            sample_prior: bool = False,
            prior_file: str = None,
            device: str = "cpu",
            debug: bool = False,
    ):
        """
        run the VI-HMC algorithm to sample from the posterior distribution of parameters.
        Parameters
        ----------
        train_data : torch.Dataloader
            Data used to compute the log likelihood in HMC
        valid_data : torch.Dataloader
            Data used to validate the model performance
        variance_threshold : float
            Threshold to define the captured variance in VI-HMC algorithm. This threshold determines
            the number of sensitive parameters.
        num_samples : int
            Number of samples to draw using the VI-HMC method
        num_steps : float
            Number of steps to take per trajectory
        step_size : float
            Size of each step taken in the numerical integration
        burn : int
            Number of samples to burn before collecting samples.
        loss : {'binary_class_linear_output', 'multi_class_linear_output', 'multi_class_log_softmax_output',
                'regression', 'NLL'} or function
            This determines the likelihood to be used for the model. The options correspond to:
            * 'binary_class_linear_output': model has linear output and using binary cross entropy,
            * 'multi_class_linear_output': model has linear output and using cross entropy,
            * 'multi_class_log_softmax_output': model has log softmax output and using cross entropy,
            * 'regression': model has linear output and using Gaussian likelihood (variance fixed),
            * 'NLL': Guassian negative log likelihood (variance learnt),
            * function: function of the form func(y_pred, y_true). It should return a vector (N,), where N is the number
                of data points.
        tau_out : float
            Only relevant for model_loss = 'regression' or 'NLL' (otherwise leave as 1.0). This corresponds the likelihood
            output precision. 1/variance of likelihood if Regression or variance of likelihood if NLL.
        prior_var : float
            variance of the prior distribution
        load_prior : bool
            If true load the prior distribution from saved file
        init_prior : bool
            If true initialize the HMC chain using the prior information
        sample_prior : bool
            If true initialize the HMC chains at samples taken from the prior distribution. If false initialize the HMC
            chains at the mean of the prior distribution. ``init_prior`` should be true for ``sample_prior`` to take
            effect.
        prior_file : str
            Location of the prior file
        device : name of device, or {'gpu', 'cpu'}
            The device to run on
        debug : bool
            If True HMC runs in the debug mode.

        Returns
        -------
        params_hmc: list of torch.Tensor(s)
            List of parameters samples for the sensitive parameters
        pred_list: list
            List of predictions for the validation data for each of the samples
        log_prob_list: list
            List of log probability values for each sample.
        """
        sensitivity_scores, num_params = self.eval_sensitivity(
            valid_data, var_threshold=variance_threshold
        )
        self.sens_indices = torch.argsort(sensitivity_scores, descending=True)[
                            :num_params
                            ].sort()[0]
        print("=============================================================")
        print("Sensitivity analysis results")
        print("-------------------------------------------------------------")
        print("No of total parameters: ", len(self.mean_params))
        print("No of sensitive parameters: ", num_params.detach().numpy())
        print("=============================================================")
        print("VI-HMC results")
        print("-------------------------------------------------------------")
        params_shape_list = []
        params_flattened_list = []
        prior_list = []
        if load_prior:
            build_tau = False
            mean_params = torch.load(
                f"{prior_file}/means_flattened", map_location=device
            )
            std_params = torch.load(f"{prior_file}/stds_flattened", map_location=device)
            prior_list = [mean_params[self.sens_indices], std_params[self.sens_indices]]
        else:
            build_tau = True
        for weights in self.model.parameters():
            params_shape_list.append(weights.shape)
            params_flattened_list.append(weights.nelement())
            if build_tau:
                prior_list.append(torch.tensor(prior_var))

        log_prob_func = self.define_model_log_prob(
            loss,
            train_data,
            params_flattened_list,
            params_shape_list,
            prior_list,
            tau_out,
            device=device,
        )

        if init_prior:
            learned_mus = torch.load(
                f"{prior_file}/means_flattened", map_location=device
            )
            learned_sigmas = torch.load(
                f"{prior_file}/stds_flattened", map_location=device
            )
            params_trained = (
                torch.normal(learned_mus, learned_sigmas)
                if sample_prior
                else learned_mus
            )
        else:
            params_trained = self.params_init

        params_init = params_trained[self.sens_indices].clone()
        params_hmc = samplers.sample(
            log_prob_func,
            params_init,
            num_samples=num_samples,
            num_steps_per_sample=num_steps,
            step_size=step_size,
            debug=debug,
            sampler=samplers.Sampler.HMC,
        )
        pred_list, log_prob_list = self.predict_model(
            params_hmc[burn:],
            valid_data,
            model_loss=loss,
            tau_out=tau_out,
            prior_list=prior_list,
        )
        return params_hmc, pred_list, log_prob_list
