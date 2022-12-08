from helmnet.source_module import SourceModule
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import hardtanh
from random import choice
import pytorch_lightning as pl
from torchmetrics.regression import MeanAbsoluteError
import numpy as np
from helmnet.architectures import HybridNet
from helmnet.dataloaders import get_dataset
from helmnet.spectral import LaplacianWithPML, FastLaplacianWithPML
from helmnet.replaybuffer import ReplayBuffer, Experience
from torch.optim.lr_scheduler import ReduceLROnPlateau




class IterativeSolver(pl.LightningModule):
    def __init__(
        self,
        domain_size             : int,
        k                       : float,
        omega                   : float,
        PMLsize                 : int,
        sigma_max               : float,
        source_location         : list,
        train_data_path         : str,
        validation_data_path    : str,
        test_data_path          : str,
        activation_function     : str   = "relu",
        architecture            : str   = "custom_unet",
        gradient_clip_val       : int   = 0,
        batch_size              : int   = 24,
        buffer_size             : int   = 1000,
        depth                   : int   = 4,
        features                : int   = 8,
        learning_rate           : float = 1e-4,
        loss                    : str   = "mse",
        minimum_learning_rate   : float = 1e-4,
        optimizer               : str   = "adam",
        weight_decay            : float = 0.0,
        max_iterations          : int   = 100,
        source_amplitude        : int   = 10,
        source_phase            : int   = 0,
        source_smoothing        : bool  = False,
        state_channels          : int   = 2,
        state_depth             : int   = 4,
        unrolling_steps         : int   = 10
    ):
        super().__init__()

        # Saving hyperparameters
        self.save_hyperparameters()

        # Derived modules
        self.replaybuffer   = ReplayBuffer(self.hparams.buffer_size)
        self.metric         = MeanAbsoluteError()

        self.register_buffer('sigmas', None)    # buffer to transition tensor device with module
        
        self.set_laplacian()
        self.setup_source()     # source is now a module

        # Non linear function approximator
        self.init_f()

        # Custom weight initialization
        #  TODO: Add this to the settings file
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=0.02)
                # torch.nn.init.zeros_(m.bias)

        self.f.apply(weights_init)

    def init_f(self):
        nn_name = self.hparams.architecture
        if nn_name == "custom_unet":
            self.f = HybridNet(
                activation_function = self.hparams.activation_function,
                depth               = self.hparams.depth,
                domain_size         = self.hparams.domain_size,
                features            = self.hparams.features,
                inchannels          = 6,
                state_channels      = self.hparams.state_channels,
                state_depth         = self.hparams.state_depth,
            )
        else:
            raise NotImplementedError("Unknown architecture {}".format(nn_name))

    def set_domain_size(self, domain_size, source_location = None, source_map = None):
        self.hparams.domain_size = domain_size
        self.f.domain_size = self.hparams.domain_size
        self.set_laplacian()
        self.setup_source()

        #pytorch lightning really does not like this
        self.Lap.to(self.device)
        self.source_module.to(self.device)

        if source_location is not None:
            self.set_multiple_sources([source_location])
        else:
            self.set_source_maps(source_map)
        self.f.init_by_size()
        for enc, size in zip(self.f.enc, self.f.states_dimension):
            enc.domain_size = size

    def set_laplacian(self):
        """
        self.Lap = LaplacianWithPML(
            domain_size=self.hparams.domain_size,
            PMLsize=self.hparams.PMLsize,
            k=self.hparams.k,
            sigma_max=self.hparams.sigma_max,
        ).to(self.device)
        """
        self.Lap = FastLaplacianWithPML(
            domain_size = self.hparams.domain_size,
            PMLsize     = self.hparams.PMLsize,
            k           = self.hparams.k,
            sigma_max   = self.hparams.sigma_max,
        )#.to(self.device)      #by the time of initialization the self.device variable is still cpu

        sigmax, sigmay = self.Lap.sigmas()
        sigmax = sigmax.clone().detach().to(self.device) #torch.tensor(sigmax, device=self.device)
        sigmay = sigmay.clone().detach().to(self.device) #torch.tensor(sigmay, device=self.device)
        sigmax = sigmax.unsqueeze(0)
        sigmay = sigmay.unsqueeze(0)
        self.sigmas = torch.cat([sigmax, sigmay]).float()   # buffer will be moved together with the module

    def setup_source(self):
        self.source_module = SourceModule(
            image_size  = self.hparams.domain_size,
            omega       = self.hparams.omega,
            location    = self.hparams.source_location,
            amplitude   = self.hparams.source_amplitude,
            phase       = self.hparams.source_phase,
            smooth      = self.hparams.source_smoothing
        )
        with torch.no_grad():
            self.set_source()

    def set_source_maps(self, sourceval):
        self.source = nn.Parameter(
            sourceval,
            requires_grad = False,
        )

    def set_source(self):
        sourceval = self.source_module.spatial_map(0).permute(0, 3, 1, 2)
        self.set_source_maps(sourceval)

    def reset_source(self):
        with torch.no_grad():
            if not self.source_module.get_location() == self.hparams.source_location:
                self.source_module.set_new_location(self.hparams.source_location)
                self.set_source()

    def set_multiple_sources(self, source_locations):
        sourceval_array = []
        with torch.no_grad():
            for loc in source_locations:
                self.source_module.set_new_location(loc)
                sourceval_array.append(
                    self.source_module.spatial_map(0).permute(0, 3, 1, 2)
                )
            sourceval = torch.cat(sourceval_array, 0)
            self.set_source_maps(sourceval)

    def on_after_backward(self):
        if self.hparams.gradient_clip_val > 0:
            torch.nn.utils.clip_grad.clip_grad_value_(
                self.parameters(), self.hparams.gradient_clip_val
            )

    def get_random_source_loc(self):
        """Random source location on a circle"""
        # TODO: Make it more flexible, this is basically hard coded...
        theta = torch.tensor(2 * np.pi * np.random.rand(1), device=self.device)
        L = self.hparams.domain_size // 2
        dL = L - self.hparams.PMLsize - 2
        # source_location = np.array(
        #     [int(L + dL * np.cos(theta)), int(L + dL * np.sin(theta))]
        # )
        source_location = torch.tensor(
            [int(L + dL * torch.cos(theta)), int(L + dL * torch.sin(theta))], device=self.device
        )
        return source_location

    def train_dataloader(self):
        # Making dataset of SoS
        sos_train = get_dataset(self.hparams.train_data_path)

        # Filling up experience replay
        print("Filling up Replay buffer...")

        with torch.no_grad():
            for counter in range(len(self.replaybuffer)):
                self.reset_source()  # self.set_multiple_sources([self.get_random_source_loc()])
                #sos_map = sos_train[counter].unsqueeze(0).to(self.device)  # [1, 1, 96, 96]
                sos_map = sos_train[counter].unsqueeze(0)
                sos_map = sos_map.type_as(self.source)
                k_sq, wavefield = self.get_initials(sos_map)               # ( [1, 1, 96, 96], [1, 2, 96, 96] )
                self.f.clear_states(wavefield)                             
                h_states = self.f.get_states(flatten=True)                 # [1, 2, 96x96xN]
                residual = self.get_residual(wavefield, k_sq)              # [1, 2, 96x96]
                exp = Experience(
                    wavefield[0],
                    h_states[0],
                    k_sq[0],
                    residual[0],
                    self.source[0],
                    counter * 10,
                )

                self.replaybuffer.append(exp, counter)

        # Return the dataloader of sos maps
        return DataLoader(
            sos_train,
            batch_size=self.hparams.batch_size,
            num_workers=min([self.hparams.batch_size, 32]),
            drop_last=True,
        )

    def val_dataloader(self):
        # Making dataset of SoS
        self.reset_source()
        sos_train = get_dataset(self.hparams.validation_data_path)
        # Return the dataloader of sos maps
        return DataLoader(
            sos_train,
            batch_size=self.hparams.batch_size,
            num_workers=min([self.hparams.batch_size, 32]),
        )

    def test_dataloader(self):
        self.reset_source()
        testset = get_dataset(self.hparams.test_data_path)
        # Return the dataloader of sos maps
        return DataLoader(
            testset,
            batch_size=self.hparams.batch_size,
            num_workers=min([self.hparams.batch_size, 32]),
            shuffle = True
        )

    def configure_optimizers(self):
        # TODO: Add adam betast to settings file
        if self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise NotImplementedError(
                "The optimizer {} is not implemented".format(self.hparams.optimizer)
            )

        if self.hparams.minimum_learning_rate > self.hparams.learning_rate:
            raise ValueError(
                "Minimum learning rate ({}) must be smaller than the starting learning rate ({})".format(
                    self.hparams.minimum_learning_rate, self.hparams.learning_rate
                )
            )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                min_lr=self.hparams.minimum_learning_rate,
                verbose=True,
            ),
            "monitor": "train_loss_mean",  # Default: val_loss
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def loss_function(self, x):
        if self.hparams.loss == "mse":
            #without the dimension specifier we now average across all samples in a batch instead of per sample, should be ok as long as mean(sqrt(mean(...))) ~= sqrt(mean(mean()))
            #advantage of this is that we can avoid dimension mismatching with nan results later
            return x.pow(2).mean()      
        else:
            raise NotImplementedError(
                "The loss function {} is not implemented".format(self.hparams.loss)
            )

    @staticmethod
    def test_loss_function(x):
        return x.pow(2).mean((1, 2, 3)).sqrt()

    def test_step(self, batch, batch_idx):
        self.reset_source()
        output = self.forward(
            batch,
            num_iterations=self.hparams.max_iterations,
            return_wavefields=True,
            return_states=False,
        )
        # Get loss
        losses = [self.test_loss_function(x) for x in output["residuals"]]
        losses = torch.stack(losses, 1)
        return {
            "losses": losses,
            #"wavefields": [x.cpu() for x in output["wavefields"]],         #calls to cpu seems to be unnecessary here
            "wavefields": [x for x in output["wavefields"]],      
        }

    def test_epoch_end(self, outputs):
        # Saving average losses
        print("Saving residual RMSE")
        x = []
        for o in outputs:
            x.append(o["losses"])
        all_losses = torch.cat(x, dim = 0).cpu().numpy()
        np.save("results/evolution_of_model_RMSE_on_test_set", all_losses)

        # Save wavefield
        print("Saving wavefields")
        wavefields = torch.cat(
            [torch.stack(x["wavefields"], 0) for x in outputs], 1
        ).permute(1, 0, 2, 3, 4)
        np.save("results/evolution_of_wavefields_on_test_set", wavefields.cpu().numpy())
        

    def validation_step(self, batch, batch_idx):
        self.set_multiple_sources(
            [self.get_random_source_loc() for _ in range(batch.shape[0])]
        )
        output = self.forward(
            batch,
            num_iterations=self.hparams.max_iterations,
            return_wavefields=False,
            return_states=False,
        )
        # Get loss
        loss = self.loss_function(output["residuals"][-1]).sqrt()
        # NaNs to Infs, due to Lightning bug: https://github.com/PyTorchLightning/pytorch-lightning/issues/2636
        loss[torch.isnan(loss)] = float("inf")
        sample_wavefield = (hardtanh(output["wavefields"][0][0]) + 1) / 2
        return {
            "loss": loss,
            "sample_wavefield": sample_wavefield,
            "batch_idx": batch_idx,
        }

    def validation_epoch_end(self, outputs):
        all_losses = torch.stack([x["loss"] for x in outputs]).mean()
        val_loss_mean = self.metric(all_losses, torch.zeros_like(all_losses))

        self.reset_source()
        self.logger.experiment.add_images(
            "wavefield/val_real",
            outputs[0]["sample_wavefield"][0],
            self.trainer.global_step,
            dataformats="HW",
        )
        self.logger.experiment.add_image(
            "wavefield/val_imag",
            outputs[0]["sample_wavefield"][1],
            self.trainer.global_step,
            dataformats="HW",
        )

        self.log('val_loss', val_loss_mean)
        self.log('val_terminal_loss', val_loss_mean)
        return {
            "val_loss": val_loss_mean
        }


    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        #training_epoch_end cannot return values anymore
        self.log('train_loss_mean', train_loss_mean)
        #return {"train_loss": train_loss_mean}

    def training_step(self, sos_batch, batch_idx):
        # Training phase
        maxiter = min([self.current_epoch * 20 + 1, self.hparams.max_iterations])
        # Sample from the buffer
        (
            wavefields,
            h_states,
            k_sqs,
            residual,
            sources,
            timesteps,
            indices,
        ) = self.replaybuffer.sample(self.hparams.batch_size)

        # Set the states and sources
        self.set_source_maps(sources)
        self.f.set_states(h_states, flatten=True)

        # Make N steps
        num_iterations = self.hparams.unrolling_steps       #10


        output = self.n_steps(wavefields, k_sqs, residual, num_iterations, True, True)

        # Evaluate the loss function (will backward later)
        cat_res = torch.cat(output["residuals"])

        # stack_res = torch.stack(output["residuals"])
        loss_f = cat_res.pow(2)
        loss = 1e4 * loss_f.mean()  # TODO: Use settings loss and amplify
        rel_loss_f = loss_f.mean((1, 2, 3)).sqrt().mean()
        self.logger.experiment.add_scalar(
            "loss/train", rel_loss_f, self.trainer.global_step
        )

        #Add histogram of iteration lengths
        if self.trainer.current_epoch // 50 == 0:
            self.logger.experiment.add_histogram(
                "hyper/iterations", np.array(list(timesteps)), self.trainer.global_step
            )

        # Making detached clones
        wavefields = [x.detach() for x in output["wavefields"]]
        h_states = [x.detach() for x in output["states"]]
        k_sqs = [k_sqs for x in output["wavefields"]]
        residuals = [x.detach() for x in output["residuals"]]
        sources = [x.detach() for x in self.source]

        # Adding to RB if iterations are not more than allowed
        counter = 0
        terminal_logged = False
        middle_logged = False
        iteration = np.random.choice(len(residuals))
        for sample_idx in range(self.hparams.batch_size):
            new_timesteps = timesteps[sample_idx] + iteration + 1
            res = residuals[iteration][sample_idx]
            if res.pow(2).mean() < 1 and new_timesteps < maxiter:
                self.replaybuffer.append(
                    Experience(
                        wavefields[iteration][sample_idx],
                        h_states[iteration][sample_idx],
                        k_sqs[iteration][sample_idx],
                        residuals[iteration][sample_idx],
                        sources[sample_idx],
                        new_timesteps,
                    ),
                    indices[sample_idx],
                )
            else:
                with torch.no_grad():
                    self.reset_source()
                    ksq, wf = self.get_initials(choice(sos_batch).unsqueeze(0))
                    self.f.clear_states(wf)
                    h = self.f.get_states(flatten=True)
                    res = self.get_residual(wf, ksq)
                    self.replaybuffer.append(
                        Experience(wf[0], h[0], ksq[0], res[0], self.source[0], 0),
                        indices[sample_idx],
                    )
                    counter += 1

            # Log it as wavefield at 20 steps
            if not middle_logged and new_timesteps == 20:
                self.log_wavefield(wavefields[iteration][sample_idx], "20")
                with torch.no_grad():
                    middle_loss = self.loss_function(residuals[iteration][sample_idx])
                    self.logger.experiment.add_scalar(
                        "loss/step_20",
                        middle_loss.sqrt().item(),
                        self.trainer.global_step,
                    )
                middle_logged = True

            # Log terminal wavefield
            elif new_timesteps >= maxiter and not terminal_logged:
                self.log_wavefield(wavefields[iteration][sample_idx], "terminal")
                with torch.no_grad():
                    terminal_loss = self.loss_function(residuals[iteration][sample_idx])
                    self.logger.experiment.add_scalar(
                        "loss/terminal",
                        terminal_loss.sqrt().item(),
                        self.trainer.global_step,
                    )
                    terminal_logged = True

        self.logger.experiment.add_scalar(
            "train_loss",
            loss,
            self.trainer.global_step
        )

        self.log('train_loss', loss) # force the variable to be saved in state dict of module. tensorboard logger does not do this for some reason

        #need to log with prog_bar = True to show on progress bar now
        self.log('maxiter',     maxiter, on_epoch=True, prog_bar = True)
        self.log('unrolling',   num_iterations, on_epoch=True, prog_bar = True)
        self.log('new_sos',     counter, on_epoch=True, prog_bar = True)

        return {
            "loss": loss
        }

    def log_wavefield(self, wavefield, name):
        wavefield = (hardtanh(wavefield) + 1) / 2
        self.logger.experiment.add_images(
            "wavefield/" + name + "_real",
            wavefield[0],
            self.trainer.global_step,
            dataformats="HW",
        )
        self.logger.experiment.add_image(
            "wavefield/" + name + "_imag",
            wavefield[1],
            self.trainer.global_step,
            dataformats="HW",
        )

    def get_initials(self, sos_maps: torch.tensor):
        """Gets the initial estimates for state, wavefield and residual. It
        also calculate k_sq = (omega/c)**2

        Args:
            sos_maps (tensor): Speed of sound map

        Returns:
            (tensor, tensor): k_sq, wavefield
        """
        # TODO: Make it trainable?

        k_sq = (self.hparams.omega / sos_maps) ** 2
        wavefield = torch.zeros(
            k_sq.shape[0], 2, k_sq.shape[2], k_sq.shape[3], device=k_sq.device
        )
        return k_sq, wavefield

    def apply_laplacian(self, x: torch.tensor):
        #laplacian now needs a contiguous memory to work
        return self.Lap(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)

    def get_residual(self, x: torch.tensor, k_sq: torch.tensor):
        # TODO: This should be outside of the networ, as represents the
        #      environment
        """Returns the residual wavefield

        Args:
            x (tensor): Current solution estimate for the Helmholtz equation
            k_sq (tensor): (omega/c)**2

        Returns:
            torch.tensor: the residual
        """
        return self.apply_laplacian(x) + k_sq * x - self.source

    def single_step(
        self, wavefield: torch.tensor, k_sq: torch.tensor, residual: torch.tensor, get_residual : bool = True
    ):
        #  Getting residual signal
        # residual = self.get_residual(wavefield, k_sq)

        sigmas = self.sigmas.unsqueeze(0).repeat(wavefield.shape[0], 1, 1, 1) #self.sigmas on device, no need to move

        input = torch.cat([wavefield, 1e3 * residual, sigmas], dim = 1)

        # Predicting wavefield update
        d_wavefield = self.f(input)  # *100/self.current_iterations
        up_wavefield = d_wavefield / 1e3 + wavefield
        new_residual = self.get_residual(up_wavefield, k_sq)

        # Impose Dirichlet BC on updated wavefield
        """
        dirichlet_mask = torch.zeros_like(up_wavefield)
        dirichlet_mask.requires_grad = False
        dirichlet_mask[:,:,1:-1,1:-1] = 1.
        up_wavefield = up_wavefield*dirichlet_mask
        """
        #get_residual = True
        if get_residual:
            return up_wavefield, new_residual
        else:
            return up_wavefield

    def n_steps(
        self,
        wavefield,
        k_sq,
        residual,
        num_iterations,
        return_wavefields=False,
        return_states=False,
    ):
        # Initialize containers
        wavefields = []
        residuals = []
        states = []

        # Unroll N steps
        for current_iteration in range(num_iterations):
            # Update wavefield and get residual AFTER update
            wavefield, residual = self.single_step(
                wavefield, k_sq, residual, get_residual=True
            )

            #  Store
            residuals.append(residual)  # Last residual
            if return_wavefields:
                wavefields.append(wavefield)
            if return_states:
                states.append(self.f.get_states(flatten=True))

        #  Add only last wavefield if none logged
        if not return_wavefields:
            wavefields.append(wavefield)

        return {
            "wavefields": wavefields,
            "residuals": residuals,
            "states": states,
            "last_iteration": current_iteration,
        }

    def fast_forward(self, sos_maps):
        # Finite horizon value
        num_iterations = self.hparams.max_iterations

        # Initialize inputs and network states
        k_sq, wavefield = self.get_initials(sos_maps)
        self.f.clear_states(wavefield)
        residual = self.get_residual(wavefield, k_sq)
        sigmas = (
            self.sigmas.unsqueeze(0).repeat(wavefield.shape[0], 1, 1, 1).to(self.device)
        )

        # Initialize containers
        wavefields = torch.empty(
            [num_iterations] + list(wavefield.shape[1:]),
            device="cuda:1",
            dtype=torch.float32,
        )

        # Unroll N steps
        for current_iteration in range(num_iterations):
            # Loop
            wavefield, residual = self.single_step(wavefield, k_sq, residual)

            #  Store
            wavefields[current_iteration] = wavefield[0]

        return wavefields

    def forward(
        self,
        sos_maps,
        return_wavefields=False,
        return_states=False,
        num_iterations=None,
        stop_if_diverge=False,
    ):
        # Finite horizon value
        if num_iterations is None:
            num_iterations = self.hparams.max_iterations

        # Initialize inputs and network states
        k_sq, wavefield = self.get_initials(sos_maps)
        self.f.clear_states(wavefield)
        residual = self.get_residual(wavefield, k_sq)

        # Initialize containers
        wavefields = []
        residuals = []
        states = []

        # Unroll N steps
        for current_iteration in range(num_iterations):
            # Update wavefield and get residual AFTER update
            wavefield, residual = self.single_step(wavefield, k_sq, residual)

            #  Store
            residuals.append(residual)  # Last residual
            if return_wavefields:
                wavefields.append(wavefield)
            if return_states:
                states.append(self.f.get_states(flatten=True))

        #  Add only last wavefield if none logged
        if not return_wavefields:
            wavefields.append(wavefield)

        return {
            "wavefields": wavefields,
            "residuals": residuals,
            "states": states,
            "last_iteration": current_iteration,
        }

    def forward_variable_src(
        self,
        sos_maps,
        src_time_pairs,
        return_wavefields=False,
        return_states=False,
        num_iterations=None,
        stop_if_diverge=False,
    ):
        # Finite horizon value
        if num_iterations is None:
            num_iterations = self.hparams.max_iterations

        # Extract source insertion times
        new_src_times = src_time_pairs["iteration"]
        src_maps = iter(src_time_pairs["src_maps"])

        # Initialize inputs and network states
        k_sq, wavefield = self.get_initials(sos_maps)
        self.f.clear_states(wavefield)
        residual = self.get_residual(wavefield, k_sq)

        # Initialize containers
        wavefields = []
        residuals = []
        states = []

        # Unroll N steps
        for current_iteration in range(num_iterations):
            # Update source map if needed
            if current_iteration in new_src_times:
                self.set_source_maps(next(src_maps))
                # _, wavefield = self.get_initials(sos_maps)
                # self.f.clear_states(wavefield)
                residual = self.get_residual(wavefield, k_sq)

            # Update wavefield and get residual AFTER update
            wavefield, residual = self.single_step(wavefield, k_sq, residual)

            #  Store
            residuals.append(residual)  # Last residual
            if return_wavefields:
                wavefields.append(wavefield)
            if return_states:
                states.append(self.f.get_states(flatten=True))

        #  Add only last wavefield if none logged
        if not return_wavefields:
            wavefields.append(wavefield)

        return {
            "wavefields": wavefields,
            "residuals": residuals,
            "states": states,
            "last_iteration": current_iteration,
        }
