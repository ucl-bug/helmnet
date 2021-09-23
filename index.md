![Cover](/images/cover.png)

<div style="text-align: justify">
Transcranial ultrasound therapy is increasingly used for the non-invasive treatment of brain disorders. However, conventional numerical wave solvers are currently too computationally expensive to be used online during treatments to predict the acoustic field passing through the skull (e.g., to account for subject-specific dose and targeting variations). As a step towards real-time predictions, in the current work, a fast iterative solver for the heterogeneous Helmholtz equation in 2D is developed using a fully-learned optimizer. The lightweight network architecture is based on a modified UNet that includes a learned hidden state. The network is trained using a physics-based loss function and a set of idealized sound speed distributions with fully unsupervised training (no knowledge of the true solution is required). The learned optimizer shows excellent performance on the test set, and is capable of generalization well outside the training examples, including to much larger computational domains, and more complex source and sound speed distributions, for example, those derived from x-ray computed tomography images of the skull.
</div>

## Learned iterative solver
---
Given an invertible linear operator $A$, in our case the Helmholtz operator, and a source term $\rho$, our goal is to find a $u$ such that $Au = \rho$.

The learned iterative solver updates the current estimate of the wavefield $u_k$ as 

$$
u_{k+1} = u_k + f_\theta(u_k, c, \rho)
$$

Where $f_\theta$ is a learned function. To leverage the knowledge of the forward operator $A$, the learned function is defined as

$$
u_{k+1} = u_k + f_\theta(u_k, e_k), \qquad e_k = A(c)u_k - \rho
$$

The learned solver is implemented as a lightweight UNet, trained by minimizing the residual error.