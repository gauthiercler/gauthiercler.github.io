---
layout: distill
title: "From templates to neural networks"
date: 2026-07-31
description: An intuitive journey on profiled side channel attacks
tags: side-channel “neural networks” template
related_posts: false
related_publications: true

bibliography: template-dl.bib
---

Profiled attacks are the strongest scenario for an attacker when performing side channel attacks. Thanks to the access to an open device, leakage characterization can be performed prior to the attack on the closed target device. Since 2002, the standard tool for profiled attacks was the template method. However, since, neural networks have been introduced as a powerful alternative and studied thoroughly. 

Through this post, we will see the link between templates and neural networks and see that the latter ones are not always the definitive answer for profiled attacks.

No deep learning background is required beyond linear algebra and probability.

## Context

Let’s consider an embedded device (a smart card, microcontroller, IOT device, ...) running cryptographic routines where a sensitive secret key is manipulated.

While the used algorithm is mathematically hardened, the device can leak information physically through side channels, such as power consumption, electromagnetic emanation, timing or temperature.

The resulting signal of each execution can be recorded as a trace vector $\mathbf{x}$ of sampled points.

The relationship between the obtained trace and the secret is defined through a leakage model, the most standard ones being:
- the Identity model, where the signal depends on the exact intermediate value
- the Hamming weight/distance model, where for CMOS circuits, consumption is tied to bit transitions

Let’s now consider for the sake of simplicity a widely studied cryptographic primitive target, the AES first round S-box output $y = \text{Sbox}(p \oplus k)$ for a single byte, with known plaintext $p$ and secret key $k$. Recovering $y$ allows to recover $k$ since $p$ is known. The search space is a 256 classes (8 bit) classification problem.

Theoretically, this could be tackled entirely through a non-profiled attack such as a CPA or MIA, without the need of an open device. However, we focus in this post on the other approach, where prior characterization of the leakage is performed on a clone open device.

While less realistic than non profiled ones, profiled attacks are the best scenario for an attacker and the worst for the targeted entity.

The profiled classification problem can be defined as follow. Given a set of independent samples $\mathbf{x}$, their associated target classes $y$, and a predictor $f(\mathbf{x})$ which is trained using the data to predict the accurate class label $\hat{y}$ or a probability distribution over all possible labels. This is known as the training phase. This corresponds in our case to the access to the open device, where the attacker can obtain many tuples $(\mathbf{x}, y)$ with $\mathbf{x}$ being acquired signal trace and $y$ the actual value targeted.

Then, the fitted model $f$ is applied on traces from the target device to predict the intermediate values of the unknown fixed key. This is known as the matching phase.

Note that we assume perfect portability between open and target devices. In practice, between devices portability and measurement differences can affect the matching phase. <d-cite key="choudary_portability"></d-cite>

Now, the question that arises is how to construct an accurate model that maps the leakage observed on a device $A$ to a device $B$.

## Template attack

The template is one of the fundamental approaches for performing profiled attacks.

A template can be seen as a profiled classifier.

The general form for template attack, as introduced in the original paper <d-cite key="chari_template"></d-cite> is the Quadratic Discriminant Analysis (QDA).
The main assumption is the multivariate normality. The template is a deterministic method which models the conditional probability $p(x|k)$ from the data with the multivariate normal density function of dimension $d$:

$$p(x|k) = \frac{1}{\sqrt{(2\pi)^d|\Sigma_k|}}\exp{(-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k))}$$
where $x$ is a vector of dimension $d$ for each trace, $\mu_k$ and $\Sigma_k$ are the mean vector and covariance matrix for class $k$.

From there the prediction step uses the discriminant (or score function) for each class guess:

$$\delta_k(x) = -\frac{1}{2}\log|\Sigma_k|-\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}_k(x-\mu_k) + \log P(k)$$

<aside><p>In practice, if the prior P(k) is uniform (for example 256 classes AES byte), it is often dropped, then the QDA becomes a pure MLE classifier instead of a MAP classifier</p></aside>

Then the prediction is the class with the highest score:

$$\hat{y} = \arg \max_k \delta_k(x)$$

It is also possible to map to the probability simplex $p(x) \in \Delta^{K-1}$ using the softmax function $\sigma : \mathbb{R}^K \rightarrow \Delta^{K-1}$ such that for each class, the probability is:

$$p_k(x) = \frac{\exp(\delta_k(x))}{\sum_{j=1}^K\exp(\delta_j(x))}$$

This allows to recover the bayesian posterior $p(k|x)$.

While powerful, the main problem of the QDA method is that it scales poorly. Indeed, many samples are needed for each class to ensure correct covariance matrices estimation and avoid singularities.

To reduce the complexity, many works use instead the Linear Discriminant Analysis (LDA), also known as a pooled template <d-cite key="choudary_lda"></d-cite> in the side channel domain. It reduces to a single shared covariance matrix across all classes, so $\Sigma_k = \Sigma$ for all $k$. In other words, the ellipsoid shape of all classes is identical, only shifted by their means. <aside><p>this is known as Homoscedasticity.</p></aside>

Since the covariance matrix is now shared, the quadratic term $x^T\Sigma^{-1}x$ becomes identical to all classes and can be dropped, thus the discriminant becomes:

$$\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log P(k)$$
with the first term being strictly linear to $x$.

The above discriminant for LDA can also be reformulated as:
$$\delta_k(x) = \mathbf{w}^T_kx + b_k$$
with the dot product $\mathbf{w}_k = \Sigma^{-1}\mu_k$ and the bias $b_k = -\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log P(k)$. We will see later how this relates to neural networks.

The assumption of shared covariances for the LDA is consistent for side channel as we expect the measurement noise components to behave similarly accross all classes (in case of first order leakages).

## Limits

Still, while templates are powerful tool for profiled attacks, they emphasize some limitations.

Firstly, their decision boundary is defined and can be limiting.
The QDA boundary is a quadratic hypersurface since only the two first statistical moments are used (mean and variance). The LDA on the other side limits to the first order moment expression (since the covariance matrix is shared), and thus the boundary becomes a flat hyperplane.

For instance, on a masking countermeasure of order $d$, the lowest statistical moment of leakage is $d+1$, equal to the number of shares. [in univariate case].

A workaround for this problem can be to perform some non-linear projections to a subspace where the template can be applied (for example using kernel methods).
This is however heavy and complex leakage models are still hard to approximate.
Alternatively, the attacker is required to perform recombination of shares beforehand (for example centered product of shares time points).

Another limitation of the template for profiled attacks is its poor resistance to jitter on the traces.
The template models a multivariate distribution, where each dimension corresponds to realizations of a single temporal point. Therefore precise alignment is necessary for optimal results and to avoid unmatched indices.

Finally, template scale poorly when the number of dimensions increases. This is primarily due to cost for the inversion of the covariance matrix ($\mathcal{O(d^3)}$ for the LDA, $\mathcal{O(k \cdot d^3)}$ for the QDA).
It is often better to first perform a dimensionality reduction (using a PCA for example) or select only the most relevant point of interests beforehand.

Let now see how neural networks come as a relevant alternative against these limitations as they have the ability to learn these steps.


## Neural networks

As we have seen, templates handle the profiling task by maximizing the likelihood in closed form from the data.

Neural networks instead solve the problem through an optimization procedure.

A elementary neural network layer consists in the following:

$$h = \phi(\mathbf{W}x + b)$$

where $\phi$ is an activation function. If $\phi$ is the identity function, it simply consists in the affine transformation, which is equivalent to the LDA reformulation from previous section. Indeed, in this scenario the LDA and neural network spans the same hyperplane family.

The strength of neural networks lie in the choice of $\phi$. For example, if we consider the ReLU activation $\phi(x) = \max(0,x)$, the boundary becomes non-linear [relu is one of the fundamental non linear activation for hidden layers. many variants where derived form ReLU (ELU, SELU, GELU, Leaky ReLU). This allows neural networks to be universal function approximator with enough capacity of $\mathbf{W}$.

This can then be used as function composition to build layers:

$$f(x) = f_L(\cdots f_2(f_1(x)))$$

This composition of layers is known as Multilayer Perceptron (MLP).

It is possible to imagine the neural network composition as a straight line bended at each activation. The following figure illustrates the boundary difference between the LDA, QDA and neural network:

{% include figure.liquid loading="eager" path="assets/img/boundary.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption=caption_text%}

Without non linear activations, the model would simply be a composition of affine maps, and would not be able to model higher degrees polynomial. For the side channel task, it means that if the neural network capacity is sufficient (enough layers and neurons per layers), it can theoretically models/represents masked leakage of any order with enough leaky data examples.
[Masking remains really effective since the number of traces needed grows exponentially as the number of shares increases. For high order masking, some help is often needed such as data augmentation or precomputed products.

The prediction obtained through this composition (the forward pass) is then compared to the expected value, using a loss (or cost function) $\mathcal{L}$ which define the model objective.

For instance, in a classification problem, the cross entropy loss is typically used:

$$\mathcal{L} = -\sum_{k=1}^K y_k \log p_k$$

between the expected $y$ (one hot encoded labels) and predicted probability distribution $p$ (using softmax as last layer activation).
Minimization of the categorical cross-entropy with one-hot labels is equivalent to maximizing the conditional log-likelihood like the template fitting.

The optimization procedure consists in finding weights $\mathbf{W}$ that minimize this loss. Gradients are propagated from the loss back to the first layer using the chain rule backpropagation (the backward pass).

This forward/backward cycle is repeated several times on batches of data until convergence is reached (a local minima).

Unlike the template which output is deterministic, the neural network convergence results are sensitive to the weights initialization.

As long as the whole architecture is differentiable, neural networks can be applied to various tasks with defined objectives. Additionally, we saw now the fully connected block (all neurons are interconnected), but other blocks are possible.

At this point we see that neural networks are overcoming template limitations for higher order leakage extraction.

Still, this architecture is likely to struggle against jitter or desynchronization countermeasures like the template. For optimal efficiency, alignment is still recommended.


Additionally, the training for the neural network is a stochastic process and thus partially depends on the weights random initialization.

## Convolutions

We saw that MLP can model non-linear high order leakages, but like the template they still require aligned temporal points of each sub-leakage for optimal exploitation.


It is possible to learn inside the model the feature extraction step. This can be performed by using convolution layers. This is analogous to convolution used in signal processing.

This can be seen as a pre processing step, but it is powerful as it is done automatically during the training.

Essentially, convolutions for neural networks are set of independent filters which parameters are learnt. these filters are slid along the trace like a moving operator with the matrix being the learned convolution weights.

Additionally, in their most basic form, these convolutions are completed with pooling layers, which are used to perform some down sampling. Average or Max pooling layers are usually used.
<aside><p>more complex convolutions blocks are also used, such as residual blocks, unet,...</p></aside>

The output of these blocks is then flattened and usually plugged into the logical part of the network which consists in an MLP classification head.
[keep in mind that unlimited architecture variants are possible, this is just for explanation purposes]. 

Extensive research has been performed in the side channel domain since the introduction of neural networks for profiled attacks (2016), Some works aimed at proposing methodology for selecting the optimal architecture parameters <d-cite key="zaid_methodology"></d-cite>, visualisation and explainability <d-cite key="masure_gradient"></d-cite>.

It is also relevant to note that convolutions can have a recombination effect if multiple shares are manipulated at close temporal intervals.

This allows a fully differentiable self-contained preprocessing/filtering + feature extraction + classifier head pipeline.

## Accuracy is not the key

In machine learning field, for general purpose classification accuracy is almost always the considered metric.

However, for side channel profiled attacks with traces accumulation, thats not necessarily the case.

As the predictions are combined among several traces, the goal is to obtain a decrease of entropy. That is why the rank of the correct key (argsort) is seen as a better indicator of performance. Even an average rank slightly higher to random can be sufficient for key recovery by accumulating enough traces. <aside><p>If the rank of the correct value is always 2, the resulting accuracy is equal to zero.</p></aside>

It mainly depends on what is the goal and the attack path.

As introduced previously , if we target an AES encryption and we try to recover the unknown fixed key on the target device, we can perform scores accumulation, analogously to the CPA, which is accumulating traces and computing the Pearson coefficient for each guess.

We can do it in similar fashion here.
As described above for both template and neural network, we obtain a probability distribution over all class guesses, for each trace.

From there, several methods can be used to combine these probabilities.
One way can be to simply multiply each guess value across several realizations. After accumulation, the largest resulting probability is the most probable candidate.
<aside><p>For numerical stability on large set of accumulations, we often do the sum of log’s probabilities instead</aside><p>
Others smarter approaches exist to combine probabilities, such as histogram exploration or belief propagation (for several intermediate values).

For neural networks, it is a good habit to monitor the average rank of the correct class (on batches) during the training. This allows to see if the model is able to learn on the data or not.

The rest of the attack is almost identical to the CPA. Instead of working with Pearson correlation values, it now uses probabilities.
The decrease in guessing entropy (average rank on several attacks) during accumulation until the remaining key space is small enough to brute-force.

## Conclusion

In this post, we covered templates and neural networks as tool for profiled side channel attacks. We saw how the latter one can overcome the limitations of the former one, using different paradigms.

Nevertheless, it is important to keep in mind that templates remain competitive for profiled side channel attacks, especially for simpler leakage models. Unlike Neural networks, they do not require extensive architecture or parameters tuning. Especially consider the LDA if traces are limited.

Additionally, working with neural networks for profiled attacks is not as straightforward as with templates. It requires some expertise for model training and architecture tuning.

Here is a table which summarizes briefly the main differences between the two approaches:

|  | Template (LDA/QDA) | Neural network (MLP/CNN) |
|:--|:--|:--|
| Model type | Generative, $p(x\mid y)$ | Discriminative, $p(y\mid x)$ |
| Estimation | Closed-form | Gradient descent |
| Parametric | No | Architecture + hyper-parameters |
| Decision boundary | Hyperplane / quadratic | Arbitrary (depends on capacity) |
| Reproducibility | Deterministic | Stochastic, sensitive to initialization |
| Higher-order recombination | Manual, beforehand | Learned |
| Feature selection | POI / PCA needed | Automated |
| Desync robustness | Needs alignment | Pooling + augmentation |
| Ease of use | Straightforward | Requires expertise |
