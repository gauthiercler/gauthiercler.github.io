---
layout: distill
title: Adversarial Attacks
date: 2021-05-22
# description: Adversarial Attacks
tags: adversarial
# categories: sample-posts
related_posts: false
related_publications: true

bibliography: adversarial.bib
---

Adversarial attacks on machine learning is a wide domain of interest. In this first, we cover white box adversarial attacks using the gradient sign method. In a white box setting, the attacker has access to all the information used for training the model, its weights and parameters, along the used datasets for training.

### FGSM

The Fast Gradient Sign Method (FGSM) is a straightforward approach proposed in <d-cite key="goodfellow2015explainingharnessingadversarialexamples"></d-cite> to generate adversarial sample from a trained model. Given the model parameters $\boldsymbol{\theta}$, some input samples $\boldsymbol{x}$ and associated ground truth label $y$, the general objective that the model aims at minimizing during training is defined as:

$$
J(\boldsymbol{\theta}, \boldsymbol{x}, y)
$$

The adversarial perturbation is given as:

$$
\boldsymbol{\eta} = \epsilon \text{sign}(\nabla_{\boldsymbol{x}} J(\boldsymbol{\theta}, \boldsymbol{x}, y)).
$$

By taking the gradient of the loss w.r.t to $\boldsymbol{x}$, it provides the direction in which the input $\boldsymbol{x}$ should change in order to maximize the loss. Thus, a new generated adversarial sample can simply be generated as $\boldsymbol{x'} = \boldsymbol{x} + \boldsymbol{\eta}$. Given a small value of $\epsilon$, a new generated sample cannot be distinguished by the human eye from a real one, but is sufficient to deceive the model. In other words, a slight alteration of the input image pixels causes a consequent change in the model embedding.

{% assign caption_text = "Image from " %}
{% assign caption_text = caption_text | append: ' <d-cite key="goodfellow2015explainingharnessingadversarialexamples"></d-cite>.' %}
{% include figure.liquid loading="eager" path="assets/img/fgsm.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption=caption_text%}

However, this approach only causes the model to missclassify samples.
Instead, an variant can be used to target a specific class to be predicted by the model, maximizing $p(\hat{y}|\boldsymbol{x})$, where $\hat{y}$ corresponds to the label of the targeted class. Previous perturbation now becomes:

$$
\boldsymbol{\eta} = \epsilon \text{sign}(\nabla_{\boldsymbol{x}} J(\boldsymbol{\theta}, \boldsymbol{x}, \hat{y})).
$$

This time, as we aim at going in the direction that minimize the loss (instead of maximizing it) of the target class $\hat{y}$, the adversarial sample becomes $\boldsymbol{x'} = \boldsymbol{x} - \boldsymbol{\eta}$. The choice of the target class can be motivated by a specific attack strategy (for example predict dogs as cats). In <d-cite key="kurakin2017adversarialexamplesphysicalworld"></d-cite>, authors suggest to choose the least likely class based on the trained model prediction:

$$
\hat{y} = \underset{y}{\arg\min} \{p(y|\boldsymbol{x})\} 
$$

This is motivated by the fact that least likely predicted class in is higly dissimilar to the actual true class. This can be used to maximize the model misclassification.

This adversarial process can also be applied iteratively with a small step size. 

$$
\boldsymbol{x'}_{n + 1} = \boldsymbol{x'}_{n} - \epsilon \text{sign}(\nabla_{\boldsymbol{x}_n} J(\boldsymbol{\theta}, \boldsymbol{x}, \hat{y}))
$$

<aside><p>Note that clipping may be necessary depending on the data used.</p></aside>


with $\boldsymbol{x'}_{0} = \boldsymbol{x}$. This latter approach is more effective than the former one step generation as it produce smaller perturbations and is less destructive.

### Adversarial training

This adversarial generation method can be used as a regularization for training more robust models. In its simplest way, it can be formulated as:

$$
\tilde{J}(\boldsymbol{\theta}, \boldsymbol{x}, y) = \alpha J(\boldsymbol{\theta}, \boldsymbol{x}, y) + (1 - \alpha) \epsilon \text{sign}(\nabla_{\boldsymbol{x}} J(\boldsymbol{\theta}, \boldsymbol{x}, y)).
$$

