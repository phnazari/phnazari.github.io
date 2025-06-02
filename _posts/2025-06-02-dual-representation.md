---
layout: distill
title: The Dual Representation of ReLU Networks
description: In this post, we introduce the dual representation of fully connected feedforward ReLU networks. This is the first post of a three part series on the geometry of generalization of deep neural networks.
tags: Deep-Learning Geometry Generalization
giscus_comments: true
date: 2025-06-02
featured: false
mermaid:
  enabled: true
  zoomable: true
code_diff: false
map: false
chart:
  chartjs: false
  echarts: false
  vega_lite: false
tikzjax: false
typograms: false
theorems: true
citation: true

authors:
  - name: Philipp Nazari
    affiliations:
      name: ETH Zürich, Max Planck Institute for Intelligent Systems

bibliography: dual-representation.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
  - name: ReLU Networks
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Affine Geometry
    subsections:
      - name: Affine and CPA Functions
      - name: Affine Dualities
      - name: CPA Functions as Upper Convex Hulls
  - name: Dual Representation of Neural Networks
    subsections:
      - name: Neural Networks and Affine Geometry
      - name: Example


# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .statement {
    border-left: 4px solid #000; padding-left: 10px;
    margin: 10px 0;
    border-color: var(--global-theme-color);
  }

  .equation-container {
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
  }

  .MathJax {
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
  }

  /* COUNTERS */
  body {
    counter-reset: statement-counter figure-counter;
  }

  /* Style and number theorems */
  .statement {
      counter-increment: statement-counter;
      margin: 1em 0;
  }

  .figure {
      counter-increment: figure-counter;
  }

  .definition::before {
      content: "Definition " counter(statement-counter);
      font-weight: bold;
  }

  .lemma::before {
      content: "Lemma " counter(statement-counter);
      font-weight: bold;
  }

  .corollary::before {
      content: "Corollary " counter(statement-counter);
      font-weight: bold;
  }

  .proposition::before {
      content: "Proposition " counter(statement-counter);
      font-weight: bold;
  }

  .remark::before {
      content: "Remark " counter(statement-counter);
      font-weight: bold;
  }

  .figure::before {
      content: "Figure " counter(figure-counter);
      font-weight: bold;
  }

  html {
      scroll-behavior: smooth;
  }

---

# The Dual Representation

This is part one of a three part series on the geometry of generalization, which is the essence of my [master thesis](/assets/pdf/Master_Thesis.pdf)<d-cite key="nazari2025thesis"></d-cite>. In this series, we will present a novel perspective to think about why overparameterized networks generalize well.

The series is structured in the following way:

<ul>
<li>In <a href="{% post_url 2025-06-02-dual-representation %}">this post</a>, we establish a dual representation of fully connected feedforward ReLU networks.</li>
<li>In part two, we show how this dual representation can be used to derive complexity measures for these networks.</li>
<li>In part three, we will use these complexity measures to find evidence for the volume hypothesis<d-cite key="chiang2022loss"></d-cite>, an approach to explain why overparameterized models generalize well.</li>
</ul>


## Introduction
The constructions put forward in this post are inspired by Piwek et al.<d-cite key="piwek2023exact"></d-cite>. In particular, we will establish a <em>dual representation</em> of fully connected feedforward ReLU networks, a symbolic representation that allows thinking about ReLU networks and their complexity in an abstract setting. This will be useful in part two and three of this series.

The dual representation has been established in a number of previous works<d-cite key="piwek2023exact, zhang2018tropical, alfarra2022decision"></d-cite> and utilizes tropical geometry. While this is an interesting theory, one can use the close relationship between tropical geometry and affine geometry to make the constructions easier to interpret.

However, throughout this series, I will refrain from proving all statements in detail. In particular, one can quickly forget about tropical geometry and its relationship to affine geometry. The curious reader may refer to Chapter 3 and Chapter 4 of my thesis<d-cite key="nazari2025thesis"></d-cite>, in particular Section 4.2, for more details on this relationship.

Finally, for every claim made in this series, I will refer to the corresponding statement in the thesis. There, one can find the proof, as well as a reference to the corresponding statement by Piwek et al.<d-cite key="piwek2023exact"></d-cite> wherever appropriate.

## ReLU Networks
To introduce notation and get everybody on board, we start by quickly defining fully connected feedforward ReLU networks.

<div class="statement definition" id="def:nn">
<strong>(Fully Connected Feedforward Networks)</strong>
    A fully connected feedforward network $\mathcal N := \mathbb R^{d} \to \mathbb R^{n_L}$ takes as an input a vector $\mathbf x \in \mathbb R^{d}$ and returns an output $\mathbf y := \mathbf a_L$. It is defined inductively by

    $$
    \left\{
    \begin{aligned}
    \mathbf a_0 &:= \mathbf x \\
    \mathbf a_{l+1} &= \rho_{t_{l+1}}\left({\mathbf W}_{l+1} \mathbf a_l + \mathbf b_{l+1}\right), \; 0 = 1,\ldots,L-1,
    \end{aligned}
    \right.
    $$

    where $\mathbf W_{l+1} \in \mathbb R^{n_{l+1}, n_{l}}$ and $\mathbf b_{l+1} \in \mathbb R^{n_{l+1}}$ are the <em>weight matrix</em> and <em>bias vector</em> at layer $l+1$. Furthermore, $\rho_{t_{l+1}}(x) = \max(x, t_{l-1})$ is the <em>activation function</em> at layer $l+1$ with <em>threshold</em>  $t_l \in \mathbb R \cup \{-\infty\}$. The number $L$ is called the <em>depth</em> of the network, while $n_l$ is the <em>width</em> of layer $l$. The network is <em>deep</em> if $L \gg 1$.
</div>

The two activation functions we consider are:
<ul>
    <li> the ReLU $\rho_0(x) = \max(x,0)$</li>
    <li> the identity $\rho_{-\infty}(x) = x$.</li>
</ul>

This leads to the following definition:
<div class="statement definition">
<strong>(ReLU Networks)</strong>
A network in the sense of <a href="#def:nn" class="cite-stmt hover-link">Definition</a> with ReLU activations (and potentially a linear activation at the last layer) is called a <em>ReLU network</em>.
</div>

## Affine Geometry

In this section, we introduce fundamental concepts of affine geometry, covering basic definitions, the dual representation of affine functions, and their connection to upper convex hulls. Throughout this section, fix an integer $d \in \mathbb N$.
### Affine and CPA Functions
We begin with some fundamental concepts.


<div class="statement definition" id="def:affine-functions">
<strong>(Affine Functions)</strong>
Given a vector $\mathbf{a} \in \mathbb R^d$ and a scalar $b \in \mathbb R$, we define the affine function with parameters $\mathbf a$ and $b$ as

$$
\begin{align*}
    f_{\mathbf a, b} \colon \mathbb R^d &\to \mathbb R \\
    \mathbf x &\mapsto \langle \mathbf a, \mathbf x \rangle + b,
\end{align*}
$$

where $\langle \cdot, \cdot \rangle$ is the Euclidean inner product on $\mathbb R^d$.
</div>

Ultimately, we will be taking maxima over affine maps. To classify such maxima, we introduce CPA functions:

<div class="statement definition">
<strong>(CPA Functions)</strong>
We say that a function $f \colon \mathbb R^d \to \mathbb R$ is CPA if it is convex and piecewise affine. We denote by $\text{CPA}(d)$ the set of CPA functions $\mathbb R^d \to \mathbb R$.
</div>

It turns out that the class of CPA functions coincides with the class of maxima over affine functions:

<div class="statement proposition" id="proposition:cpa">
<strong>(Characterizing CPA Functions, Proposition 2 in Piwek et al.<d-cite key="piwek2023exact"></d-cite>)</strong>
Every function $F \colon \mathbb R^d \to \mathbb R$ of the form

$$
\begin{equation*}
    F(\mathbf x) = \max\{f_1(\mathbf x),\ldots,f_n(\mathbf x)\}
\end{equation*}
$$

with affine functions $f_i \colon \mathbb R^d \to \mathbb R$ is CPA. Also every CPA function with a finite number of affine pieces is of this form.
</div>

Later in this series, we will also consider differences of CPA functions:
<div class="statement definition">
<strong>(DCPA Functions)</strong>
We say that a function $f \colon \mathbb R^d \to \mathbb R$ is DCPA if it can be written as the difference of two CPA functions. We denote by $\text{DCPA}(d)$ the set of DCPA function $\mathbb R^d \to \mathbb R$.
</div>

### Affine Dualities
In this section, we mainly follow the argument presented by Piwek et al.<d-cite key="piwek2023exact"></d-cite>, which allows mapping an affine function $f_{\mathbf a, b} \colon \mathbb R^d \to \mathbb R$ to a dual space. As an outlook, exploring this transformation will ultimately lead to understanding how ReLU networks can be understood as DCPA functions.

<div class="statement definition">
The graph of an affine function $\mathbb R^d \to \mathbb R$ defines a hyperplane in <em>real space</em>, which we define as $\mathcal R := \mathbb R^d \times \mathbb R = \mathbb R^{d+1}$.

The space of affine functions whose graph lies in $\mathcal R$ is called <em>real affine space</em>, denoted by $\text{Aff}_{\mathfrak R}(d)$.
</div>

As mentioned in <a href="#def:affine-functions" class="cite-stmt hover-link">Definition</a>, any affine function $f_{\mathbf{a},b} \in \text{Aff}_{\mathfrak{R}}(d)$ is characterized by its parameters $(\mathbf a, b) \in \mathbb R^{d+1}$:

<div class="statement definition">
We refer to the copy of $\mathbb R^{d+1}$ that parametrizes affine functions in $\text{Aff}_{\mathfrak{R}}(d)$ as <em>dual space</em>, denoted by $\mathcal D$.
</div>

The following lemma is a natural consequence of this construction, as it allows translating between real affine space and dual space:

<div class="statement lemma"><strong>(Lemma 3.2.1 in Nazari<d-cite key="piwek2023exact"></d-cite>)</strong>
For any fixed dimension $d$, there exists a bijection between dual space and real affine space, given by

$$
\begin{align*}
    \mathcal R \colon \mathcal D &\xrightarrow{\sim} \text{Aff}_{\mathfrak R}(d) \\
    (\mathbf x, y) &\mapsto f_{\mathbf x, y}.
\end{align*}
$$
</div>


<div class="row mt-3 align-items-center">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/example_dual_point.svg" class="img-fluid rounded" zoomable=true caption="<strong>(a)</strong>" %}
    </div>
    <div class="caption-left">
    </div>
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/example_dual_plane.svg" class="img-fluid rounded" zoomable=true caption="<strong>(b)</strong>" %}
    </div>
</div>
<div class="caption-left figure" id="fig:ex-dual">
    Example of the dual representation of an affine map $f{\mathbf a, b}$ with $\mathbf a = (-1/2, -3/4)$, $b=3/4$. Subfigure <strong>(b)</strong> contains the graph of $f_{\mathbf a,b}$ and Subfigure <strong>(a)</strong> contains the parameterizing dual point $(\mathbf a, b) \in \mathcal D$. The map $\mathcal R$ assigns to the point $(\mathbf a,b)$ the affine map $f_{\mathbf a, b}$.
</div>


An example for $\mathcal R$ can be found in <a href="#fig:ex-dual" class="cite-fig hover-link">Figure</a>. It has the following properties.

<div class="statement proposition"><strong>(Proposition 3.2.2 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>
Let $\{\mathbf x_i, y_i\}_{i=1,\ldots,n} \subseteq \mathcal D$ be a set of dual points. Then the following are true:

<ol type="i">
<li>

$\mathcal R$
is a linear operator, i.e., for any set of scalars $\{\alpha_i\}_{i=1,\ldots,n} \subseteq \mathbb R$,

$$
\begin{equation*}
    \mathcal R\left(\sum_{i=1}^n \alpha_i (\mathbf x_i, y_i)\right) = \sum_{i=1}^n \alpha_i \mathcal R((\mathbf x_i,y_i)).
\end{equation*}
$$
</li>
<li>

The set of dual points is linearly independent if and only if the corresponding set $\{\mathcal R((\mathbf x_i,y_i))\}_{i=1,\ldots,n}$ of affine functions is linearly independent.
</li>
<li>

The set of dual points is affinely independent if and only if the corresponding set $\{\mathcal R((\mathbf x_i,y_i))\}_{i=1,\ldots,n}$ of affine functions is affinely independent.
</li>
</ol>
</div>

Since both $\mathcal R$ and $\mathcal D$ are copies of $\mathbb R^{d+1}$, it is natural to ask whether we can reverse their roles in the above construction. The answer to this question is yes. We define <em>dual affine space</em> $\text{Aff}_{\mathfrak D}(d)$ as the space of affine functions with graph in $\mathcal D$. Analogously to the above construction, these affine functions are parameterized by points in $\mathcal R$, though with a slight caveat:

<div class="statement lemma">
For any fixed dimension $d$, there exists a bijection between dual affine space and real space. It is given by

$$
\begin{align*}
        \check{\mathcal R} \colon \text{Aff}_{\mathfrak D}(d) &\xrightarrow{\sim} \mathcal R \\
        f_{\mathbf a, b} &\mapsto (-\mathbf a, b).
\end{align*}
$$
</div>
<div>
<a href="#fig:real-dual-diagram" class="cite-fig hover-link">Figure</a> provides an overview over the relationship between $\mathcal R, \mathcal D, \text{Aff}_{\mathfrak R}(d)$ and $\text{Aff}_{\mathfrak D}(d)$.
</div>

<div class="row mt-3 w-50 mx-auto">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/diagram.svg" class="img-fluid rounded" zoomable=true%}
    </div>
</div>
<div class="caption-left figure" id="fig:real-dual-diagram">
    Diagram indicating the relationship between real (affine) and dual (affine) space.
</div>

Note that, compared to $\mathcal R$, the function $\check{\mathcal R}$ includes an additional minus and maps in the opposite direction. This is essential for ensuring that the duality properties in the following proposition hold:
<div class="statement proposition">
<strong>(Duality Properties, Proposition 7 in Piwek et al.<d-cite key="piwek2023exact"></d-cite>) </strong>
The maps $\mathcal R$ and $\check{\mathcal R}$ have the following properties:
<ol style="i">
<li>

A dual point $\mathbf c \in \mathfrak D$ lies on the graph of a dual affine function $f_{\mathbf a,b} \in \text{Aff}_{\mathfrak D}(d)$ if and only if the graph of the corresponding real affine function $\mathcal R(\mathbf c)$ contains the corresponding real point $\check{\mathcal R}(f_{\mathbf a,b})$.
</li>
<li>

A dual point $\mathbf c \in \mathcal D$ lies above the graph of a dual affine function $f_{\mathbf a,b} \in \text{Aff}_{\mathfrak D}(d)$ if and only if the real point $\check{\mathcal R}(f_{\mathbf a,b})$ lies below the graph of $\mathcal R(c)$.
</li>
</ol>
</div>

### CPA Functions as Upper Convex Hulls
In the previous section, we explored a duality that allows identifying affine maps with the vector containing their parameters. In this section, we apply these results to maxima over affine functions, which, by <a href="#proposition:cpa" class="cite-stmt hover-link">Proposition</a>, can be understood as CPA functions.

In light of the duality results from the previous section, CPA functions correspond to finite sets of dual points:

<div class="statement definition" id="def:q">
On the set $\mathcal P_\text{fin}(\mathcal D)$ of finite subsets of $\mathcal D$, the operator

$$
\begin{align*}
    \mathcal Q \colon \mathcal P_\text{fin}(\mathcal D) &\to \text{CPA}(d)\\
    S &\mapsto \mathcal Q(S) := \max_{\mathbf s \in S} \mathcal R(\mathbf s)
\end{align*}
$$

assigns to a set of dual points the associated CPA function

$$
\begin{align*}
    \max_{\mathbf s \in S}\mathcal R(\mathbf s)(\mathbf x) = \max_{(\mathbf a, b) \in S} \langle \mathbf x, \mathbf a \rangle + b.
\end{align*}
$$

We define $\mathcal Q(\emptyset) := 0$. On a vector of finite sets of dual points, $Q$ acts component-wise.
</div>

Note that, by <a href="#proposition:cpa" class="cite-stmt hover-link">Proposition</a>, the operator $\mathcal Q$ does indeed map to $\text{CPA}(d)$.


<div class="row mt-3 w-50 mx-auto">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/example_uch.svg" class="img-fluid rounded" zoomable=true%}
    </div>
</div>
<div class="caption-left figure" id="fig:example-uch">
    Example of an upper convex hull. Let $S$ be the union of all displayed points. The blue points correspond to $\mathcal U^*(S)$. In particular, $\mathcal Q(S)$ is uniquely identified by those points.
</div>



Our next objective is to establish a connection between CPA functions and upper convex hulls. To begin, we first state the following proposition:

<div class="statement proposition">
<strong>(Maximality of Upper Convex Hull, Proposition 3.3.2 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>
Let $S \subseteq \mathcal D$ be a finite set of points. Then for every point $w \in \mathcal D$ lying below or on $\mathcal U(S)$, the affine function dual to $w$ lies fully below the maximum of the affine functions whose duals lie in $\mathcal U^*(S)$. That is,

$$
\begin{equation}
    \mathcal R(w) \leq \max\{\mathcal R(s) | s \in \mathcal U^*(S)\} = \mathcal Q(\mathcal U^*(S)).
\end{equation}
$$

If $w$ lies truly below $\mathcal U(S)$, then even

$$
\begin{equation}
    \mathcal R(w) < \mathcal Q(\mathcal U^*(S)).
\end{equation}
$$
</div>
Having established this proposition, the identification of CPA functions with upper convex hulls is a corollary:


<div class="statement corollary" id="corollary:upper-convex-hull-rep">
<strong>(CPAs as Upper Convex Hulls, Corollary 3.3.3 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>
Every CPA function $\mathcal Q(S)$ can be uniquely represented as an upper convex hull in dual space. That is, $\mathcal Q(S) = \mathcal Q(\mathcal U^*(S))$.
</div>

A visualization of <a href="#corollary:upper-convex-hull-rep" class="cite-stmt hover-link">Corollary</a> can be found in <a href="#fig:example-uch" class="cite-fig hover-link">Figure</a>.

## Dual Representation of Neural Networks

Using the above constructions, we will now establish a connection between fully connected feedforward networks and DCPA functions. This will ultimately enable us to translate the network to dual space.

### Neural Networks and Affine Geometry
In order to establish the connection, we first need to develop some more machinery, beginning with the definition of how to sum two sets:

<div class="statement definition">
Given two non-empty sets $X, Y \subseteq \mathbb R^{d+1}$, we define

$$
\begin{equation*}
    X \style{transform: rotate(45deg); display: inline-block;}{\boxtimes}
 Y := \{\mathbf x + \mathbf y | \mathbf x \in X, \mathbf y \in Y\}
\end{equation*}
$$

to be the <em>Minkowski sum</em> of $X$ and $Y$. We define $X \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} \emptyset := X$. On vectors of sets of dual points, we define $\style{transform: rotate(45deg); display: inline-block;}{\boxtimes}$ to act component-wise.
</div>

Next, we list some properties of the operator $\mathcal Q$ (see <a href="#def:q" class="cite-stmt hover-link">Definition</a>), which assigns to a set of dual points the corresponding CPA function:
<div class="statement lemma">
<strong>(Properties of $\mathcal Q$, Lemma 5.1.2 in <d-cite key="nazari2025thesis"></d-cite>)</strong>
For any two sets of points $X, Y \subseteq \mathcal D$ and every non-negative scalar $\alpha \geq 0$, the following are true:
<ol style="i">
<li>

$\mathcal Q(X \cup Y) = \max\{\mathcal Q(X), \mathcal Q(Y)\}$
</li>
<li>

$\mathcal Q(X \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} Y) = \mathcal Q(X) + \mathcal Q(Y)$
</li>
<li>

$\alpha \cdot \mathcal Q = \mathcal Q(\alpha \cdot X)$, where the multiplication on the right hand side is the natural multiplication of a set with a real number.
</li>
</ol>
</div>

Neural networks rely heavily on matrix multiplications. On our mission to translate them to dual space, we must establish the concept of matrix multiplication in the dual setting:
<div class="statement definition">
We define the multiplication of an $m \times n$ matrix $\mathbf A$ with a vector $X$ of $n$ finite sets of dual points as

$$
\begin{align*}
    \cdot \colon \mathbb R^{m, n} \times \left(P_\text{fin}(\mathcal D)\right)^n &\to \left(P_\text{fin}(\mathcal D)\right)^m\\
    (\mathbf A, X) &\mapsto \mathbf A \cdot X
\end{align*}
$$
where

$$
    (\mathbf A \cdot X)_i := {\style{transform: rotate(45deg); display: inline-block; font-size: 150%;}{\boxtimes}}_{j=1}^{n} \mathbf A_{ij} \cdot X_j \quad \forall i=1,\ldots,m.
$$
In the notation above, $\left(P_\text{fin}(\mathcal D)\right)^n$ denotes the $n$-fold Cartesian product of $P_\text{fin}(\mathcal D)$ with itself and ${\style{transform: rotate(45deg); display: inline-block; font-size: 150%;}{\boxtimes}}_{j=1}^n$ denotes the Minkowski sum over the sets indexed by $\{1,\ldots,n\}$.
</div>
The following lemma shows that matrix multiplication and the $\mathcal Q$-operator commute:

<div class="statement lemma">
<strong>(Matrix Multiplication, Lemma 5.1.4 in Nazari<d-cite key="nazari2025thesis"></d-cite>) </strong>

Let $X \in \left(P_\text{fin}(\mathcal D)\right)^n$ be a vector of finite sets of dual points and $\mathbf A \in \mathbb R_+^{m, n}$ a matrix with non-negative entries. Then

$$
\begin{equation*}
    \mathbf A \mathcal Q(X) = \mathcal Q(\mathbf A \cdot X).
\end{equation*}
$$
</div>

In order to account for biases, we define how to add a scalar to a set of dual points:
<div class="statement definition">
A scalar can be added to a set of dual points by adding the scalar to the last entry of each point in the set:

$$
\begin{align*}
    \boxplus \colon P_\text{fin}(\mathcal D) \times \mathbb R &\to P_\text{fin}(\mathcal D) \\
    (X, \alpha) &\mapsto X \boxplus \alpha,
\end{align*}
$$
where $X \boxplus \alpha$ is the set

$$
\begin{equation*}
    X \boxplus \alpha := \{(\mathbf x, y + \alpha) | (\mathbf x, y) \in X\}.
\end{equation*}
$$
</div>
It turns out that $\mathcal Q$ is also well behaved with respect to $\boxplus$:

<div class="statement lemma">
<strong>(Lemma 5.1.6 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>
For any finite set of dual points $X \subseteq \mathcal D$ and scalar $\alpha \in \mathbb R$, it holds that

$$
\begin{equation*}
    \mathcal Q(X) + \alpha = \mathcal Q(X \boxplus \alpha).
\end{equation*}
$$
</div>

We are now ready to present the following fundamental proposition that establishes the connection between ReLU networks and differences of piecewise affine functions:

<div class="statement proposition" id="proposition:nn-as-affine-map">
<strong>(Dual Representation, Proposition 5.1.7 in <d-cite key="nazari2025thesis"></d-cite>) </strong>
Assume that a neural network in the sense of <a href="#def:nn" class="cite-stmt hover-link">Definition</a> can, up to layer $l-1$, be written as a DCPA function ${\mathbf a}_{l-1} = \mathcal Q(P_{l-1})- \mathcal Q(N_{l-1})$ for some vectors of finite sets of dual points $P_{l-1}, N_{l-1}$. Then, after writing ${\mathbf W}_l = {\mathbf W}_l^+ - {\mathbf W}_l^-$ using matrices ${\mathbf W}_l^+$ and ${\mathbf W}_l^-$ with non-negative entries, also the network up to the $l$'th layer can be written as a DCPA function

$$
\begin{equation*}
    {\mathbf a}_l = \mathcal Q(P_l) - \mathcal Q(N_l)   
\end{equation*}
$$

with
$$
\begin{align*}
    N_l &= ({\mathbf W}_l^- \cdot P_{l-1}) \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} ({\mathbf W}_l^+ \cdot N_{l-1}) \\
    P_l &= \left( \left(\left({\mathbf W}_l^+ \cdot P_{l-1}) \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} ({\mathbf W}_l^- \cdot N_{l-1}\right)\right) \boxplus {\mathbf b}_l \right) \cup
    \begin{cases}
        N_l \boxplus t_l, & t_l \neq - \infty \\
        \emptyset, & t_l = -\infty.
    \end{cases}
\end{align*}
$$
</div>

The following corollary makes sure <a href="#proposition:nn-as-affine-map" class="cite-stmt hover-link">Proposition</a> can actually be applied to ReLU networks by establishing the base case:

<div class="statement corollary" id="corollary:dual-rep">
Every neural network $\mathcal N$ in the sense of <a href="#def:nn" class="cite-stmt hover-link">Definition</a> can be written as a DCPA function

$$
\begin{equation*}
    \mathcal N = \mathcal Q(P) - \mathcal Q(N)
\end{equation*}
$$
for some vectors of sets of dual points $P, N \subseteq \mathcal D$. We call $(P, N)$ the <em>dual representation</em> of $\mathcal N$.
</div>

The dual representation is denoted by $(P, N)$ since those two sets tell us which points are classified positively and negatively:

<div class="statement proposition">
<strong>(Positive and negative samples, Proposition 5.1.11 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>
Let $\mathcal N = \mathcal Q(P) - \mathcal Q(N)$ be a ReLU binary classification network. Then the following are true for an input $\mathbf x \in \mathbb R^d$:
$$
\begin{align}
    \mathcal N(\mathbf x) \geq 0 &\iff \mathcal Q(P \cup N)(\mathbf x) = \mathcal Q(P)(\mathbf x) \\
    \mathcal N(\mathbf x) \leq 0 &\iff \mathcal Q(P \cup N)(\mathbf x) = \mathcal Q(N)(\mathbf x).
\end{align}
$$

</div>

The following remark summarizes the dual representation and why it is relevant.
<div class="statement remark">
<a href="#proposition:nn-as-affine-map" class="cite-stmt hover-link">Proposition</a> and <a href="#corollary:dual-rep" class="cite-stmt hover-link">Corollary</a> are important tools throughout the rest of this series. We want to use this remark to highlight their significance. Let $\mathcal N \colon \mathbb R^d \to \mathbb R$ be a ReLU network with $L$ layers.

<ol>
<li>

Given the weights of $\mathcal N$, the dual representation provides a symbolic representation $(P_l, N_l)$ of $\mathcal N$ up to layer $l$.
</li>
<li>

It is given by two $n_l$-dimensional vectors $P_l, N_l$ of finite sets of dual points, where $n_l$ is the width of layer $l$. The dual points are $d+1$-dimensional.
</li>
<li>

After each layer $l$, the sets of dual points can be replaced by their upper convex hull (see <a href="#corollary:upper-convex-hull-rep" class="cite-stmt hover-link">Corollary</a>). In particular, for every $i \in \{1,\ldots,n_l\}$, the set $(P_l)_i \subseteq \mathcal D = \mathbb R^{d+1}$ can be replaced by  its upper convex hull vertices $\mathcal U^*((P_l)_i)$. The same holds for $(N_l)_i$.
</li>
<li>

As we will see later, this symbolic representation allows counting the number of affine regions defined by $\mathcal N$. In the case of binary classification, it furthermore allows counting the linear pieces in the decision boundary (Post 2).
</li>
</ol>
</div>
Scattered throughout this series, we employ a running example to highlight the above points.


### Example
In this example, we construct the dual representation of a toy example in two dimensions (see Example 5.2.6 in Nazari<d-cite key="nazari2025thesis"></d-cite>). Throughout this series, we will revisit and use this example to explain various aspect of the discussed duality result.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/example_linear_regions.png" class="img-fluid rounded" zoomable=true caption="<strong>(a)</strong>"%}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/example_decision_boundary.png" class="img-fluid rounded" zoomable=true caption="<strong>(b)</strong>"%}
    </div>
</div>
<div class="caption-left figure">
    Subfigure <strong>(a)</strong> shows the affine regions defined by the network defined in Equation \eqref{eq:tropical-toy-example}. Subfigure <strong>(b)</strong> shows its decision boundary. Negatively classified regions (threshold at $0$) are colored red and positively classified regions are colored blue.
</div>

Specifically, consider the $3$ layer network

$$
\begin{align}
\label{eq:tropical-toy-example}
    \mathcal N \colon \mathbb R^2 &\to \mathbb R \\
    {\mathcal N}(x) &= {\mathbf W}_3\rho_0\left({\mathbf W}_2\rho_0\left({\mathbf W}_1 \mathbf x + {\mathbf b}_1\right) + {\mathbf b}_2\right) + b_3
\end{align}
$$

where

$$
\begin{align*}
    {\mathbf W}_1 &= \begin{pmatrix}
        -1 & -1 \\
        1 & -2
    \end{pmatrix},
    {\mathbf b}_1 = \begin{pmatrix}
        1 \\
        -1
    \end{pmatrix},
    {\mathbf W}_2 = \begin{pmatrix}
        -1 & 2 \\
        2 & -1
    \end{pmatrix},
    {\mathbf b}_2 = \begin{pmatrix}
        1 \\ 2
    \end{pmatrix},
    {\mathbf W}_3 = \begin{pmatrix}
        3, -1
    \end{pmatrix},
    b_3 = 2.
\end{align*}
$$

We use <a href="#proposition:nn-as-affine-map" class="cite-stmt hover-link">Proposition</a> to iteratively construct the dual representation of $\mathcal N$, starting with $P_0 = (\{(1,0,0)\}, \{(0,1,0)\})$ and $N_0 = (\emptyset)$, as in the proof of <a href="#corollary:dual-rep" class="cite-stmt hover-link">Corollary</a>.

After the first layer, the dual representation of $N_1$ can be computed as

$$
\begin{align*}
    N_1 &= {\mathbf W}_1^- P_0 \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} {\mathbf W}_1^+ N_0 = \begin{pmatrix}
        1 & 1 \\
        0 & 2
    \end{pmatrix} P_0
\end{align*}
$$

and thus

$$
\begin{align*}
    (N_1)_1 &= {\style{transform: rotate(45deg); display: inline-block; font-size: 150%;}{\boxtimes}}_{j=1}^2 ({\mathbf W}_1^-)_{1j} (P_0)_j = 1\left\{\left((1,0,0)\right)\right\} \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} 1\left\{\left(0,1,0\right)\right\} = \left\{\left(1,1,0\right)\right\} \\
    (N_1)_2 &= {\style{transform: rotate(45deg); display: inline-block; font-size: 150%;}{\boxtimes}}_{j=1}^2 ({\mathbf W}_1^-)_{2j} (P_0)_j = 0\left\{\left((1,0,0)\right)\right\} \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} 2\left\{\left(0,1,0\right)\right\} = \left\{\left(0,2,0\right)\right\},
\end{align*}
$$

which implies

$$
\begin{equation*}
    N_1 = \left(\left\{\left(1,1,0\right)\right\}, \left\{\left(0,2,0\right)\right\}\right).
\end{equation*}
$$

Similarly,

$$
\begin{align*}
    P_1 &= {\mathbf W}_1^+ P_0 \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} {\mathbf W}_1^- N_0 \boxplus {\mathbf b}_1 \cup N_1 = \begin{pmatrix}
        0 & 0 \\
        1 & 0
    \end{pmatrix} P_0 \boxplus {\mathbf b}_1 \cup N_1
\end{align*}
$$

and thus

$$
\begin{align*}
    (P_1)_1 &= {\style{transform: rotate(45deg); display: inline-block; font-size: 150%;}{\boxtimes}}_{j=1}^{2} ({\mathbf W}_1^+)_{1j} (P_0)_j \boxplus ({\mathbf b}_1)_1 \cup (N_1)_1 = 0\left\{\left((1,0,0)\right)\right\} \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} 0\left\{\left(0,1,0\right)\right\} \boxplus 1 \cup (N_1)_1 \\
    &= \left\{\left(0,0,1\right), \left(1,1,0\right)\right\} \\
    (P_1)_2 &= {\style{transform: rotate(45deg); display: inline-block; font-size: 150%;}{\boxtimes}}_{j=1}^{2} ({\mathbf W}_1^+)_{2j} (P_0)_j \boxplus ({\mathbf b}_1)_2 \cup (N_1)_2 = 1\left\{\left((1,0,-1)\right)\right\} \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} 0\left\{\left(0,1,0\right)\right\} \boxplus -1 \cup (N_1)_2 \\
    &= \left\{\left(1,0,0\right), \left(0, 2, 0\right)\right\},
\end{align*}
$$

which implies

$$
\begin{equation*}
    P_1 = \left(\left\{\left(0,0,1\right),\left(1,1,0\right)\right\}, \left\{\left(0,2,0\right),\left(1,0,-1\right)\right\}\right).
\end{equation*}
$$

After repeating these steps for layer $2$ and $3$ (with a slight adaptation for the last linear layer as in <a href="#proposition:nn-as-affine-map" class="cite-stmt hover-link">Proposition</a>), one arrives at the following final dual representation of $\mathcal N = \mathcal Q(P) - \mathcal Q(N)$:

$$
\begin{align*}
    N &= \left\{(3, 17, 4), (2, 16, 5), (5, 19, 2), (3, 14,2), (2,16,3),(5,19,0),(6,17,-1),(0,14,7)\right\} \\
    P &= \left\{(2,16,5),(5,19,2),(5,19,5),(11,7,-1),(12,5,-2),(3,14,4),(6,17,1),(6,17,4)\right\}.
\end{align*}
$$

Here, $N := N_3$ and $P := P_3$. Additionally, note that we have identified the one-dimensional vectors of sets $N_3$ and $P_3$ with their only entry.

By <a href="#corollary:upper-convex-hull-rep" class="cite-stmt hover-link">Corollary</a>, the CPA functions $\mathcal Q(N)$ and $\mathcal Q(P)$ are uniquely identified by the upper convex hulls of $N$ and $P$, i.e.,

$$
\begin{equation*}
    \mathcal Q(N) = \mathcal Q(\mathcal U^*(N)), \quad \mathcal Q(P) = \mathcal Q(\mathcal U^*(P)).
\end{equation*}
$$

This allows restricting our attention to subsets of $N$ and $P$. Specifically, the upper convex hull points can be determined\footnote{We use SciPy to do so, see the code for more details.} as

$$
\begin{align*}
    \mathcal U^*(N) &= \left\{(5, 19, 2), (3, 14, 2), (6, 17, -1), (0, 14, 7)\right\} \\
    \mathcal U^*(P) &= \left\{(2,16,5), (3,14,4), (5,19,5), (12,5,-2)\right\}.
\end{align*}
$$

<a href="#fig:tropical-toy-example" class="cite-fig hover-link">Figure</a> contains a plot of the dual representation of this toy example, as well as the upper convex hulls.

<div class="row mt-3 w-50 mx-auto">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/example_dual.svg" class="img-fluid rounded" zoomable=true%}
    </div>
</div>
<div class="caption-left figure" id="fig:tropical-toy-example">
    Dual representation of the two-dimensional toy-example defined in Equation \eqref{eq:tropical-toy-example}. Red points correspond to $N$, blue points are $P$. The red polygon is $\mathcal U(N)$, the blue polygon is $\mathcal U(P)$. Note that, in theory, both $\mathcal U(N)$ and $\mathcal U(P)$ are polyhedral complexes, i.e., they can consist of multiple facets.
</div>

<!--
---

## Notation
<p>The following table contains a (non-exhaustive) selection of the most frequently used notation. We provide links to formal definitions for non-generic symbols.</p>
<table>
    <tr>
        <th class="notation">Notation</th>
        <th class="description">Description</th>
    </tr>
    <tr>
        <td>\(\mathbb{N}\)</td>
        <td>The set of natural numbers \(\{1, 2, 3, \dots\}\)</td>
    </tr>
    <tr>
        <td>\(\mathbb{N}_0\)</td>
        <td>The set of natural numbers including \(0\)</td>
    </tr>
    <tr>
        <td>\([m:n]\)</td>
        <td>The set \(\{m, m+1, \ldots, n\}\) for \(m,n \in \mathbb{N}_0\)</td>
    </tr>
    <tr>
        <td>\(\mathbf{v}\)</td>
        <td>Multi-dimensional vector</td>
    </tr>
    <tr>
        <td>\(\mathbf{v}^T\)</td>
        <td>Transpose of a vector \(\bm{v}\)</td>
    </tr>
    <tr>
        <td>\((\mathbf{x}, y)\)</td>
        <td>Point in \(\mathbb{R}^{d+1}\) with \(\bm{x} \in \mathbb{R}^d\) and \(y \in \mathbb{R}\)</td>
    </tr>
    <tr>
        <td>\(\langle \cdot, \cdot \rangle\)</td>
        <td>Euclidean inner product</td>
    </tr>
    <tr>
        <td>\(\|\cdot\|\)</td>
        <td>Norm of a vector or function</td>
    </tr>
    <tr>
        <td>\(\|\cdot\|_2\)</td>
        <td>Euclidean norm</td>
    </tr>
    <tr>
        <td>\(\xrightarrow{\sim}\)</td>
        <td>Bijection</td>
    </tr>
    <tr>
        <td>\(|\cdot|\)</td>
        <td>Cardinality of a set</td>
    </tr>
    <tr>
        <td>\(\mathbf{A}\)</td>
        <td>Matrix</td>
    </tr>
    <tr>
        <td>\(\mathbf{A}_{i:}\)</td>
        <td>\(i\)'th row of matrix \(\mathbf{A}\)</td>
    </tr>
    <tr>
        <td>\(x^+\)</td>
        <td>\(\max(0, x)\)</td>
    </tr>
    <tr>
        <td>\(x^-\)</td>
        <td>\(\max(0, -x)\)</td>
    </tr>
    <tr>
        <td>\(\mathcal{N}(\mu, \sigma^2)\)</td>
        <td>Gaussian distribution</td>
    </tr>
    <tr>
        <td>\(H_n\)</td>
        <td>\(n\)'th Harmonic number \(\sum_{i=1}^k \frac{1}{k}\)</td>
    </tr>
    <tr>
        <td>\(f \sim g\)</td>
        <td>The functions \(f\) and \(g\) are asymptotically equivalent, \(\lim_{n \to \infty} \frac{f(n)}{g(n)} = 1\)</td>
    </tr>
    <tr>
        <td>\(g = \mathcal{O}(f)\)</td>
        <td>There exist constants \(C > 0\) and \(N \in \mathbb{N}\) such that \(|g(n)| \leq C |f(n)|\) for all \(n \geq N\)</td>
    </tr>
    <tr>
        <td>\(g = \Theta(n)\)</td>
        <td>There exist constants \(c_1, c_2 > 0\) and \(N \in \mathbb{N}\) such that \(c_1 f(n) \leq g(n) \leq c_2 f(n)\) for all \(n \geq N\)</td>
    </tr>
    <tr>
        <td>\(\sqcup\)</td>
        <td>Disjoint union</td>
    </tr>
    <tr>
        <td>\(\boxplus\)</td>
        <td>Sum of a scalar and a set of vectors, Definition~\ref{def:boxplus}</td>
    </tr>
    <tr>
        <td>\(\style{transform: rotate(45deg); display: inline-block;}{\boxtimes}\)</td>
        <td>Minkowski sum</td>
    </tr>
    <tr>
        <td>\(f_{\mathbf{a}, b}\)</td>
        <td>Affine map \(f_{\mathbf{a}, b}(\mathbf{x}) = \langle \mathbf{a}, \mathbf{x} \rangle + b\)</td>
    </tr>
    <tr>
        <td>\(\rho_t(\cdot)\)</td>
        <td>The function \(\max(\cdot, t)\)</td>
    </tr>
    <tr>
        <td>\(\mathcal N\)</td>
        <td>Fully connected ReLU network, see Definition~\ref{def:nn}</td>
    </tr>
    <tr>
        <td>\(\mathfrak{c}\)</td>
        <td>Complexity measure for neural network \(\mathcal N\), for example number of affine regions or linear pieces in decision boundary</td>
    </tr>
    <tr>
        <td>\(\mathcal{B}\)</td>
        <td>Decision boundary of a binary classification network, Definition~\ref{def:decision-boundary}</td>
    </tr>
    <tr>
        <td>\(L\)</td>
        <td>Depth of neural network</td>
    </tr>
    <tr>
        <td>\(d\)</td>
        <td>Input dimension for neural network</td>
    </tr>
    <tr>
        <td>\(n_i\)</td>
        <td>Width of layer \(i=1,\ldots,L\) of neural network</td>
    </tr>
    <tr>
        <td>\(\nabla\)</td>
        <td>Gradient operator</td>
    </tr>
    <tr>
        <td>\((\mathbf{x}, y) \in f\)</td>
        <td>\((\mathbf{x}, y)\) lies on the graph of \(f\), Definition~\ref{def:point-function}</td>
    </tr>
    <tr>
        <td>\((\mathbf{x}, y) \succ f\)</td>
        <td>\((\mathbf{x}, y)\) lies above the graph of \(f\), Definition~\ref{def:point-function}</td>
    </tr>
    <tr>
        <td>\(\text{affhul}(X)\)</td>
        <td>Affine hull of \(X\), the smallest affine space containing \(X\)</td>
    </tr>
    <tr>
        <td>\(\mathcal{C}(X)\)</td>
        <td>Convex hull of \(X\)</td>
    </tr>
    <tr>
        <td>\(\mathcal{U}(X)\)</td>
        <td>Upper convex hull of \(X\), Definition~\ref{def:upper-convex-hull}</td>
    </tr>
    <tr>
        <td>\(\mathcal{U}_k(X)\)</td>
        <td>\(k\)-skeleton of \(\mathcal{U}(X)\), Definition~\ref{def:upper-convex-hull}</td>
    </tr>
    <tr>
        <td>\(\mathcal{U}^*(X)\)</td>
        <td>Upper convex hull vertices of \(X\), Definition~\ref{def:upper-convex-hull}</td>
    </tr>
    <tr>
        <td>\(f \| \mathcal{U}(X)\)</td>
        <td>Affine function \(f\) is tangent to the upper convex hull of \(X\), Definition~\ref{def:tangent-to-uch}</td>
    </tr>
    <tr>
        <td>CPA</td>
        <td>Convex and piecewise affine function</td>
    </tr>
    <tr>
        <td>DCPA</td>
        <td>Difference of convex and piecewise affine functions</td>
    </tr>
    <tr>
        <td>CPA($d$)</td>
        <td>Set of CPA functions \(\mathbb{R}^d \to \mathbb{R}\)</td>
    </tr>
    <tr>
        <td>DCPA($d$)</td>
        <td>Set of DCPA functions \(\mathbb{R}^d \to \mathbb{R}\)</td>
    </tr>
    <tr>
        <td>\(\mathcal{Q}(S)\)</td>
        <td>CPA function induced by a set \(S\) of dual points, Definition~\ref{def:q}</td>
    </tr>
    <tr>
        <td>\(\mathcal{T}(F)\)</td>
        <td>Tessellation induced by CPA function \(F\), Definition~\ref{def:tessellation}</td>
    </tr>
    <tr>
        <td>\(\mathcal{T}_k(F)\)</td>
        <td>\(k\)-skeleton of \(\mathcal{T}(F)\), Definition~\ref{def:k-tessellation}</td>
    </tr>
    <tr>
        <td>\(\mathcal{T}(S)\)</td>
        <td>Tessellation induced by a CPA function \(\mathcal{Q}(S)\)</td>
    </tr>
    <tr>
        <td>\(\mathcal{T}(P,N)\)</td>
        <td>Tessellation induced by the DCPA function \(\mathcal{Q}(P) - \mathcal{Q}(N)\), Definition~\ref{def:tessellation-refinement}</td>
    </tr>
    <tr>
        <td>\((P_l, N_l)\)</td>
        <td>Dual representation of \(\mathcal{Q}(P_L) - \mathcal{Q}(N_L)\) up to layer \(l\), Corollary~\ref{corollary:dual-rep}</td>
    </tr>
    <tr>
        <td>\(\sigma\)</td>
        <td>Cell in \(\mathcal{T}(F)\)</td>
    </tr>
    <tr>
        <td>\(\eta\)</td>
        <td>Face in \(\mathcal{Q}(S)\)</td>
    </tr>
    <tr>
        <td>\(\mathfrak{P}(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)\)</td>
        <td>Set of paths of dual points, Definition~\ref{def:path-of-dual-points}</td>
    </tr>
    <tr>
        <td>\(\mathfrak{P}(P, N)\)</td>
        <td>Set of paths of \(d\)-cells, Definition~\ref{def:path-of-cells}</td>
    </tr>
    <tr>
        <td>\(|\Sigma|\)</td>
        <td>Support of a polyhedral complex \(\Sigma\), Definition~\ref{def:polyhedral-complex}</td>
    </tr>
    <tr>
        <td>\(\mathfrak R\)</td>
        <td>Real space, Page~\pageref{sec:affine-dualities}</td>
    </tr>
    <tr>
        <td>\(\mathfrak D\)</td>
        <td>Dual space, Page~\pageref{sec:affine-dualities}</td>
    </tr>
    <tr>
        <td>\(\text{Aff}_{\mathfrak R}(d)\)</td>
        <td>Real affine space, Page~\pageref{sec:affine-dualities}</td>
    </tr>
    <tr>
        <td>\(\text{Aff}_{\mathfrak D}(d)\)</td>
        <td>Dual affine space, Page~\pageref{sec:affine-dualities}</td>
    </tr>
    <tr>
        <td>\(\mathcal{R}\)</td>
        <td>Bijection between \(\mathcal D\) and \(\text{Aff}_{\mathfrak R}(d)\), Lemma~\ref{lemma:aff-real-iso}</td>
    </tr>
    <tr>
        <td>\(\check{\mathcal{R}}\)</td>
        <td>Bijection between \(\mathcal R\) and \(\text{Aff}_{\mathfrak D}(d)\), Lemma~\ref{lemma:aff-dual-iso}</td>
    </tr>
    <tr>
        <td>\(S_I\)</td>
        <td>Given an index-set \(I\) and an indexed set \(S\), \(S_I := \{s_i \mid i \in I\}\)</td>
    </tr>
    <tr>
        <td>\(\mathbf{A}_=\)</td>
        <td>Set of implicit equality constraints of a polynomial \(\{\mathbf{A} \mathbf{x} \geq \mathbf{b}\}\), Definition~\ref{def:decompose-implicit-equality}</td>
    </tr>
    <tr>
        <td>\(\mathbf{A}^{\sigma}\)</td>
        <td>Inequality constraints defining a cell \(\sigma\), Remark~\ref{remark:cells-as-systems}</td>
    </tr>
    <tr>
        <td>\(f_m\)</td>
        <td>Mirror-map, Definition~\ref{def:mirrow-map}</td>
    </tr>
</table>
-->