---
layout: distill
title: Dual Complexity Measures for ReLU Networks
description: In this post, we introduce dual complexity measures for fully connected feedforward ReLU networks. This is part two of a three part series on the geometry of generalization of deep neural networks.
tags: Deep-Learning Geometry Generalization
giscus_comments: true
date: 2025-06-04
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
  - name: Tessellations
    subsections:
      - name: Example
  - name: Decision Boundary
    subsections:
      - name: Characterizing $k$-cells
      - name: Application to the Decision Boundary
      - name: Example
  - name: Affine Regions
    subsections:
      - name: Refinements
      - name: Characterizing $k$-cells, Part II
      - name: Counting Affine Regions
      - name: In Practice
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

  .theorem::before {
      content: "Theorem " counter(statement-counter);
      font-weight: bold;
  }

  .conjecture::before {
      content: "Conjecture " counter(statement-counter);
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

# Dual Complexity Measures

This is part two of a three part series on the geometry of generalization, which is the essence of my [master thesis](/assets/pdf/Master_Thesis.pdf)<d-cite key="nazari2025thesis"></d-cite>. In this series, we will present a novel perspective on generalization of overparameterized networks.

The series is structured in the following way:

<ul>
  <li>In <a href="{% post_url 2025-06-02-dual-representation %}">part one</a>, we established a dual representation of fully connected feedforward ReLU networks.</li>
  <li>In <a href="{% post_url 2025-06-04-dual-complexity-measures %}">this post</a>, we show how this dual representation can be used to derive complexity measures for these networks.</li>
  <li>In part three (coming soon!!), we will use these complexity measures to find evidence for the volume hypothesis<d-cite key="chiang2022loss"></d-cite>, an approach to explain why overparameterized models generalize well.</li>
</ul>

## Introduction

The constructions put forward in this post are inspired by Piwek et al.<d-cite key="piwek2023exact"></d-cite>. In particular, we utilize the dual representation introduced in <a href="{% post_url 2025-06-02-dual-representation %}">part one</a> to assess the complexity of ReLU networks.

As complexity measures, we use the number of affine regions in the setting of regression (as in <d-cite key="montufar2014number, zhang2018tropical, raghu2017expressive, pascanu2013number"></d-cite>) and the number of linear pieces in the decision boundary in the setting of binary classification (as in <d-cite key="piwek2023exact"></d-cite>). As we will show, these two complexity measures can be directly tied to the dual representation: the number of linear pieces in the decision boundary corresponds to the number of a specific kind of edge in an upper convex hull (<a href="#corollary:nn-boundary-dual" class="cite-stmt hover-link">Corollary</a>), and the number of affine regions corresponds to the number of vertices in another upper convex hull (<a href="#corollary:affine-pieces-mod" class="cite-stmt hover-link">Corollary</a>).

Finally, for every claim made in this series, I will refer to the corresponding statement in the thesis. There, one can find a proof as well as a reference to the corresponding statement by Piwek et al.<d-cite key="piwek2023exact"></d-cite> wherever appropriate.

## Tessellations

CPA functions (see <a href="{% post_url 2025-06-02-dual-representation %}/#affine-and-cpa-functions">this section</a> of the previous post) induce a <em>tessellation</em> of $\mathbb R^d$. It plays an important role in understanding ReLU networks:
<div class="statement definition" id="def:tessellation">
<strong>(Tessellations)</strong>
    Given a CPA function $F(\mathbf x) := \max\{f_1(\mathbf x),\ldots,f_n(\mathbf x)\}$, a <em>cell</em> induced by $F$ is
    
    $$
    \begin{equation*}
        \{\mathbf x \in \mathbb R^d \;|\; f_i(\mathbf x) = f_{i'}(\mathbf x) \geq f_j(\mathbf x) \; \text{for all } i,i' \in I, j \in J\},
    \end{equation*}
    $$

    where $I, J$ are disjoint sets whose union is $\{1,2,...,n\}$.
    The set of all cells induced $F$ is called the <em>tessellation</em> induced by $F$ and denoted by $\mathcal T(F)$.
</div>

<a href="#fig:tessellation" class="cite-fig hover-link">Figure</a> contains an example of a tessellation. By a slight abuse of notation, we will write $\mathcal T(S)$ for the tessellation induced by the CPA function $\mathcal Q(S)$ for any set of points $S$.

The following lemma establishes a connection between tessellations and polyhedral complexes:
<div class="statement lemma">
<strong>(Lemma 3.4.2 in Nazari<d-cite key="nazari2025key"></d-cite>)</strong>
The tessellation induced by a CPA function $F$ forms a polyhedral complex.
</div>

This lemma tells us that we may think of a tessellation as a polyhedral complex and thus the following definition makes sense:
<div class="statement definition">
    Let $F$ be a CPA function. We denote by $\mathcal T_k(F)$ the $k$-skeleton of the tessellation induced by $F$. The support of $\mathcal T_{d-1}(F)$ is also called an <em>affine (or tropical) hypersurface</em>.
    % The $d$-cells are called <em>affine regions</em>, since they are the largest connected sets on which $F$ is an affine function.
</div>

### Example
As an example, <a href="#fig:tessellation" class="cite-fig hover-link">Figure</a> shows the tessellation induced by the CPA function
\begin{equation}
\label{eq:example-tessellation}
    f \colon \mathbb R^2 \to \mathbb R, \quad (x,y) \mapsto \max \{1+2x, 1+2y, 2+x+y, 2+x, 2+y, 2\}.
\end{equation}
The blue lines correspond to points on which two affine functions agree and are larger than the others. They form the $1$-skeleton of the tessellation. The intersections of these lines are the $0$-cells. On each of the white convex regions (the $2$-cells) $f$ is affine. 

<a href="#fig:tessellation" class="cite-fig hover-link">Figure</a> also illustrates how the tesselation forms a polyhedral complex: the face of any polyhedron is again a polyhedron (for example, the faces of the white convex regions are the $1$-cells), and the intersection of any two polyhedra is either empty or again a face.

<div class="row mt-3 w-50 mx-auto figure-content" id="fig:tessellation">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/tessellation.png" class="img-fluid rounded" zoomable=true%}
    </div>
</div>
<div class="caption-left figure">
  Figure 1 in <d-cite key="zhang2018tropical"></d-cite>. Example of a tessellation, induced by the DCPA-function given in Equation \eqref{eq:example-tessellation}.
</div>


## Decision Boundary

We saw in <a href="{% post_url 2025-06-02-dual-representation %}">the previous post</a> how we can identify neural networks as DCPA functions using affine geometry. In this section, we use this result to characterize the decision boundary of ReLU binary classification networks. This will eventually allow counting the linear pieces inside the decision boundary.

Throughout this section, let $S \subseteq \mathcal D$ be a set of dual points whose upper convex hull has vertices
$$
\mathcal U^*(S) = \{s_1,\ldots,s_n\} = \{(\mathbf a_1,b_1),\ldots,(\mathbf a_n,b_n)\}.
$$
Furthermore, given a set of indices $I \subseteq \{1,\ldots,n\}$, we introduce the short-hand notation
$$
S_I := \{s_i \mid i \in I\}
$$
for the subset of $S$ indexed by $I$.


### Characterizing $k$-cells

<!-- Towards quantifying the decision boundary, we will establish in Theorem <em>see below</em> a bijection between the $k$-cells in $\mathcal T(S)$ and the $(d-k)$-faces in $\mathcal U(S)$. In Section <em>see below</em>, we will use this result to translate the decision boundary to dual space, where its complexity is easier to quantify. -->

Let $\sigma \in \mathcal T(S)$ be a cell in the tessellation induced by $S$. Using the definition and the fact that CPA functions are uniquely characterized by their upper convex hulls (see see <a href="{% post_url 2025-06-02-dual-representation %}/#cpa-functions-as-upper-convex-hulls">this section</a> of the previous post), one can quickly confirm that $\sigma$ is the solution of a system of linear inequalities and equalities:

$$
\begin{align}
\label{eq:system-zero}
\left\{
\begin{aligned}
\mathcal R(s_i)(\mathbf x) &= \mathcal R(s_{i'})(\mathbf x) \quad \forall\, i,i' \in I^{\sigma}_= \\
\mathcal R(s_i)(\mathbf x) &\ge \mathcal R(s_j)(\mathbf x) \quad \forall\, i \in I_=^{\sigma},\, j \in I^{\sigma}_+,
\end{aligned}
\right.
\end{align}
$$

<div>
where $I^{\sigma}_{=}$ and $I^{\sigma}_{+}$ form a disjoint partition of $\{1,2,\ldots,n\}$. W.l.o.g., this partition can be chosen in such a way that no index can be moved from $I^{\sigma}_+$ to $I^{\sigma}_=$ without altering the solution space.
</div>

Using that $\mathcal R(s_i)(\mathbf x) := \langle \mathbf a_i, \mathbf x \rangle + b_i$ for all $s_i = (\mathbf a_i, b_i) \in S$, System \eqref{eq:system-zero} can be re-written as a system of linear inequalities:

<div class="statement definition" id="def:cells-as-systems">
<strong>(Cells as System of Linear Inequalities)</strong>  
Any cell $\sigma \in \mathcal T(S)$ can be written as a system of linear inequalities and equalities $\sigma = \{\mathbf A^{\sigma}_= \mathbf x = \mathbf b^{\sigma}_=\} \cap \{\mathbf A^{\sigma}_+ \mathbf x \ge \mathbf b^{\sigma}_+\}$ in the following way.

Fix a dual point $s_{k_{\sigma}} \in S_{I^{\sigma}_=}$.  
The matrix $\mathbf A^{\sigma}_= \in \mathbb R^{|I^{\sigma}_=| \times d}$ containing the equality constraints has as its rows the vectors $\bigl(\mathbf a_{k_{\sigma}} - \mathbf a_i \mid i \in I^{\sigma}_=\bigr)$ and the corresponding vector $\mathbf b^{\sigma}_= \in \mathbb R^{|I^{\sigma}_=|}$ has entries $\bigl(b_i - b_{k_{\sigma}} \mid i \in I^{\sigma}_=\bigr)$ (in the same order).

Similarly, the matrix $\mathbf A^{\sigma}_+ \in \mathbb R^{|I^{\sigma}_+| \times d}$ containing the inequality constraints has as its rows the vectors $\bigl(\mathbf a_{k_{\sigma}} - \mathbf a_i \mid i \in I^{\sigma}_+\bigr)$ and the corresponding vector $\mathbf b^{\sigma}_+ \in \mathbb R^{|I^{\sigma}_+|}$ has entries $\bigl(b_i - b_{k_{\sigma}} \mid i \in I^{\sigma}_+\bigr)$.
</div>

<div class="statement remark" id="remark:cells-as-systems">
<strong>(Remark 6.1.2 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
The joint system of linear equalities and linear inequalities in <a href="#def:cells-as-systems" class="cite-stmt hover-link">Definition</a> can be translated to a system of just inequalities

$$
\sigma = \{\mathbf A^{\sigma} \mathbf x \ge \mathbf b^{\sigma}\} = \{\mathbf A^{\sigma}_= \mathbf x = \mathbf b^{\sigma}_=\} \cap \{\mathbf A^{\sigma}_+ \mathbf x \ge \mathbf b^{\sigma}_+\}
$$

by rewriting every equality as two inequalities. The resulting matrix $\mathbf A^{\sigma} \in \mathbb R^{2|I_=^{\sigma}| \times d}$ contains the rows of $\mathbf A^{\sigma}_=$, as well as their negatives, and the rows of $\mathbf A^{\sigma}_+$. The vector $\mathbf b^{\sigma} \in \mathbb R^{2|I_=^{\sigma}|}$ can be constructed analogously.
</div>

Using this representation of $\sigma$ as a system of linear inequalities, the following proposition describes the dimension of $\sigma$:

<div class="statement proposition" id="proposition:dim-cell-rank">
<strong>(Proposition 6.1.3 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $\sigma \in \mathcal T(S)$ be a cell in the tessellation induced by $S$. Then

$$
\dim \sigma \;=\; d \;-\; \mathrm{rank}\,\mathbf A^{\sigma}_=.
$$
</div>

As a next step, we aim to understand how the cell $\sigma$ looks in dual space. We start with the following proposition:

<div class="statement proposition" id="proposition:dim-face-rank">
<strong>(Proposition 6.1.4 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $\sigma \in \mathcal T(S)$ be a cell. Then the convex hull $\mathcal C(S_{I_=^{\sigma}})$ is a face in $\mathcal U(S)$ of dimension

$$
\dim \mathcal C\bigl(S_{I_=^{\sigma}}\bigr) \;=\; \mathrm{rank}\,\mathbf A_=^{\sigma}.
$$
</div>

The last proposition tells us that $\mathcal C(S_{I_=^{\sigma}})$ is a face in $\mathcal U(S)$. The next proposition uses the face $\mathcal C(S_{I_=^{\sigma}})$ to explain how $\sigma$ translates to dual space. But first, we define what it means for an affine function to be tangent to an upper convex hull:

<div class="statement definition" id="def:tangent-to-uch">
<strong>(Tangent Affine Function)</strong>  
Given an affine function $f \colon \mathbb R^d \to \mathbb R$, we say that $f$ is <em>tangent</em> to the upper convex hull $\mathcal U(S)$ if
<ol style="i">
  <li>$f$ lies above $\mathcal U(S)$, i.e., $f \succeq \mathcal U(S)$.</li>
  <li>The graph of $f$ intersects the upper convex hull, i.e., $\mathrm{graph}(f) \cap \mathcal U(S) \neq \emptyset$.</li>
</ol>
In this case, we write $f \;\|\; \mathcal U(S)$.
</div>

<div class="statement proposition" id="proposition:one-cell-one-plane">
<strong>(Proposition 6.1.6 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $\sigma \in \mathcal T(S)$ be a cell in the tessellation induced by $S$. Then there is a one-to-one correspondence between points in $\sigma$ and dual planes tangent to the upper convex hull of $S$ which contain the face $\mathcal C(S_{I_=^{\sigma}})$:

$$
\{\mathbf x \in \sigma\} \;\longleftrightarrow\; \{\,f \in \text{Aff}_{\mathfrak D}(d) \mid f \| \mathcal U(S) \text{ and } f \supseteq \mathcal C(S_{I_=^{\sigma}})\}.
$$
Similarly, every face in $\mathcal U(S)$ defines a cell in this way.
</div>

By <a href="#proposition:dim-face-rank" class="cite-stmt hover-link">Proposition</a> and <a href="#proposition:one-cell-one-plane" class="cite-stmt hover-link">Proposition</a>, cells in the tessellation induced by $S$ are closely related to faces in the upper convex hull of $S$. The following theorem makes this relationship precise:

<div class="statement theorem" id="theorem:cell-face-bijection">
<strong>(Theorem 6.1.7 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
There exists a one-to-one correspondence between $k$-cells in $\mathcal T(S)$ and $(d-k)$-faces in $\mathcal U(S)$. Specifically, the following map is a bijection:

$$
\begin{aligned}
\Phi \colon \mathcal T_k(S) &\;\xrightarrow{\sim}\; \mathcal U_{\,d-k}(S) \\
\sigma &\;\mapsto\; \mathcal C\bigl(S_{I_=^{\sigma}}\bigr).
\end{aligned}
$$
</div>

### Application to the Decision Boundary

In this section, we use the bijection from <a href="#theorem:cell-face-bijection" class="cite-stmt hover-link">Theorem</a> to characterize the decision boundary of a ReLU binary classification network $\mathcal N = \mathcal Q(P) - \mathcal Q(N) \colon \mathbb R^d \to \mathbb R$.

As a quick reminder, the network’s decision boundary is given by
$$
\mathcal B \;=\; \bigl(\mathcal Q(P) - \mathcal Q(N)\bigr)^{-1}(0).
$$
Consequently, we are interested in studying zero-sets of DCPA functions. We start with a special case.

<div class="statement proposition" id="proposition:linear-pieces-special">
<strong>(Decision Boundary I, Proposition 6.2.1 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $F = \mathcal Q(P)$ and $G = \mathcal Q(N)$ be CPA functions $\mathbb R^d \to \mathbb R$ for some finite sets of dual points $P, N \subseteq \mathcal D$. Assume that no point of $P$ lies on $\mathcal U(N)$ and vice versa. Let $D$ be the zero-set of $F - G$. Then $D$ is the union of precisely those $(d-1)$-cells of $\mathcal T(P \cup N)$ which (in the sense of <a href="#theorem:cell-face-bijection" class="cite-stmt hover-link">Theorem</a>) correspond to edges (i.e., $1$-faces) of $\mathcal U(P \cup N)$ with one end in $P$ and the other end in $N$.
</div>

<a href="#proposition:linear-pieces-special" class="cite-stmt hover-link">Proposition</a> handles the special case that $P \cap \mathcal U(N) = N \cap \mathcal U(P) = \emptyset$. The following proposition handles the general case.

<div class="statement proposition" id="proposition:linear-pieces-general">
<strong>(Decision Boundary II, Proposition 6.2.2 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $F = \mathcal Q(P)$ and $G = \mathcal Q(N)$ be CPA functions $\mathbb R^d \to \mathbb R$ for some finite sets of dual points $P, N \subseteq \mathcal D$. Let $D$ be the zero-set of $F - G$. Then $D$ is the union of precisely those $(d-1)$-cells of $\mathcal T(P \cup N)$ which (in the sense of <a href="#theorem:cell-face-bijection" class="cite-stmt hover-link">Theorem</a>) correspond to edges of $\mathcal U(P \cup N)$ containing points from both $P$ and $N$.
</div>

For completeness, the following corollary applies these findings to neural networks.

<div class="statement corollary" id="corollary:nn-boundary-dual">
<strong>(Corollary 6.2.3 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $\mathcal Q(P) - \mathcal Q(N) \colon \mathbb R^d \to \mathbb R$ be a ReLU binary classification network. Then the number of linear pieces in the decision boundary of $\mathcal N$ equals the number of edges in $\mathcal U(P \cup N)$ containing points from both $P$ and $N$. 
</div>

<div class="row mt-3 w-50 mx-auto figure-content" id="fig:example-surfaces">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/example-dual-rep.svg" class="img-fluid rounded" zoomable=true%}
    </div>
</div>
<div class="caption-left figure">
    An example of points $(P,N)$ (points in $N$ are red, points in $P$ are blue), defining a ReLU network $\mathcal N = \mathcal Q(P) - \mathcal Q(N) \colon \mathbb R^2 \to \mathbb R$. There are four edges (light blue) contributing to the decision boundary of $\mathcal N$, since they contain both red and blue points. However, only three of them start and end in different colors.
</div>

Some thoughts on the difference between <a href="#proposition:linear-pieces-special" class="cite-stmt hover-link">Proposition</a> and <a href="#proposition:linear-pieces-general" class="cite-stmt hover-link">Proposition</a> can be found in Remark 6.2.4 in Nazari<d-cite key="nazari2025thesis"></d-cite>.

### Example

In this subsection, we continue the toy-example from <a href="{% post_url 2025-06-02-dual-representation %}/#example">the previous post</a> (see Example 5.2.6 in Nazari<d-cite key="nazari2025thesis"></d-cite>). By <a href="#corollary:nn-boundary-dual" class="cite-stmt hover-link">Corollary</a>, the number of linear pieces in its decision boundary is the same as the number of edges in $\mathcal U(P \cup N)$ containing points from both $P$ and $N$.

Specifically, one can compute
$$
\mathcal U^*(P \cup N) = \{(5,19,5),\,(0,14,7),\,(12,5,-2)\}
$$
<div>
where $(5,19,5),(12,5,-2) \in \mathcal U^*(P)$ and $(0,14,7) \in \mathcal U^*(N)$ (see <a href="#fig:tropical-toy-example-union" class="cite-fig hover-link">Figure</a>). Thus, there are three edges in $\mathcal U(P \cup N)$, two of which contribute to the network’s decision boundary since they contain vertices from both $P$ and $N$. This confirms Figure <em>see caption below</em>, which shows the decision boundary of $\mathcal N$ and confirms that, indeed, it consists of two linear pieces.
</div>

<div class="row mt-3 w-50 mx-auto figure-content" id="fig:tropical-toy-example-union">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/running-example-uch.svg" class="img-fluid rounded" zoomable=true%}
    </div>
</div>
<div class="caption-left figure">
    Two-dimensional toy-example defined in <a href="{% post_url 2025-06-02-dual-representation %}">part one</a>. Red points correspond to $N$, blue points are $P$. The green polygon is $\mathcal U(P \cup N)$. Note that, in theory, $\mathcal U(P \cup N)$ and is a polyhedral complex, i.e., it can consist of multiple facets. Note also how there are red and blue points in $\mathcal U^*(P \cup N)$, ultimately contributing to the decision boundary.
</div>




## Affine Regions

In the previous section, we used the upper convex hull of $P \cup N$ to characterize the decision boundary of a ReLU binary classification network $\mathcal Q(P) - \mathcal Q(N)$. In this section, we take a similar approach to characterize the network's affine regions.


### Refinements

So far, we have studied what it means for a CPA function to induce a tessellation of $\mathbb R^d$. The following definition clarifies what it means for a DCPA function to do so.

<div class="statement definition" id="def:tessellation-refinement">
<strong>(Tessellation Induced by a DCPA Function)</strong>  
Let $F = \mathcal Q(P) - \mathcal Q(N)$ be a DCPA function. We then define the tessellation $\mathcal T(P,N)$ induced by $F$ to consist of all non-empty pairwise intersections of cells induced by $P$ and $N$, i.e.

$$
\mathcal T(P, N) := \{\sigma \cap \sigma' \mid \sigma \in \mathcal T(P),\; \sigma' \in \mathcal T(N)\} \setminus \emptyset.
$$
</div>

As it turns out, $\mathcal T(P,N)$ is closely related to tessellations induced by different CPA functions:

<div class="statement definition">
<strong>(Refinements)</strong>  
Let $\mathcal T$ and $\mathcal F$ be tessellations of $\mathbb R^d$. We say that $\mathcal T$ is a <em>refinement</em> of $\mathcal F$ if every cell of $\mathcal T$ is contained in a cell of $\mathcal F$. In this case, we write $\mathcal T \ll \mathcal F$.
</div>

<div class="statement lemma" id="lemma:refinement">
<strong>(Lemma 3.4.7 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Given two sets of dual points $P, N \subseteq \mathcal D$, it holds that

$$
\mathcal T(P \cup N) \ll \mathcal T(P, N) \ll \mathcal T(N).
$$
</div>





### Characterizing $k$-cells, Part II


Any cell $\sigma \in \mathcal{T}(P, N)$ is of the form $\sigma = \sigma'\cap \sigma''$ for some $\sigma' \in \mathcal{T}(P)$ and $\sigma'' \in \mathcal{T}(N)$. In <a href="#def:cells-as-systems" class="cite-stmt hover-link">Definition</a> and <a href="#remark:cells-as-systems" class="cite-stmt hover-link">Remark</a>, we saw how $\sigma'$ and $\sigma''$ can be expressed as the solution of a system of linear inequalities $\sigma' = \{\mathbf A^{\sigma'} \mathbf x \ge \mathbf b^{\sigma'}\}$, $\sigma'' = \{\mathbf A^{\sigma''} \mathbf x \ge \mathbf b^{\sigma''}\}$. This induces a similar representation for $\sigma$:

$$
\begin{equation}
    \sigma = \left\{
      \begin{bmatrix}
        \mathbf A^{\sigma'} \\
        \mathbf A^{\sigma''}
    \end{bmatrix} \mathbf x \;\ge\;
    \begin{bmatrix}
        \mathbf b^{\sigma'} \\
        \mathbf b^{\sigma''}
    \end{bmatrix}
    \right\}.
\end{equation}
$$

Analogously to the previous section, we now turn our attention to the induced system of implicit equalities.

<div class="statement definition">
<strong>(Refined Cells as System of Linear Inequalities)</strong>  
Let $\{\mathbf A_=^{\sigma',\sigma''} \mathbf x = \mathbf b_=^{\sigma',\sigma''}\}$ be the system of implicit equalities in $\sigma$ coming from $\sigma'$. That is, any row $\mathbf a_i^{\sigma'} \in \mathbf A_=^{\sigma',\sigma''}$ is also a row in $\mathbf A^{\sigma'}$ and satisfies

$$
\langle \mathbf a_i^{\sigma'}, \mathbf x \rangle = b_i^{\sigma'} \quad \forall\, \mathbf x \in \sigma' \cap \sigma''.
$$
We write $I_=^{\sigma',\sigma''}$ for the set indexing these implicit equality constraints. Generally, $I_=^{\sigma'} \subseteq I_=^{\sigma',\sigma''}$, since the latter could contain constraints that only become implicit equalities in combination with $\sigma''$ (see <a href="#fig:two-polytopes-implicit" class="cite-fig hover-link">Figure</a> for an example).

Similarly, define the system of implicit equalities in $\sigma$ coming from $\sigma''$ as $\{\mathbf A_=^{\sigma'',\sigma'} \mathbf x = \mathbf b_=^{\sigma'',\sigma'}\}$. That is, any row $\mathbf a_i^{\sigma''} \in \mathbf A_=^{\sigma'',\sigma'}$ is also a row in $\mathbf A^{\sigma''}$ and satisfies

$$
\langle \mathbf a_i^{\sigma''}, \mathbf x \rangle = b_i^{\sigma''} \quad \forall\, \mathbf x \in \sigma' \cap \sigma''.
$$
Again, let $I_=^{\sigma'',\sigma'}$ be the set indexing these implicit equality constraints.
</div>

<div class="row mt-3 w-50 mx-auto figure-content" id="fig:two-polytopes-implicit">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/green-line.svg" class="img-fluid rounded" zoomable=true%}
    </div>
</div>
<div class="caption-left figure">
  Example of a cell $\sigma$ (green line) in the tessellation $\mathcal{T}(P, N)$, formed by the intersection of a cell ${\sigma}^{\prime} \in \mathcal{T}(P)$ (blue) and a cell ${\sigma}^{\prime\prime} \in \mathcal{T}(N)$ (red). Both ${\sigma}^{\prime}$ and ${\sigma}^{\prime\prime}$ have co-dimension one, each satisfying a single equality constraint. Their intersection imposes an additional equality constraint, resulting in $\sigma$ having co-dimension $2$.
</div>

The following proposition describes the dimension of $\sigma$ in this setup:
<div class="statement proposition" id="proposition:dim-cell-refined">
<strong>(Proposition 7.0.2 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $\sigma = \sigma' \cap \sigma'' \in \mathcal T(P,N)$ be a cell in the tessellation induced by $\mathcal Q(P) - \mathcal Q(N)$. Then

$$
\dim \sigma \;=\; d \;-\; \mathrm{rank}\!
\begin{bmatrix}
    \mathbf A_=^{\sigma',\,\sigma''} \\
    \mathbf A_=^{\sigma'',\,\sigma'}
\end{bmatrix}.
$$
</div>

Like derived for the decision boundary in the previous section, the next step is to understand how $\sigma$ appears in dual space. We begin with the following proposition:
<div class="statement proposition" id="proposition:cell-face-cont-2">
<strong>(Proposition 7.0.3 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $\sigma = \sigma' \cap \sigma'' \in \mathcal T_k(P,N)$ be a $k$-cell in the tessellation induced by $\mathcal Q(P) - \mathcal Q(N)$. Then

$$
\mathcal C\bigl(P_{\,I_=^{\sigma',\,\sigma''}} \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N_{\,I_=^{\sigma'',\,\sigma'}}\bigr) \;\in\; \mathcal U_{\,d-k}(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N).
$$
</div>

The last proposition tells us that $\mathcal C\bigl(P_{\,I_=^{\sigma',\,\sigma''}} \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N_{\,I_=^{\sigma'',\,\sigma'}}\bigr)$ is a face in $\mathcal U(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$. As in the previous section, the next proposition uses this face to explain how $\sigma$ looks in dual space:
<div class="statement proposition">
<strong>(Proposition 7.0.4 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $\sigma = \sigma' \cap \sigma'' \in \mathcal T(P, N)$ be a cell in the tessellation induced by $\mathcal Q(P) - \mathcal Q(N)$. Then there is a one-to-one correspondence between points in $\sigma$ and dual planes tangent to the upper convex hull $\mathcal U(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ containing the face 
\[
\mathcal C\bigl(P_{\,I_=^{\sigma',\,\sigma''}} \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N_{\,I_=^{\sigma'',\,\sigma'}}\bigr):
\]
\[
\{\mathbf x \in \sigma\} 
\;\longleftrightarrow\; 
\{\,f \in \text{Aff}_{\mathfrak D}(d) \mid f \| \mathcal U(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N) \text{ and } 
f \supseteq \mathcal C(P_{\,I_=^{\sigma',\,\sigma''}} \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N_{\,I_=^{\sigma'',\,\sigma'}})\}.
\]
Similarly, every face in $\mathcal U(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ corresponds to a cell in this way.
</div>

The following theorem makes precise the relationship between cells and faces that was introduced in the previous two propositions:
<div class="statement theorem" id="theorem:cell-face-bijection-2">
<strong>(Theorem 7.0.5 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
There exists a one-to-one correspondence between $k$-cells in $\mathcal T(P,N)$ and $(d-k)$-faces in $\mathcal U(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$. Specifically, the following map is a bijection:

$$
\begin{aligned}
\Psi \colon \mathcal T_k(P, N) &\xrightarrow{\sim} \mathcal U_{\,d-k}(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N) \\[6pt]
\sigma = \sigma' \cap \sigma'' \; &\mapsto \; 
\mathcal C\bigl(P_{\,I_=^{\sigma',\,\sigma''}} \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N_{\,I_=^{\sigma'',\,\sigma'}}\bigr).
\end{aligned}
$$
</div>

### Counting Affine Regions

In the special case where $\dim \sigma = d$, the result in <a href="#theorem:cell-face-bijection-2" class="cite-stmt hover-link">Theorem</a> allows counting the number of affine regions defined by a ReLU network. Indeed, we will show in this section how the affine regions can be constructed from $\mathcal T_d(P,N)$ as a set of equivalence classes. We will then translate this observation to dual space.

Let $\sigma \in \mathcal T_d(P,N)$ be a $d$-cell in the tessellation induced by $\mathcal N := \mathcal Q(P) - \mathcal Q(N)$. Then $\mathcal N$ is an affine map when restricted to $\sigma$. In particular, there exist $p^{\sigma} \in P$ and $n^{\sigma} \in N$ such that

$$
\begin{equation}
\label{eq:fbmx-aff}
    \mathcal N(\mathbf x) \;=\; \bigl(\mathcal R(p^{\sigma}) - \mathcal R(n^{\sigma})\bigr)(\mathbf x) 
    \;=\; \mathcal R(p^{\sigma} - n^{\sigma})
    \quad \forall\, \mathbf x \in \sigma.
\end{equation}
$$

However, two $d$-cells can define the same affine map:
<div class="statement proposition" id="proposition:same-aff-in-dual">
<strong>(Proposition 7.1.1 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $\sigma, \sigma' \in \mathcal T_d(P,N)$ be two distinct $d$-cells. Then $\sigma$ and $\sigma'$ define the same affine map if and only if the corresponding vertices 
$\Psi(\sigma) = p^{\sigma} + n^{\sigma}$, 
$\Psi(\sigma') = p^{\sigma'} + n^{\sigma'} \in \mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ satisfy 

$$
p^{\sigma} - n^{\sigma} \;=\; p^{\sigma'} - n^{\sigma'}.
$$
</div>

If two neighboring $d$-cells, i.e., two $d$-cells which share a $(d-1)$-face, define the same affine map, they are part of the same affine region. This implies that affine regions are coarser than $\mathcal T_d(P,N)$. The rest of this section makes this observation more precise and translates it to dual space.

<div class="statement definition">
<strong>(Adjacency)</strong>  
We make the following two definitions:
<ol style="i">
  <li>We say that two $d$-cells in $\mathcal T_d(P,N)$ are <em>adjacent</em> if they share a $(d-1)$-face.</li>
  <li>We say two vertices $p_1 + n_1,\,p_2 + n_2 \in \mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ are <em>adjacent</em> if there exists an edge $\tau \in \mathcal U_1(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ going from $p_1 + n_1$ to $p_2 + n_2$.</li>
</ol>
</div>

The following proposition relates these two notions of adjacency:
<div class="statement proposition" id="proposition:adjacency-in-dual">
<strong>(Proposition 7.1.3 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
Let $\sigma_1, \sigma_2 \in \mathcal T_d(P,N)$ be two distinct $d$-cells. Then $\sigma_1$ and $\sigma_2$ are adjacent if and only if the corresponding vertices 
$\Psi(\sigma_1) = p_1 + n_1$, 
$\Psi(\sigma_2) = p_2 + n_2 \in \mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ are adjacent (that is, satisfy $p_1 - n_1 = p_2 - n_2$).
</div>

We can now construct the equivalence relation which will identify adjacent $d$-cells in $\mathcal T_d(P,N)$ defining the same affine function:
<div class="statement definition" id="def:path-of-cells">
<strong>(Path of $d$-cells)</strong>  
A <em>path of $d$-cells</em> is a sequence $(\sigma_1,\ldots,\sigma_n) \subseteq \mathcal T_d(P,N)$ of $d$-cells such that
<ol style="i">
  <li>$\mathcal Q(P) - \mathcal Q(N)$ defines the same affine map on $\sigma_i$ and $\sigma_{i+1}$ for all $i=1,\ldots,n-1$.</li>
  <li>$\sigma_i$ is adjacent to $\sigma_{i+1}$ for all $i=1,\ldots,n-1$.</li>
</ol>
We write $\mathfrak P(P, N)$ for the set of all paths of $d$-cells in $\mathcal T_d(P,N)$.
</div>

<div class="statement definition">
<strong>(Equivalence of $d$-cells)</strong>  
Given two $d$-cells $\sigma, \sigma' \in \mathcal T_d(P,N)$, we write $\sigma \sim \sigma'$ if there exists a path of $d$-cells from $\sigma$ to $\sigma'$.
</div>

Clearly, $\sim$ defines an equivalence relation.

By <a href="#proposition:adjacency-in-dual" class="cite-stmt hover-link">Proposition</a> and <a href="#proposition:same-aff-in-dual" class="cite-stmt hover-link">Proposition</a>, this equivalence relation translates to dual space. This motivates the following definition:
<div class="statement definition" id="def:path-of-dual-points">
<strong>(Path of dual points)</strong>  
A <em>path of dual points</em> is a sequence $\bigl(p^{\sigma_1} + n^{\sigma_1},\ldots,p^{\sigma_n} + n^{\sigma_n}\bigr) \subseteq \mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ of dual points such that
<ol style="i">
  <li>$p^{\sigma_i} + n^{\sigma_i}$ is adjacent to $p^{\sigma_{i+1}} + n^{\sigma_{i+1}}$ for all $i=1,\ldots,n-1$.</li>
  <li>$p^{\sigma_i} - n^{\sigma_i} = p^{\sigma_{i+1}} - n^{\sigma_{i+1}}$ for all $i=1,\ldots,n-1$.</li>
</ol>
We write $\mathfrak P(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ for the set of all paths of dual points in $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$.
</div>

In particular, this definition induces an equivalence relation $\sim$ on $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$, where $p_1 + n_1 \sim p_2 + n_2$ if and only if there exists a path of dual points from $p_1 + n_1$ to $p_2 + n_2$.


<div class="row mt-3" id="fig:path-correspondence">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/path-of-d-cells.svg" class="img-fluid rounded" zoomable=true caption="<strong>(a)</strong>"%}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/path_of_dual_points.svg" class="img-fluid rounded" zoomable=true caption="<strong>(b)</strong>"%}
    </div>
</div>
<div class="caption-left figure">
  Example for the one-to-one correspondence between paths of $d$-cells in $\mathcal{T}_d(P,N)$ and paths of dual points in $\mathcal{U}^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ (<a href="#proposition:path-correspondence" class="cite-stmt hover-link">Proposition</a>). Subfigure <strong>(a)</strong> shows a path of $d$-cells $(\sigma_1,\sigma_2,\sigma_3)$ (blue), consisting of adjacent cells that define the same affine function. In dual space (Subfigure <strong>(b)</strong>), this corresponds to a path (thick, blue) of adjacent vertices $p^{\sigma_i} + n^{\sigma_i}$ with the property that $p^{\sigma_i} - n^{\sigma_i} = p^{\sigma_j} - n^{\sigma_j}$ for all $i,j=1,2,3$.
</div>


The following proposition relates paths in $\mathcal T_d(P,N)$ to paths in $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$:
<div class="statement proposition" id="proposition:path-correspondence">
<strong>(Proposition 7.1.7 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
There exists a one-to-one correspondence between paths of $d$-cells in $\mathcal T_d(P,N)$ and paths of dual points in $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$. It is given by

$$
\begin{aligned}
\Theta \colon \mathfrak P(P,N) &\;\to\; \mathfrak P(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N) \\[4pt]
(\sigma_1,\ldots,\sigma_n) 
&\;\mapsto\; (\,p^{\sigma_1} + n^{\sigma_1},\ldots,p^{\sigma_n} + n^{\sigma_n}\,).
\end{aligned}
$$

</div>

We finally make precise the one-to-one correspondence between affine regions and equivalence classes of $d$-cells in $\mathcal T_d(P,N)$:
<div class="statement corollary" id="corollary:affine-pieces-mod">
<strong>(Corollary 7.1.8 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
There exists a one-to-one correspondence between affine regions of a ReLU network $\mathcal Q(P) - \mathcal Q(N) \colon \mathbb R^d \to \mathbb R$ and equivalence classes in $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)/\sim$.
</div>
<div>
To better understand the space $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)/\sim$, define an unweighted graph $G = (V, E)$ with vertices $V := \mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N) \subseteq \mathbb R^{d+1}$ and edges $E := \mathcal U_1(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$. If $d=2$, then clearly $G$ is planar.
The set $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)/\sim$ arises from the graph $G$ by contracting exactly the paths in $\mathfrak P(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ (see <a href="#fig:graph-with-path" class="cite-fig hover-link">Figure</a>).
</div>

### In Practice
We have seen above that $d$-cells in $\mathcal T(P,N)$ can be finer than the affine regions of $\mathcal N = \mathcal Q(P) - \mathcal Q(N)$. Since also the activation regions~<d-cite key="hanin2019deep"></d-cite> are generally finer than the affine regions~<d-cite key="hanin2019deep"></d-cite>, one might ask whether the $d$-cells in $\mathcal{T}(P, N)$ are the same as the activation regions. However, this is not the case. As noted by Hanin and Rolnick~<d-cite key="hanin2019deep"></d-cite>, zeroing out a subnetwork may lead to different activation patterns that coalesce into a single linear region.

Importantly, the zeroed-out subnetwork does not affect the upper convex hulls of $P$ and $N$ and therefore does not influence the tessellation $\mathcal T(P, N)$. That is, two activation patterns that only differ in the zeroed-out subnetwork do not influence $\mathcal{T}(P, N)$.

Therefore, if $\mathcal N$ restricts to the same affine map on two adjacent cells $\sigma_1, \sigma_2 \in \mathcal T_d(P,N)$, two adjacent activation regions which do not just differ by a zeroed-out subnetwork need to coalesce into the same affine region. We conjecture that this happens with probability zero.

By the above argument, this means that the corresponding points $p_1 + n_1$ and $p_2 + n_2$ in dual space lie on the upper convex hull and satisfy $p_1 - n_1 = p_2 - n_2$. This furthermore motivates the conjecture, as this seems to be unlikely.

These considerationd lead us to conjecture that, in networks with random parameters, the $d$-cells are almost surely the same as the affine regions:
<div class="statement conjecture" id="conjecture:counting-affine-pieces">
<strong>(Conjecture 7.1.10 in Nazari<d-cite key="nazari2025thesis"></d-cite>)</strong>  
The number of affine regions of a random ReLU network $\mathcal Q(P) - \mathcal Q(N)$ is almost surely equal to the number of points in $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$.
</div>

<div class="row mt-3 w-50 mx-auto figure-content" id="fig:graph-with-path">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/graph.svg" class="img-fluid rounded" zoomable=true%}
    </div>
</div>
<div class="caption-left figure">
  An example of the graph $G$ induced by the $1$-skeleton $\mathcal U_1(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ for input-dimension $d=2$. By  <a href="#corollary:affine-pieces-mod" class="cite-stmt hover-link">Corollary</a>, the affine regions induced by $\mathcal Q(P) - \mathcal Q(N)$ correspond to the vertices in the (multi)-graph $G'$ obtained from $G$ by identifying all the vertices along paths $P \in \mathfrak P(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ (red).
</div>

### Example

In this subsection, we continue the example of the toy network. By <a href="#corollary:affine-pieces-mod" class="cite-stmt hover-link">Corollary</a>, the number of affine regions defined by $\mathcal N$ corresponds to the number of vertices in $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)/\sim$.

Specifically, we compute

$$
\begin{align*}
    P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N &= \{\,(7, 35, 10), (12, 34, 0), (4, 32, 8), (12, 34, 3), (3, 28, 11), (17, 24, 0), (13, 23, 2), \\
    &\phantom{=} (11, 36, 1), (11, 36, 4), (14, 21, 3), (2, 30, 12), (16, 26, 1), (16, 26, -1), (10, 38, 7),  \\
    &\phantom{=} (10, 38, 4), (12, 19, 5),(6, 31, 8), (10, 38, 5), (6, 31, 11), (8, 33, 4), (11, 21, 6), \\
    &\phantom{=} (8, 33, 7), (4, 32, 10), (8, 36, 6), (5, 33, 9), (6, 28, 6), (8, 36, 9), (13, 23, 4), (11, 36, 3), \\
    &\phantom{=} (18, 22, -3), (5, 33, 12), (11, 36, 6), (5, 30, 7), (15, 22, 2), (7, 35, 5), (9, 34, 5), \\
    &\phantom{=} (7, 35, 8), (9, 34, 8), (8, 33, 6), (14, 24, 3), (15, 19, 0), (8, 33, 9), (9, 31, 3),  \\
    &\phantom{=} (17, 24, -2), (9, 31, 6), (14, 21, 1), (10, 38, 2), (5, 30, 9), (7, 35, 7)\,\}
\end{align*}
$$

and
$$
\begin{align*}
    \mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N) &= \{\,(18, 22, -3), (15, 19, 0), (10, 38, 7), (12, 19, 5), \\
    &\phantom{=} (17, 24, 0), (2, 30, 12), (3, 28, 11), (5, 33, 12)\,\}.
\end{align*}
$$

One can quickly see that $\mathfrak P(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N) = \emptyset$, i.e., there are no adjacent dual points $p_1 + n_1,\,p_2 + n_2 \in \mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$ satisfying $p_1 - n_1 = p_2 - n_2$.

<div>
This implies $\mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)/\sim = \mathcal U^*(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$, giving $8$ affine regions. This is confirmed by <a href="{% post_url 2025-06-02-dual-representation %}#fig:example-affregs">this figure</a> from the previous post, which plots the tessellation induced by $\mathcal N$ and contains $8$ affine regions.
</div>

<div class="row mt-3 w-50 mx-auto figure-content" id="fig:tropical-toy-example-sum">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/dual_representation/running-example-sum.svg" class="img-fluid rounded" zoomable=true%}
    </div>
</div>
<div class="caption-left figure">
  Two-dimensional toy-example continued. Purple points correspond to $P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N$. The purple polygon is $\mathcal U(P \style{transform: rotate(45deg); display: inline-block;}{\boxtimes} N)$.
</div>