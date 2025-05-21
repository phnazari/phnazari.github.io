# The Geometry of Generalization
## Part 1: Tropical Geometry and ReLU Networks


This is part 1 of a two part series on the geometry of generalization. In this series I present the work I did for my master thesis (which can be found [here](/assets/pdf/Master_Thesis.pdf)).


In this first part, I show how to establish a symbolic (or "dual") representation of fully connected feed forward ReLU networks using tropical geometry. This connection has been studied in a number of previous works [cite these here], and I will build upon the constructions and proofs put forward by Piwek et al. [1].

Throughout this post, I will refrain from proving all statements in detail. However, all of the proofs can be found in my thesis [thesis]. Wherever necessary, I will concretely link to the proof in [the thesis](/assets/pdf/Master_Thesis.pdf).



### Affine Geometry

In this chapter, we introduce fundamental concepts of affine geometry, covering basic definitions, the dual representation of affine functions, and their connection to upper convex hulls. We also explore tessellations induced by maxima over affine functions.

Throughout this chapter, fix an integer $d \in \mathbb N$.
#### Affine and (D)CPA Functions
We begin by introducing fundamental concepts.


<div style="border-left: 4px solid #000; padding-left: 10px; margin: 10px 0;">
  <strong>Definition 1:</strong> <br>
Given a vector $\mathbf{a} \in \mathbb R^d$ and a scalar $b \in \mathbb R$, we define the affine function with parameters $\mathbf a$ and $b$ as
\begin{align*}
    \fab \colon \mathbb R^d &\to \mathbb R \\
    \mathbf x &\mapsto \langle \mathbf a, \mathbf x \rangle + b,
\end{align*}
where $\langle \cdot, \cdot \rangle$ is the standard Euclidean inner product on $\mathbb R^d$.
</div>

\begin{definition}[Affine Functions]
\label{def:affine-functions}
    Given a vector $\mathbf a \in \mathbb R^d$ and a scalar $b \in \mathbb R$, we define the affine function with parameters $\mathbf a$ and $b$ as
    \begin{align*}
        \fab \colon \mathbb R^d &\to \mathbb R \\
        \mathbf x &\mapsto \langle \mathbf a, \mathbf x \rangle + b,
    \end{align*}
    where $\langle \cdot, \cdot \rangle$ is the standard Euclidean inner product on $\mathbb R^d$.
\end{definition}
Ultimately, we will be taking maxima over affine maps. To classify such functions, we introduce the following concept:
\begin{definition}[CPA Functions]
\label{def:cpa}
    We say that a function $f \colon \mathbb R^d \to \mathbb R$ is CPA if it is convex and piecewise affine. We denote by $\cpad$ that set of CPA functions $\mathbb R^d \to \mathbb R$.
\end{definition}
It turns out that the class of CPA functions coincides with the class of maxima over affine functions:
\begin{proposition}[Characterizing CPA Functions~{\cite[Proposition 2]{piwek2023exact}}]
    \label{proposition:cpa}
    Any function $F \colon \mathbb R^d \to \mathbb R$ of the form
    \begin{equation*}
        F(\mathbf x) = \max\{f_1(\mathbf x),\ldots,f_n(\mathbf x)\}
    \end{equation*}
    with affine functions $f_i \colon \mathbb R^d \to \mathbb R$ is CPA. Also every CPA function with a finite number of affine pieces is of this form.
\end{proposition}
Later in this work, we will also consider differences of CPA functions:
\begin{definition}[DCPA Functions]
\label{def:dcpa}
    We say that a function $f \colon \mathbb R^d \to \mathbb R$ is DCPA if it can be written as the difference of two CPA functions. We denote by $\dcpad$ the set of DCPA function $\mathbb R^d \to \mathbb R$.
\end{definition}

\section{Affine Dualities}
\label{sec:affine-dualities}
In this section, we mainly follow the construction presented in~\cite{piwek2023exact}, which allows mapping an affine function $f \colon \mathbb R^d \to \mathbb R$ to a ``dual space``. As an outlook, exploring this transformation will ultimately lead to understanding how ReLU networks can be understood as DCPA functions.

The graph of an affine function $\mathbb R^d \to \mathbb R$ defines a hyperplane in \textit{real space}, which we define as $\RealSpace \coloneqq \mathbb R^d \times \mathbb R = \mathbb R^{d+1}$. The space of affine functions whose graph lies in $\RealSpace$ is called \textit{real affine space}, denoted by $\RealAffSpace$.

As mentioned in Definition~\ref{def:affine-functions}, any affine function $f_{\mathbf a, b} \in \RealAffSpace$ is characterized by its parameters $(\mathbf a, b) \in \mathbb R^{d+1}$. We refer to the copy of $\mathbb R^{d+1}$ that parametrizes affine functions in $\RealAffSpace$ as the \textit{dual space} $\mathcal D$.

The following lemma is a natural consequence of this construction, as it allows translating between real affine space and dual space:
\begin{lemma}
    \label{lemma:aff-real-iso}
    For any fixed dimension $d$, there exists a bijection between dual space and real affine space, given by
    \begin{align*}
        \mathcal R \colon \DualSpace &\xrightarrow{\sim} \RealAffSpace \\
        (\mathbf x, y) &\mapsto f_{\mathbf x, y}.
    \end{align*}
\end{lemma}

\begin{figure}
\centering
\begin{subfigure}{0.45\textwidth}
\begin{tikzpicture}[baseline={(0,0,0)}] % Align at the origin of the 3D coordinate system
\begin{axis}[
    axis lines=middle,
    xlabel=$x$,
    ylabel=$y$,
    zlabel=$z$,
    xmin=-2, xmax=2,
    ymin=-2, ymax=2,
    zmin=-2, zmax=2,
    grid=major,
    xtick=\empty,
    ytick=\empty,
    ztick=\empty,
    view={60}{30}
]
\addplot3[
    only marks,
    mark=*,
    mark size=2pt,
    color=blue
] coordinates {
    (-2/4, -3/4, 3/4)
} node[font=\normalsize, above right, text=blue] {$(\mathbf a, b)$};
\path (2, 1, -0.5) node[font=\normalsize] {$\DualSpace$};
\end{axis}
\end{tikzpicture}
\caption{}
\label{fig:ex-dual-point}
\end{subfigure}
\begin{subfigure}{0.45\textwidth}
\begin{tikzpicture}[baseline={(0,0,0)}] % Align at the origin of the 3D coordinate system
\begin{axis}[
    axis lines=middle,
    xlabel=$x$,
    ylabel=$y$,
    zlabel=$z$,
    xmin=-2, xmax=2,
    ymin=-2, ymax=2,
    zmin=-2, zmax=2,
    grid=major,
    xtick=\empty,
    ytick=\empty,
    ztick=\empty,
    view={60}{30}
]
\addplot3[
    surf,
    shader=flat,
    opacity=0.5,
    color=blue,
    y domain=-1.5:1.5
] {(-2*x-3*y+3)/4};
\path (-2, -1, 0.5) node[font=\normalsize] {$\RealAffSpace$};
\path (1, 1, 2) node[font=\normalsize, text=blue] {$f_{\mathbf a, b}$};
\end{axis}
\end{tikzpicture}
\caption{}
\label{fig:ex-dual-plane}
\end{subfigure}%
\caption{Example of the dual representation of an affine map $f{\mathbf a, b}$ with $\mathbf a = (-1/2, -3/4)$, $b=3/4$. Subfigure~(\subref{fig:ex-dual-plane}) contains the graph of $f_{\mathbf a,b} \in \RealAffSpace$ and Subfigure~(\subref{fig:ex-dual-point}) contains the parameterizing dual point $(\mathbf a, b) \in \DualSpace$. The map $\mathcal R$ assign to the point $(\mathbf a,b)$ the affine map $f_{\mathbf a, b}$.}
\label{fig:ex-dual}
\end{figure}

An example for $\mathcal R$ can be found in Figure~\ref{fig:ex-dual}. It has the following properties.
\begin{proposition}
\label{proposition:r-properties}
Let $\{\mathbf x_i, y_i\}_{i=1,\ldots,n} \subseteq \DualSpace$ be a set of dual points. Then the following are true:
\begin{enumerate}[i)]
    \item $\mathcal R$ is a linear operator, i.e., for any set of scalars $\{\alpha_i\}_{i=1,\ldots,n} \subseteq \mathbb R$, 
    \begin{equation*}
        \mathcal R\left(\sum_{i=1}^n \alpha_i (\mathbf x_i, y_i)\right) = \sum_{i=1}^n \alpha_i \mathcal R((\mathbf x_i,y_i)).
    \end{equation*}
    \item The set of dual points is linearly independent if and only if the corresponding set $\{\mathcal R((\mathbf x_i,y_i))\}_{i=1,\ldots,n}$ of affine functions is linearly independent.
    \item The set of dual points is affinely independent if and only if the corresponding set $\{\mathcal R((\mathbf x_i,y_i))\}_{i=1,\ldots,n}$ of affine functions is affinely independent.
\end{enumerate}
\end{proposition}
\begin{proof}
    $i)$ can be confirmed by an easy calculation. $ii)$ follows from $i)$ and $iii)$ from $ii)$ and Lemma~\ref{lemma:aff-lin-equiv}.
\end{proof}
Since both $\RealSpace$ and $\DualSpace$ are copies of $\mathbb R^{d+1}$, it is natural to ask whether we can reverse their roles in the above construction. The answer to this question is yes. We define \textit{dual affine space} $\DualAffSpace$ as the space of affine functions with graph in $\DualSpace$. Analogously to the above construction, these affine functions are parameterized by points in $\RealSpace$, though with a slight caveat:
% of $\RealSpace$ and $\DualSpace$ in the above constructions. The answer is ``yes``, although with a slight caveat.
% To do so, we start with the following definition:
%\begin{definition}[Dual Affine Space]
%Analogous to real affine space, we define the \textit{dual affine space} $\DualAffSpace$ as the space of affine functions $\mathbb R^d \to \mathbb R$ with graph in $\DualSpace$.
%\end{definition}
% With this definition at hand, we can make a statement analogous to Lemma~\ref{lemma:aff-real-iso}:
\begin{lemma}
    \label{lemma:aff-dual-iso}
    For any fixed dimension $d$, there exists a bijection between dual affine space and real space. It is given by
    \begin{align*}
           \check{\mathcal R} \colon \DualAffSpace &\xrightarrow{\sim} \RealSpace \\
            f_{\mathbf a, b} &\mapsto (-\mathbf a, b).
    \end{align*}
\end{lemma}
%In particular, since we will be using $\check{\mathcal R}^{-1}$ repeatedly later on, for any $(\mathbf a, b) \in \RealSpace$, the associated affine map is
%\begin{align*}
%    \check{\mathcal R}^{-1}(\mathbf a, b) \colon \mathbb R^d &\to \mathbb R \\
%    \mathbf x &\mapsto -\langle \mathbf a, \mathbf x \rangle + b.
%\end{align*}
Figure~\ref{fig:real-dual-diagram} provides an overview the relationship between $\RealSpace, \DualSpace, \RealAffSpace$ and $\DualAffSpace$.

\begin{figure}[h]
\centering
\begin{tikzcd}[scale=2.5]
\mathrm{Aff}_\mathfrak{R}(d) \arrow[r, "\text{graph}"] & \mathfrak{R} \\
\mathfrak D \arrow[u, "\rotatebox{90}{$\sim$}", "{\mathcal{R}}"'] & \arrow[l, "\text{graph}"', swap] \mathrm{Aff}_\mathfrak{D}(d) \arrow[u, "\rotatebox{90}{$\sim$}"', "\check{\mathcal{R}}"]
\end{tikzcd}
\caption{Diagram indicating the relationship between real (affine) and dual (affine) space.}
\label{fig:real-dual-diagram}
\end{figure}

Note that, compared to $\mathcal R$, the function $\check{\mathcal R}$ includes an additional minus and maps in the opposite direction. This is essential for ensuring that the duality properties in the following proposition hold:
\begin{proposition}[Duality Properties~{\cite[Proposition 7]{piwek2023exact}}]
\label{proposition:duality-results}
The maps $\mathcal R$ and $\check{\mathcal R}$ have the following properties (using notation from Definition~\ref{def:set-function}):
\begin{enumerate}
    \item A dual point $\mathbf c \in \mathfrak D$ lies on the graph of a dual affine function $\fab \in \DualAffSpace$ if and only if the graph of the corresponding real affine function $\mathcal R(\mathbf c)$ contains the corresponding real point $\check{\mathcal R}(\fab)$:
    \begin{equation*}
        \mathbf c \in \fab \iff \check{\mathcal R}(\fab) \in \mathcal R(\mathbf c)
    \end{equation*}
    \item A dual point $\mathbf c \in \DualSpace$ lies above the graph of a dual affine function $\fab \in \DualAffSpace$ if and only if the real point $\check{\mathcal R}(\fab)$ lies below the graph of $\mathcal R(c)$:
    \begin{equation*}
        \mathbf c \succ \fab \iff \mathcal R(\mathbf c) \succ \check{\mathcal R}(\fab) 
    \end{equation*}
\end{enumerate}
\end{proposition}

\section{CPA Functions as Upper Convex Hulls}
In the previous section, we explored a duality that enables us to identify affine maps with the vector containing their parameters. In this section, we apply these results to maxima over affine functions, which, by Proposition~\ref{proposition:cpa}, can be understood as CPA functions.

In light of the duality results from the previous section, CPA functions correspond to finite sets of dual points:

<div style="border-left: 4px solid #000; padding-left: 10px; margin: 10px 0;">
  <strong>Definition 1:</strong> <br>
</div>

\begin{definition}
\label{def:q}
    On the set $\mathcal P_\text{fin}(\DualSpace)$ of finite subsets of $\DualSpace$, the operator
    \begin{align*}
        \mathcal Q \colon \mathcal P_\text{fin}(\DualSpace) &\to \cpad\\
        S &\mapsto \mathcal Q(S) \coloneqq \max_{\mathbf s \in S} \mathcal R(\mathbf s)
    \end{align*}
    assigns to a set of dual points the associated CPA function
    \begin{align*}
        \max_{\mathbf s \in S}\mathcal R(\mathbf s)(\mathbf x) = \max_{(\mathbf a, b) \in S} \langle \mathbf x, \mathbf a \rangle + b.
    \end{align*}
    We define $\mathcal Q(\emptyset) \coloneqq 0$. On a vector of finite sets of dual points, $Q$ acts component-wise.
\end{definition}
Note that, by Proposition~\ref{proposition:cpa}, the operator $\mathcal Q$ does indeed map to $\cpad$.

\begin{figure}
\centering
\begin{subfigure}{0.45\textwidth}
\begin{tikzpicture}
\begin{axis}[view={80}{20},xtick=\empty,ytick=\empty,ztick=\empty]
\addplot3[
    opacity=0.5,
    table/row sep=\\,
    patch,
    patch type=polygon,
    vertex count=3,
    colormap/blackwhite,
    patch table with point meta={%
    % pt1 pt2 pt3 pt4 pt5 cdata
        0 1 4 0.5\\
        0 2 4 0.5\\
        2 3 4 0.5\\
        1 3 4 0.5\\
    }
]
table {
    x y z\\
    0 0 0\\% 0 USED
    1 0 0\\% 1 USED
    0 1 0\\% 2 USED 
    1 1 0\\% 3 USED 
    0.5 0.5 1\\% 4 USED
};

% Additional points (different color and marker style)
\addplot3[
    only marks, 
    mark=*, % Use square markers
    color=blue, % Set a fixed color for additional points
    mark size=2pt
]
table[row sep=\\] {
    x y z\\
    0 1 0\\%3
    1 0 0\\%1
    0 0 0 \\
    0.5 0.5 1\\%0
    1 1 0 \\
};

% Additional points (different color and marker style)
\addplot3[
    only marks, 
    mark=*, % Use square markers
    color=red, % Set a fixed color for additional points
    mark size=2pt
]
table[row sep=\\] {
    x y z\\
    0.75 0.75 0.5\\
    0.75 0.25 0.5\\
    0.5 0.5 -1.0\\
};

% replicate the vertex list to show \coordindex:
%\addplot3[only marks,nodes near coords=\coordindex]
%table[row sep=\\] {
%0 2 0\\ 2 2 0\\ 0 1 3\\ 0 0 3\\
%1 0 3\\ 2 0 2\\ 2 0 0\\ 1 1 2\\
%};
\end{axis}
\end{tikzpicture}
\end{subfigure}
\caption{Example of an upper convex hull. Let $S$ be the union of all displayed points. The blue points correspond to $\mathcal U^*(S)$, the black surface is $\mathcal U_2(S)$. In particular, $\mathcal Q(S)$ is uniquely identified by only the blue points.}
\label{fig:example-uch}
\end{figure}


Our next objective is to establish a connection between CPA functions and upper convex hulls. To begin, we first state the following proposition:
\begin{proposition}[Maximality of Upper Convex Hull~{\cite[Proposition 9]{piwek2023exact}}]
\label{proposition:maximality-of-u}
    Let $S \subseteq \DualSpace$ be a finite set of points. Then for every point $w \in \DualSpace$ lying below or on $\mathcal U(S)$ (in the sense of Definition~\ref{def:set-point}), the affine function dual to $w$ lies fully below the maximum of the affine functions whose duals lie in $\mathcal U^*(S)$. That is,
    \begin{equation}
        \mathcal R(w) \leq \max\{\mathcal R(s) \vertical s \in \mathcal U^*(S)\} = \mathcal Q(\mathcal U^*(S)).
    \end{equation}
    If $w$ lies truly below $\mathcal U(S)$, then even
    \begin{equation}
        \label{eq:w-truly-below}
        \mathcal R(w) < \mathcal Q(\mathcal U^*(S)).
    \end{equation}
\end{proposition}
\begin{proof}
    The proof follows largely the same structure as~\cite[Proposition 9]{piwek2023exact}, with a few minor adaptations.

    Let $(\mathbf x_1,y_1),\ldots,(\mathbf x_n,y_n) \in \DualSpace$, $n \geq 3$, be distinct dual points.
    We start with the following two observations:
    \begin{enumerate}[i)]
        \item if $(\mathbf x_1, y_1)$ lies directly below $(\mathbf x_2, y_2)$, i.e., $\mathbf x_1 = \mathbf x_2$ and $y_1 < y_2$, then the dual plane related to $(\mathbf x_1, y_1)$ lies below $(\mathbf x_2, y_2)$, i.e., $\mathcal R((\mathbf x_1, y_1))(\mathbf x) < \mathcal R((\mathbf x_2, y_2))(\mathbf x)$ for all $\mathbf x \in \mathbb R^d$
        \item if $(\mathbf x_n, y_n)$ lies on a face of $\mathcal U(S)$ spanned by $(\mathbf x_1,y_1),\ldots,(\mathbf x_{n-1},y_{n-1}) \in \mathcal U^*(S)$, then $\mathcal R(\mathbf x_n,y_n) \leq \max\{\mathcal R((\mathbf x_i, y_i) \vertical i=1,\ldots,n-1\}$.
    \end{enumerate}
    Claim $i)$ is trivial. For claim $ii)$, assume there exist $\alpha_i \in [0,1]$, $\sum_{i=1}^n \alpha_i = 1$, s.t.
    \begin{equation*}
    (\mathbf x_n, y_n) = \sum_{i=1}^{n-1} \alpha_i (\mathbf x_i, y_i).
    \end{equation*}
    Then
    \begin{equation*}
    \mathcal R((\mathbf x_n, y_n))(\mathbf x) = \sum_{i=1}^{n-1} \alpha_i \mathcal R((\mathbf x_i, y_i))(\mathbf x)\quad \forall \mathbf x \in \mathbb R^d
    \end{equation*}
    by linearity of $\mathcal R$ (see Proposition~\ref{proposition:r-properties}). In particular,
    \begin{equation*}
    \mathcal R(\mathbf x_n,y_n)(\mathbf x) \leq \max\{\mathcal R((\mathbf x_i, y_i))(\mathbf x) \vertical i=1,\ldots,n-1\} \quad \forall \mathbf x \in \mathbb R^d.
    \end{equation*}
    This shows claim $ii)$.

    The proposition then follows from the following observation. Assume that the point $w$ lies below or on $\mathcal U(S)$. Let $(\mathbf x_1, y_1)$ be a point directly above $w$ lying on $\mathcal U(S)$. Then, by $i)$, $\mathcal R(w) < \mathcal R((\mathbf x_1, y_1))$ if $w$ does not lie on $\mathcal U(S)$ and $\mathcal R(w) \leq \mathcal R((\mathbf x_1, y_1))$ otherwise. Furthermore, by $ii)$, $\mathcal R((\mathbf x_1, y_1)) \leq \max\{\mathcal R(s) \vertical s \in \mathcal U^*(S)\}$. This shows the claim.
\end{proof}
Having established this proposition, the identification of CPA functions with upper convex hulls is a corollary:
\begin{corollary}[CPAs as Upper Convex Hulls]
\label{corollary:upper-convex-hull-rep}
    Every CPA function $\mathcal Q(S)$ can be uniquely represented as an upper convex hull in dual space. That is, $\mathcal Q(S) = \mathcal Q(\mathcal U^*(S))$
\end{corollary}
\begin{proof}
    Let $\mathcal Q(S)$ be a CPA function. Then for any $\mathbf x \in \mathbb R^d$,
    \begin{align*}
        \mathcal Q(S)(\mathbf x) &= \max_{s \in S}\mathcal R(s)(\mathbf x) \\
        &= \max\{\max_{\mathclap{s \in \mathcal U^*(S)}}{R(s)}(\mathbf x), \max_{\mathclap{s \in S \setminus \mathcal U^*(S)}} \mathcal R(s)(\mathbf x)\} \\
        \overset{\ref{proposition:maximality-of-u}}&{=} \max_{s \in \mathcal U^*(S)} R(s)(\mathbf x) \\
        &= \mathcal Q(\mathcal U^*(S))(\mathbf x).
    \end{align*}
    This shows the claim.
\end{proof}
A visualization of Corollary~\ref{corollary:upper-convex-hull-rep} can be found in Figure~\ref{fig:example-uch}.

\section{Tessellations}
CPA functions induce a \textit{tessellation} of $\mathbb R^d$. It plays an important role in understanding ReLU networks:
\begin{definition}[Tessellation]
\label{def:tessellation}
    Given a CPA function $F(\mathbf x) \coloneqq \max\{f_1(\mathbf x),\ldots,f_n(\mathbf x)\}$, a \textit{cell} induced by $F$ is
    \begin{equation*}
        \{\mathbf x \in \mathbb R^d \vertical f_i(\mathbf x) = f_{i'}(\mathbf x) \geq f_j(\mathbf x) \; \text{for all } i,i' \in I, j \in J\},
    \end{equation*}
    where $I, J$ are disjoint sets whose union is $\{1,2,...,n\}$.
    % Its \textit{dimension} is the smallest dimension of an affine subspace on $\mathbb R^d$ containing it. The set of all regions of dimension $k$ (\textit{$k$-cells}) will be denoted by $\mathcal T_k(F)$.
    The set of all cells induced $F$ is called the \textit{tessellation} induced by $F$ and denoted by $\mathcal T(F)$.
\end{definition}
Figure~\ref{fig:example-affregs} contains an example of a tessellation. By a slight abuse of notation, we will write $\mathcal T(S)$ for the tessellation induced by the CPA function $\mathcal Q(S)$.

The following lemma establishes a connection between tessellations and polyhedral complexes, which were discussed in Section~\ref{sec:polyhedral-complexes}:
\begin{lemma}
    The tessellation induced by a CPA function $F$ forms a polyhedral complex.
\end{lemma}
\begin{proof}
    Every cell of $F$ is a polyhedron since it is defined by a set of linear inequalities. It is left to show that the following two properties hold (see Definition~\ref{def:polyhedral-complex}):
    \begin{enumerate}[i)]
        \item any face of an cell is also a cell,
        \item the intersection of two cells is either empty or a face of both cells.
    \end{enumerate}
    But this follows directly from the definition of the tessellation. Indeed, let $\cell$ be a cell defined by two sets $I$ and $J$, as in Definition~\ref{def:tessellation}. Then a face of $\cell$ is a cell associated with two sets $I' \supseteq I$, $J' \subseteq J$ obtained by moving indices from $J$ to $I$ (note that, at a face of $\cell$, there are more active equality constraints). This shows $i)$. To see that $ii)$ holds, observe that the intersection of two cells, associated with sets $I$, $J$ and $I', J'$, respectively, is the cell associated with the sets $I \cap I'$, $J \cup J' \cup (I \Delta I`)$ (here, $\Delta$ denotes the symmetric difference of sets).
\end{proof}
This last lemma tells us that we may think of a tessellation as a polyhedral complex and the following definition makes sense:
\begin{definition}
    \label{def:k-tessellation}
    Let $F$ be a CPA function. We denote by $\mathcal T_k(F)$ the $k$-skeleton of the tessellation induced by $F$. The support of $\mathcal T_{d-1}(F)$ is also called an \textit{affine (or tropical) hypersurface}.
    % The $d$-cells are called \textit{affine regions}, since they are the largest connected sets on which $F$ is an affine function.
\end{definition}

%\begin{definition}[Affine Hypersurface]
%    \label{def:affine-hypersurface}
%    The \textit{affine hypersurface} of a CPA function $F(x) = \max_i %f_i$ is the set
%    \begin{equation*}
%        \mathcal H(F) = \{x \in \mathbb R^d \vertical f_i(x) = f_j(x) = %F(x) \text{ for some } i \neq j\}.
%    \end{equation*}
%end{definition}

%Before focussing on how to apply the affine duality to the context of ReLU networks, we make one final definition:
%\begin{definition}[Upper Hull]
%    Let $S \subseteq \mathbb R^{d+1}$ be a finite set of points. We denote by $\mathcal C(S)$ the \textit{convex hull} of $S$ and by
%    \begin{equation*}
%        \mathcal U(S) \coloneqq \{(\mathbf x, y) \in \mathcal C(S) \vertical (\mathbf x, x+ \varepsilon) \notin \mathcal C(S)\; \text{for any} \; \varepsilon > 0\}
%    \end{equation*}
%    the \textit{upper hull} of $S$.
%\end{definition}
\begin{example}
As an example, Figure~\ref{fig:tessellation} shows the tessellation induced by $P$ and $N$.
\begin{equation}
\label{eq:example-tessellation}
    f \colon \mathbb R^2 \to \mathbb R, \quad (x,y) \mapsto \max \{1+2x, 1+2y, 2+x+y, 2+x, 2+y, 2\}.
\end{equation}
The blue lines correspond to points on which two affine functions agree and are larger than the others. They form the $1$-skeleton of the tessellation. The intersections of these lines are the $0$-cells. On each of the white convex regions (the $2$-cells) $f$ is affine. 

Figure~\ref{fig:tessellation} also illustrates how the tesselation forms a polyhedral complex: the face of any polyhedron is again a polyhedron (for example, the faces of the white convex regions are the $1$-cells), and the intersection of any two polyhedra is either empty or again a face.
\end{example}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.3\linewidth]{media/tessellation.png}
    \caption{Figure 1 in~\cite{zhang2018tropical}. Example of a tessellation, induced by the DCPA-function given in Equation~\eqref{eq:example-tessellation}.}
    \label{fig:tessellation}
\end{figure}

So far, we have studied what it means for a CPA function to induce a tessellation of $\mathbb R^d$. The following definition clarifies what it means for a DCPA to do so.
\begin{definition}
    \label{def:tessellation-refinement}
    Let $F = \mathcal Q(P) - \mathcal Q(N)$ be a DCPA function. We then define the tessellation $\mathcal T(P,N)$ induced by $F$ to consist of all non-empty pairwise intersections of cells induced by $P$ and $N$, i.e.
    \begin{equation*}
        \mathcal T(P, N) \coloneqq \{\cell \cap \cell' \vertical \cell \in \mathcal T(P), \; \cell' \in \mathcal T(N)\} \setminus \emptyset.
    \end{equation*}
\end{definition}
As it turns out, $\mathcal T(P,N)$ is closely related to tessellations induced by different CPA functions:
\begin{definition}[Refinements]
    Let $\mathcal T$ and $\mathcal F$ be tessellations of $\mathbb R^d$. We say that $\mathcal T$ is a \textit{refinement} $\mathcal F$ if every cell of $\mathcal T$ is contained in a cell of $\mathcal F$. In this case, we write $\mathcal T \ll \mathcal F$.
\end{definition}
\begin{lemma}
    \label{lemma:refinement}
    Given two sets of dual points $P, N \subseteq \DualSpace$, it holds that
    \begin{equation*}
         \mathcal T(P \cup N) \ll \mathcal T(P, N) \ll \mathcal T(N).
    \end{equation*}
\end{lemma}
\begin{proof}
For ease of notation, enumerate $N = \{n_1,\ldots,n_m\}$ and $P = \{p_1,\ldots,p_k\}$ with $m,k \in \mathbb N$.

% On the one hand, a cell of $\mathcal T(P, N)$ is given by the solution of a system
A cell of $\mathcal T(N)$ is given by the solution of a system
\begin{align}
\label{eq:region-n}
\left\{
    \begin{aligned}
        \mathcal R(n_i) &= \mathcal R(n_j) \;\; \forall i,j \in I \\
        \mathcal R(n_i) &\geq \mathcal R(n_j) \;\; \forall i \in I, \; j \in J
    \end{aligned}
\right.
\end{align}
for some disjoint partition $I \disjointunion J = \{1,\ldots,m\}$.

A cell of  $\mathcal T(P, N)$ is given by the solution of a system
\begin{align}
\label{eq:region-p-comma-n}
\left\{
\begin{aligned}
    \mathcal R(n_i) &= \mathcal R(n_j) \;\; \forall i,j \in I \\
    \mathcal R(p_{i'}) &= \mathcal R(p_{j'}) \;\; \forall i',j' \in I' \\
    \mathcal R(n_i) &\geq \mathcal R(n_j) \;\; \forall i \in I, \; j \in J \\
    \mathcal R(p_{i'}) &\geq \mathcal R(p_{j'}) \;\; \forall i' \in I', \; j' \in J'
\end{aligned}
\right.
\end{align}
for some disjoint partitions $I \disjointunion J = \{1,\ldots,m\}$ and $I' \disjointunion J' = \{1,\ldots,k\}$.

A cell of $\mathcal T(P \cup N)$ is given by the solution of a system
\begin{align}
\label{eq:region-p-cup-n}
\left\{
\begin{aligned}
    \mathcal R(n_i) &= \mathcal R(n_j) \;\; \forall i,j \in I \\
    \mathcal R(p_{i'}) &= \mathcal R(p_{j'}) \;\; \forall i',j' \in I' \\
    \mathcal R(n_i) &= \mathcal R(p_{j'}) \;\; \forall i\in I, \; j' \in I' \\
    \mathcal R(n_i) &\geq \mathcal R(n_j) \;\; \forall i \in I, \; j \in J \\
    \mathcal R(p_{i'}) &\geq \mathcal R(p_{j'}) \;\; \forall i' \in I', \; j' \in J' \\
    \mathcal R(n_i) &\geq \mathcal R(p_{j'}) \;\; \forall i \in I, \; j' \in J' \\
    \mathcal R(p_{i'}) &\geq \mathcal R(n_j) \;\; \forall i' \in I', \; j \in J
\end{aligned}
\right.
\end{align}
for some disjoint partitions $I \disjointunion J = \{1,\ldots,m\}$ and $I' \disjointunion J' = \{1,\ldots,k\}$.

Clearly, any solution to System~\eqref{eq:region-p-cup-n} also solves System~\eqref{eq:region-p-comma-n} and any solution to System~\eqref{eq:region-p-comma-n} also solves System~\eqref{eq:region-n}. This implies the claim.
\end{proof}


