
#import "@preview/bloated-neurips:0.7.0": botrule, midrule, neurips2025, paragraph, toprule, url
#import "@preview/lovelace:0.3.0": *
// #import "@preview/algo:0.3.6": algo, i, d, comment, code
#import "@preview/algorithmic:1.0.7"
#import algorithmic: style-algorithm, algorithm-figure


#let todo(body) = {
  highlight(
    text(
      upper(body),
      size: 1.5em,
    ),
    stroke: red,
  )
}

#let boxed(body) = block(
  fill: luma(90%),
  stroke: 1pt+luma(80%),
  inset: 3pt,             // padding inside
  radius: 4pt,            // optional rounded corners
  body
)

#let affls = (
  cmu: (
    department: "Electrical and Computer Engineering",
    institution: "Carnegie Mellon University",
    // location: "Pittsburgh, PA",
    // country: "USA"
  ),
)

#let authors = (
   (name: "Krish Agarwal",
   affl: "cmu",
   email: "krisha2@andrew.cmu.edu",
   equal: true),
  (name: "Marco Christiani",
   affl: "cmu",
   email: "marcochr@andrew.cmu.edu",
   equal: true),
  (name: "Jiang Meng Liao",
   affl: "cmu",
   email: "jiangmel@andrew.cmu.edu",
   equal: true),
)

#show: neurips2025.with(
  title: [Federated Matching with Transformers],
  authors: (authors, affls),
  keywords: ("Machine Learning", "NeurIPS"),
  abstract: [
    Naive FedAvg can face convergence issues when clients take a large amount of local steps per communication round and diverge from one another. A common source of divergence is heterogeneous data, but another major source of divergence can be weight permutation invariance of neural network architectures. This motivates finding approaches that account for model structure during weight aggregation to improve convergence. Some existing approaches show promising results, like FedMA, which performs channel-wise matching in convolutional neural networks to average the closest convolution kernel channels with each other between multiple client models. In this work, we find that similar permutation invariance exists in transformer architectures, specifically in attention blocks at a head-level granularity. To address this permutation invariance, we propose our method, _FedMAT_, to find a (partial) bipartite matching of attention heads between the global model and each client model in a federated learning setting to perform better client model aggregation and achieve faster convergence. Our work may be found online 
    #footnote[Our code is made available at https://github.com/Marco-Christiani/fedmat].
  ],
  bibliography: bibliography(
    "main.bib",
    style: "chicago-author-date"
  ),
  bibliography-opts: (title: none, full: true),  // Only for example paper.
  appendix: [],
  accepted: true,
)

// Naive FedAvg #cite(<pmlr-v54-mcmahan17a>) is known to struggle under heterogeneous data, as seen in prior transformer-based FL methods such as FedTP #cite(<10130784>); this motivates finding approaches that account for model structure during weight aggregation to improve convergence. FedMA #cite(<Wang2020Federated>) was proposed to address this problem by exploiting activation channel-wise permutation invariance in convolutional neural networks to average the closest convolution kernel channels with each other between multiple client models. FedMHA #cite(<DARZI2024102936>) uses regularization in the local and global transformer updates in an attempt to improve alignment but does not take advantage of the permutation symmetry of multi-head attention. Most recently, #cite(<verma2024merging>, form: "prose") showed that explicitly matching and permuting attention heads before averaging can improve performance of the final merged model but did not consider the federated setting in their analysis. Our method _FedMAT_ builds on the ideas of FedMA #cite(<Wang2020Federated>) and explores permutation invariance across attention heads in multi-head attention networks. We propose finding a (partial) bipartite matching of attention heads with the global model with each client model in a federated learning setting to perform better client model aggregation and achieve faster convergence.

// #boxed[
//   _We just need you to do the introduction section with the problem setup, well formulated objectives and motivation. We also encourage you to describe the methodology/approach you plan to try out to solve the problem if you already have one (or many)._ 

//   #text(blue, [https://piazza.com/class/med3jy34dul20t/post/92])
  
  
//   _7th November: Checkpoint 1 - where the groups finalize problem statement and submit updated one page proposal with description of the problem setup and expected results._
  
//   _14th November: Checkpoint 2 - mid term report, groups are expected to describe their progress briefly._
  
//   #text(blue, [https://piazza.com/class/med3jy34dul20t/post/71])
// ]

= Introduction

Prior transformer-based FL methods such as FedTP #cite(<10130784>) face convergence challenges that arise from heterogenous client data. This causes clients to diverge from one another during local optimization steps, resulting in poor server aggregation.

Client divergence can be further exacerbated by permutation invariance in neural network architectures. #cite(<NEURIPS2019_31c0c178>, form: "prose") and #cite(<Wang2020Federated>, form: "prose") have shown permutation invariance in FCNNs, CNNs, and LSTMs. FedMA #cite(<Wang2020Federated>) was proposed to address this problem for convolutional neural networks by exploiting activation channel-wise permutation invariance to average the closest convolution kernel channels with each other between multiple client models. FedMHA #cite(<DARZI2024102936>) uses regularization in the local and global transformer updates in an attempt to improve alignment but does not take advantage of the permutation symmetry of multi-head attention. Most recently, #cite(<verma2024merging>, form: "prose") showed that explicitly matching and permuting attention heads before averaging can improve performance of the final merged model but did not consider the federated setting in their analysis.

We identify that multi-head attention is also permutation invariant in the ordering of its attention heads. As modern deep learning models are increasingly built using multi-head attention, we believe Federated Matching #cite(<Wang2020Federated>) can be adapted to the modern federated learning landscape. Our method _FedMAT_ builds on the ideas of FedMA #cite(<Wang2020Federated>) and explores permutation invariance across attention heads in multi-head attention networks. We propose finding a (partial) bipartite matching of attention heads with the global model with each client model in a federated learning setting to perform better client model aggregation and achieve faster convergence.

// == Motivation
// See abstract
// // 

= Federated Matching for Transformers
== Permutation Invariance of Transformers
A transformer block is permutation invariant over the order of the attention heads. We define an $H$-headed transformer block to be

// Math syntax: https://typst.app/docs/reference/math/
// Math symbols: https://typst.app/docs/reference/symbols/sym/
#let vec = math.vec.with(delim: "[")

$
&Q_i = X W^Q_i,
K_i = X W^K_i,
V_i = X W^V_i\
&A_i  = "softmax"(frac(Q_i K_i^top, sqrt(d_k))) V_i \
&O = [ A_1 dots " " A_H ] W^O
= [ A_1 dots " " A_H ] vec(W^O_1, dots.v, W^O_H) 
= sum_(i=1)^H A_i W^O_i
$

where $X in RR^(n times d)$, $W^Q_i,W^K_i in RR^(d times d_k)$, $W^V_i in RR^(d times d_v)$, $W^O in RR^(H d_v times d)$, and $W^O_i in RR^(d_v times d)$ denotes the $d_v$ rows of $W^O$ from $(i-1) d_v + 1$ to $i d_v$. Here, $n$ is the number of tokens, $d$ is the size of the input embeddings, $d_k$ is the size of the intermediate per-head embeddings, $d_v$ is the per-head output dimensionality, and $H$ is the number of attention heads. Typically, $d_v = d_k$ and $d = H d_v$.

Taking some permutation $pi(i): [H] -> [H]$ of the order of
attention heads, if we define $A^((pi))_i = A_(pi(i))$ and $W^(O,pi)_i = W^O_(pi(i))$, then
// such that $A_i = A^((pi))_(pi(i))$, we see that
// $"MultiHead"^((pi)) = sum_(i=1)^H A^((pi))_i O^((pi))_i$.

// We assign $O^((pi))_i = O_(pi^(-1)(i))$ and find that
$
O^((pi)) &= sum_(i=1)^H A^((pi))_i W^(O,pi)_i
= sum_(i=1)^H A_(pi(i)) W^O_(pi(i)) \ &= sum_(i in {pi(1), dots, pi(H)}) A_i W^O_i = sum_(i=1)^H A_i W^O_i = O
$

By invoking associativity of addition, we prove that a multi-head attention block is in invariant over permutation of the order of the attention heads, if the corresponding rows in $W^O$ are likewise permuted.

// Should we discuss the permutation invariance as part of the introduction so we can explain the issue with naive fedavg

== FedMAT Algorithm

In Federated Matching for Transformers (FedMAT), for each client $c$ per communication round $t$, we find some permutation $pi^((c, t))(i): [H] -> [H]$ operating on the following $2 d (d_k + d_v)$-dimensional vectors

$ {"concat"("flatten"(W^Q_i), "flatten"(W^K_i), "flatten"(W^V_i), "flatten"(W^O_i))}_(i=1)^H $

for every attention head in a multi-head attention block. In FedMA, a permutation applied at one layer has to be corrected using inverse permutations for subsequent layers, prompting FedMA to use a layer-wise training schedule so permutations/inverse permutations only need to be accounted for across client models for one layer at a time. In FedMAT, we apply permutation not only to the per-head $W^Q_i$, $W^K_i$, $W^V_i$ weights, but we also apply the same permutation on the corresponding rows of the output projection $W^O_i$. This means the permutation is applied along the same dimension that is reduced when performing matrix multiplication of the attention output $A$ with the output projection $W^O$. As such, the permutations applied at a layer are fully self-contained (applying a permutation at one layer requires no downstream correction for any other layers). This enables us to avoid any layer-wise training schedule. Instead we train every layer during every step, and we perform the matching independently per layer.

To perform matching (finding the correct permutation per client), we experiment with two methods, including FedMA's extended Hungarian matching algorithm #cite(<HungarianMatchingAlgo>) and a custom greedy matcher, shown in @greedy_matching.

#show: style-algorithm
#algorithm-figure(
  "Greedy matching algorithm",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      "GreedyMatch",
      ("C", "H", "X", "R", "dist"),
      {
        Comment[$C$: number of clients]
        Comment[$H$: number of attention heads]
        Comment[$X = {(x_(g,i)) | g in [C], i in [H]}$: head-wise matching vectors for all clients]
        Comment[$R = {(r_j) | j in [H]}$: ordered reference head-wise matching vectors]
        Comment[$"dist"(dot,dot)$: a distance metric]
        Comment[Output: $(pi^((1)), dots, pi^((n)))$ where $pi^((g)) in [H]^H$ and $pi^((g))_i$ is the index of the reference matched to $x_(g,i)$]
        For(
          $g = 1, dots, C$,
          {
            // #i #comment[Compute pairwise distances between client $g$ matching vectors and reference.]\
            LineComment(Assign[$U$][${1, dots, H}$], [set of unused reference indices])
            For( 
              $i = 1, dots, H$,// Comment[greedy assignment of client heads to reference],
              {
                Assign[$j^*$][$"argmin"_(j in U) "dist"(x_(g,i), r_j$)]
                Assign[$pi^((g))_i$][$j^*$]
                Assign[$U$][$U without {j^*}$]
              }
            )
          }
        )
        Return[$(pi^((1)), dots, pi^((C)))$]
      },
    )
  }
) <greedy_matching>

FedMA #cite(<Wang2020Federated>) includes provisions in its matching
algorithm for the creation of new convolutional kernel channels for CNNs
(which would translate to new attention heads in our case),
but we will not explore this in our report due to compute and time constraints.

= Experiments

== Setup
We chose to run our experiments on the CIFAR-10 dataset; while we initially planned to train our model on the ImageNet dataset, we found that the dataset was too challenging for our models to converge within our compute and time constraints.

We selected the AdamW optimizer as the best performing optimizer, with a learning rate of $1 dot 10^(-4)$ and weight decay of  $1 dot 10^(-2)$.

We choose a batch size of 64, with 10 clients, using inverse class frequency weighted Focal Loss #cite(<lin2018focallossdenseobject>). We used several data augmentation techniques including random flips and rotations, gaussian noise, color jitter, and random erasure. We run each client for 500 local iterations before aggregating in a communication round, for 48 communication rounds.

We varied the homogeneity of class distributions between the clients as well, using a Dirichlet distribution with tunable concentration to sample a categorical distribution of clients for each class of image in our training set. We perform a strict partition of the full training set between clients.

// experiments we should present in main report: regular_homogenous, high_lr, clientwise_init_homogenous

// experiments for appendix: optimizer_ablation (refer to this in the setup to explain why we use AdamW), fedavg_homovenous_vs_heterogenous, regular_heterogenous, clientwise_init_heterogenous

// @alex is this^ good

== Matcher comparison
Considering a highly homogenous distribution of training data between clients, we notice that the models perform identically, as shown in @regular_homogenous. This is likely because the client models are all initialized using the same random noise, and in the first communication round, the client models do not diverge significantly relative to the magnitude of noise, causing both the greedy and Hungarian matchers to always select the identity permutation for all clients. This persists for all communication rounds.

We perform additional experiments to encourage more client drift. These include experiments that use separate random initialization per client (as shown in @clientwise_homogenous) as well as a high initial learning rate with learning rate decay (as shown in @high_lr). In these cases, we see that more aggressive client drift does indeed lead to differences in the results (i.e. the greedy and Hungarian matchers do not select the identity permutation). In some cases, greedy or Hungarian matching can marginally exceed the performance of vanilla FedAvg, although any improvement is very minor.

We also include additional experiments under more aggressive data heterogeneity conditions in @appendix. However, we observe that using data heterogeneity does not significantly impact the relative comparison between matching strategies.

#grid(
  columns: (33%, 33%, 33%),
  rows: (auto, auto),
  column-gutter: 1em,
  row-gutter: 1em,
  [
    #figure(
      image("assets/regular_homogenous_experiment.png"),
      caption: [Matcher comparison with homogenous data ($alpha = 100$)],
    ) <regular_homogenous>
  ],
  [
    #figure(
      image("assets/clientwise_init_homogenous.png"),
      caption: [Matcher comparison with independently initialized clients and homogenous data ($alpha = 100$)]
    ) <clientwise_homogenous>
  ],
  [
    #figure(
      image("assets/high_lr_experiment.png"),
      caption: [Matcher comparison with high learning rate ($eta=0.1, alpha = 100$)]
    ) <high_lr>
  ],
)


= Conclusion and Future Works
We find that there is no significant difference in performance between FedAvg and matched averaging methods on transformers. We believe that there are a few reasons that lead to this negative result.

First, we believe that the strongly initialized, high-parameter count attention heads are less susceptible to the same kind of cluster-shifting behavior that the authors of FedMA found for convolution kernel channels. We believe this is because attention heads typically are much larger than a convolution channel in terms of parameter count, which makes it much less likely that any sort of matching would lead to a reassignment of attention head order as a client would need to shift many more parameters to cause a difference in clustering.

Second, we believe that our holistic training method, as compared to FedMA's layer-by-layer training method, leads to the model gradients to be dispersed throughout all layers. This means that the attention heads in each layer will have less drift between clients and further reduce the effectiveness of matched averaging compared to naive averaging.

// Third, 

Future studies could explore these hypotheses by measuring matching behavior and the drifting of individual attention heads. Reducing trainable parameter count could also be a viable way to bring out the benefit of matched averaging, using methods such as LoRA and reducing the numbers of attention heads per block.

#pagebreak()

= Appendix <appendix>

Here, we present some additional experiments that we conducted to better justify our earlier claims.

In @optimizer_ablation, we present an ablation of using various optimizers with vanilla FedAvg. As shown, AdamW achieves the highest performance (under homogenous client distributions), hence we select AdamW as the optimizer for all other experiments.

In @fedavg_homo_vs_hetero, we show that performance of the server model significantly drops under data heterogeneity, as would be expected. This is particularly because we experiment under a high number of local steps, so with significant client divergence due to data heterogeneity, we naturally get poorer convergence.

In @regular_heterogenous, we show that performance between matchers is identical under heterogenous datasets as well. Similarly, the relative performance of the matchers does not change significantly when using separate random initialization per client, as shown in @clientwise_heterogenous. We would have expected that more data heterogeneity leads to more client drift, which might make permutation invariance more of an issue, but this does not seem to be the case.

// #table(
//   columns: (auto, auto, auto, auto, auto),
//   table.header([Matching Algorithm], [Communication Rounds], [Client Count], [Dirichlet Concentration], [Final Performance]),
//   "None",       $50$, $10$, $100.0$, $???$,
//   "Greedy",     $50$, $10$, $100.0$, $???$,
//   "Hungarian",  $50$, $10$, $100.0$, $???$,
// )

#grid(
  columns: (50%, 50%),
  rows: (auto, auto),
  row-gutter: 1em,
  column-gutter: 1em,
  [
    #figure(
      image("assets/fedavg_optimizer_ablation.png"),
      caption: [Comparison of various optimizers for vanilla FedAvg (homogenous, $alpha = 100$)]
    ) <optimizer_ablation>
  ],
  [
    #figure(
      image("assets/fedavg_homogenous_vs_heterogenous.png"),
      caption: [Dirichlet concentration comparison with FedAvg]
    ) <fedavg_homo_vs_hetero>
  ],
  [
    #figure(
      image("assets/regular_heterogenous_experiment.png"),
      caption: [Matcher comparison with heterogenous data ($alpha = 3$)]
    ) <regular_heterogenous>
  ],
  [
    #figure(
      image("assets/clientwise_init_heterogenous_experiment.png"),
      caption: [Matcher comparison with independently initialized clients (heterogenous, $alpha = 3$)]
    ) <clientwise_heterogenous>
  ],
)

// We have done groundwork on matching algorithms and training a baseline comparison. We have results for a preliminary ViT #cite(<wu2020visual>) trained on the CIFAR-10 dataset #cite(<Krizhevsky09learningmultiple>) normalized and interpolated to 224x224 on an A100 with batch size of 1024 and no data augmentation.  We used CIFAR-10 for iteration expediency, and we plan to use ImageNet10k #cite(<deng2009imagenet>) in our experiments for the final report. We trained for 10 epochs over the entire trainset and achieved accuracy of 51.05% on the test set.

// #figure(
//   image("assets/raw-training-loss.png"),
//   caption: [Baseline ViT training loss over 10 epochs],
// ) <rawlossfig>

// We have also ran a fine-tuning baseline with a ViT pretrained on ImageNet-21k and ImageNet 2012, with a
// final accuracy of 94.46% on the test set.

// #figure(
//   image("assets/pretrained-training-loss.png"),
//   caption: [Pretrained ViT training loss over 10 epochs],
// ) <rawlossfig>


// == Planned Work and Experiments

// We plan to compare FedMAT against the baselines of standard FedAvg and regular minibatch SGD in training a Vision Transformer #cite(<wu2020visual>) on the ImageNet10K #cite(<deng2009imagenet>) dataset. We plan to compare training from scratch and training from a pretrained checkpoint as well, for a total of six runs. 

// FedAvg will serve at the baseline method as it is a canonical federated learning approach where clients train locally for several steps then synchronize by directly averaging model parameters elementwise. This baseline ignores structural permutation invariance across attention heads, treating all parameters as aligned by position rather than by function.

// By comparing our proposed method, FedMAT, to FedAvg under identical constraints, we can isolate the impact of explicit head alignment during aggregation. We will quantify our method's efficacy in terms of convergence rate and communication efficiency.

// We believe that we cannot fairly compare against FedMA #cite(<Wang2020Federated>) as they use a different and much
// smaller architecture trained on a much smaller dataset (CIFAR-10) than would be viable for transformer networks.

// In addition to the homogenous case we will explore several sources of heterogeneity such as label and feature distribution as well as participation rates and communication frequency.

// == Expected results
// // Piazza post asks that we discuss expected results
// If the proposed technique is effective we expect convergence to be faster and more stable than FedAvg in the homogenous case and ideally across sources of heterogeneity as well. We anticipate that developing sufficiently general methods to handle distinct model architectures could prove challenging but if possible should internal semantic consistency over the course of training and yield better overall performance. 


#pagebreak()

// More broadly, a positive outcome would indicate that permutation-aware aggregation can generalize beyond convolutional filters and structural aggregation is a key ingredient in the federated training of neural networks.

// *Modalities*
// Text
// Images

// *Data heterogeneity*
// Label skew
// Quantity skew
// Feature distribution
// Domain shift: subcorpus sampling

// *Communication budget?*
// *Participation*
// Dataset: CIFAR-100

// $
// "MultiHead"^((pi)) &= [ A^((pi))_1 dots " " A^((pi))_H ]O^((pi)) = 
// [ A_(pi^(-1)(1)) dots " " A_(pi^(-1)(H)) ]
// vec(O^((pi))_1, dots.v, O^((pi))_H)\
// &= [ A_1 dots " " A_H ]
// vec(O^((pi))_(pi(1)), dots.v, O^((pi))_(pi(H)))
// = A
// vec(O^((pi))_(pi(1)), dots.v, O^((pi))_(pi(H)))
// $

/*
#line(length: 100%)

Let $O_i$ denote the rows of matrix $O$ starting from row $d_k (i - 1)$ and ending at (but not including) row $d_k i$ (using 0-indexing). This means
$
"MultiHead" = [ A_1 dots " " A_H ] O = [ A_1 dots " " A_H ] vec(O_1, dots.v, O_H) = sum_(i=1)^H A_i O_i \
$

Suppose we apply some permutation $pi(i): [H] <-> [H]$ on the order of the attention heads and concatenate the attention head outputs in this permuted order. We can also define some modified output matrix $O^((pi))$ and define
$
"MultiHead"^((pi)) &= [ A_(pi(1)) dots " " A_(pi(H)) ]O^((pi)) = 
[ A_(pi(1)) dots " " A_(pi(H)) ] vec(O^((pi))_1, dots.v, O^((pi))_H) \ &= sum_(i=1)^H A_(pi(i)) O_i^((pi)) = sum_(i in {pi(1), dots, pi(H)}) A_i O^((pi))_(pi^(-1)(i))
$

To ensure that $"MultiHead" = "MultiHead"^((pi))$, we can set $O^((pi))_i = O_(pi(i))$, so
$
"MultiHead"^((pi)) &= sum_(i in {pi(1), dots, pi(H)}) A_i O^((pi))_(pi^(-1)(i)) = sum_(i in {pi(1), dots, pi(H)}) A_i O_(pi(pi^(-1)(i))) \ &= sum_(i in {pi(1), dots, pi(H)}) A_i O_i = sum_(i=1)^H A_i O_i = "MultiHead"
$
That is, we can obtain the same output from permuting the heads by applying a compensating permutation on groups of rows of $O$. This means that, for a given attention block, there exist at least $H!$ attention blocks that are input $->$ output equivalent that simply arise from being able to permute the attention heads (and apply the compensating permutation on $O$). This presents a challenge for FedAvg, as multiple clients could theoretically learn similar models in different permutation variants, but lose significant training progress through server aggregation. In order to address this potentially lost progress, perhaps the server aggregation can be modified to account for this permutation invariance.

// where $X in RR^(n times d)$, $W^Q_i,W^K_i,W^V_i in RR^(d times d_k)$, and $O in RR^(H d_k times d)$, $n$ being the number of tokens, $d$ being the dimensionality of input embeddings, $d_k$ being the dimensionality of intermediate per-head embeddings, and $H$ being the number of attention heads.
*/

// #pagebreak()

// = fedma

// #figure(
//   kind: "algorithm",
//   supplement: [Algorithm],

//   pseudocode-list(numbered-title: [Federated Matched Averaging (FedMA)])[
//     + *Input* : local weights of _N_-layer architectures ${ W_(j,1), dots, W_(j,N) }_(j=1)^J$ from _J_ clients
//     + *Output* : global weights ${ W_1, dots, W_N }$
//     + $n = 1$
//     + *while* $n <= N$ *do*
//       + *if* $n < N$ *then*
//         + ${ Pi_j }_(j=1)^J = "BBP-MAP"({ W_(j,n) }_(j=1)^J)$;  \/\/ call BBP–MAP to solve Eq. 2
//         + $W_n = frac(1, J) sum_j W_(j,n) Pi_j^T$
//       + *else*
//         + $W_n = sum_(k=1)^K sum_j p_(j k) W_(j l, n)$ where $p_k$ is fraction of data points with label _k_ on worker _j_;
//       + *end*
//       + *for* $j in {1, dots, J }$ *do*
//         + $W_(j,n+1) arrow.l Pi_j W_(j,n+1)$  \/\/ permutate the next-layer weights
//         + Train ${ W_(j,n+1), dots, W_(j,L) }$ with $W_n$ frozen
//       + *end*
//       + $n = n + 1$
//     + *end*
//   ]
// ) <fedma>


// #figure(
//   kind: "algorithm",
//   supplement: [Algorithm],

//   pseudocode-list(numbered-title: [My cool algorithm])[
//     + do something
//     + do something else
//     + *while* still something to do
//       + do even more
//       + *if* not done yet *then*
//         + wait a bit
//         + resume working
//       + *else*
//         + go home
//       + *end*
//     + *end*
//   ]
// ) <a-cool-algorithm>

// == Math example
// #boxed([
//   *Math example*
  
//   _Aligned math_
  
//   $ sum_(k=0)^n k
//       &= 1 + ... + n \
//       &= (n(n+1)) / 2 $
  
//   *More things*
  
//   For an L-smooth function, with  
//   $eta L (1 + min( d / s^2 , sqrt(d) / s )) + eta^2 L^2 tau (tau - 1) <= 1$,  
//   and starting point $x_1$, after $t$ iterations:
  
//   $
//     // EE[ frac(1, t) sum_(k=1)^t ||nabla F(x_k)||^2 ]
//     <= frac( 2 ( F(x_1) - F_"inf" ) , eta t )
//      + frac( eta L sigma^2 , m ) ( 1 + min( d / s^2 , sqrt(d) / s ) )
//      + eta^2 L^2 sigma^2 (tau - 1)
//   $
  
//   // where $x_k$ denotes the averaged model at iteration $k$.
//   - Second term $arrow.t$ if quantization levels $s arrow.b$.  
//   - Third term: additional error due to local steps $tau$.  
//   - Thus, convergence worsens if $s$ is reduced or $tau$ is increased.
// ])

// == Retrieval of style files

// These are clickable: Sections~#ref(<gen_inst>, supplement: none), #ref(<headings>, supplement: none), and #ref(<others>, supplement: none)

// = General formatting instructions <gen_inst>

// Please pay special attention to the instructions in @others regarding figures,
// tables, acknowledgments, and references.

// = Headings: first level <headings>

// == Headings: second level

// === Headings: third level

// #paragraph[Paragraphs] There is also a `\paragraph` command available, which
// sets the heading in bold, flush left, and inline with the text, with the
// heading followed by #1em of space.

// = Citations, figures, tables, references <others>

// == Footnotes

// Footnotes should be used sparingly. If you do require a footnote, indicate
// footnotes with a number#footnote[Sample of the first footnote.] in the text.
// Place the footnotes at the bottom of the page on which they appear. Precede the
// footnote with a horizontal rule of 2~inches (12~picas).

// Note that footnotes are properly typeset _after_ punctuation marks.#footnote[As
// in this example.]

// == Figures

// #figure(
//   rect(width: 4.25cm, height: 4.25cm, stroke: 0.4pt),
//   caption: [Sample figure caption.],
//   placement: top,
// )

// #pagebreak()

// == Tables <tables>

// All tables must be centered, neat, clean and legible.  The table number and
// title always appear before the table. See @sample-table.

// This package was used to typeset @sample-table.

// #figure(
//   caption: [Sample table title.],
//   placement: top,
//   table(
//     columns: 3,
//     align: left + horizon,
//     stroke: none,
//     toprule,
//     table.header(
//       table.cell(colspan: 2, align: center)[Part], [],
//       table.hline(start: 0, end: 2, stroke: (thickness: 0.05em)),
//       [Name], [Description], [Size ($mu$m)],
//     ),
//     midrule,
//     [Dendrite], [Input terminal ], [$~100$],
//     [Axon    ], [Output terminal], [$~10$],
//     [Soma    ], [Cell body      ], [up to $10^6$],
//     botrule,
//   ),  // TODO(@daskol): Fix gutter between rows in body.
// ) <sample-table>

