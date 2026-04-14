---
title: "Advanced AI in Metamaterial Design — A Comprehensive Review (Draft)"
status: draft
created: 2026-04-13
tags: [metamaterial, AI, deep-learning, LLM, diffusion, transformer, VAE, review]
source_collection: "Zotero → ML Metamaterial → new ML + Metamaterial (GQBTWALC)"
outline: "[[Advanced AI in Metamaterial Design - Review Outline]]"
---

# Advanced AI in Metamaterial Design — A Comprehensive Review

> **Draft v0.1** — structured narrative grounded in the Zotero subcollection `new ML + Metamaterial` (≈291 items). Each claim is anchored to a specific paper by its Zotero item key (e.g. [1]). Sections §§4–7 are the intended technical core; §3 compresses classical/early ML into a single historical section at the user's request.

## Abstract

Metamaterials — artificial composites whose effective properties are controlled by their unit-cell architecture rather than their chemistry — have evolved from a niche of electromagnetic cloaking into a design-dominated paradigm spanning mechanical, acoustic, thermal, and photonic domains. The combinatorial vastness of their design space, coupled with the ill-posedness of their inverse problems, has made AI the dominant engine of progress over the last five years. This review synthesises the state of the art as of 2025–2026, with an explicit emphasis on *advanced* generative and foundation-model methods — variational and graph autoencoders, transformers, diffusion models, and large language models / agents — and treats earlier surrogate and tandem-network approaches only as necessary historical context. Drawing on ≈291 papers from the curated `new ML + Metamaterial` subcollection, we argue that the field has undergone three rapid shifts: (i) from discriminative surrogates to *generative latent design*, (ii) from single-modality regressors to *multi-modal, physics-aware generative transformers and diffusion models*, and (iii) from static networks to *agentic, LLM-driven co-design systems* that coordinate tools, simulators, literature, and experiments. We close with open challenges in data heterogeneity, benchmarking, manufacturability, and the prospects for a domain-pretrained "metamaterial foundation model."

---

## 1. Introduction

### 1.1 Metamaterials as a design problem

A metamaterial is a composite whose macroscopic response is dominated not by its chemistry but by the geometry of its periodic or quasi-periodic unit cell [2]. Since the first demonstrations of negative refractive index and perfect lensing [3], [4] and flat metaoptics [5], the field has expanded into acoustic and phononic media [6], [7], [8], mechanical media with auxetic, pentamode, and programmable responses [9], [10], [11], thermal cloaks and rectifiers [12], and shape-morphing and active architectures [13], [2]. Across all of these domains a common bottleneck persists: *designing* a unit cell, or a multiscale assembly of unit cells, whose architecture realises a target effective response.

This design problem is hard for four reasons that motivate every method reviewed below:

1. **Combinatorial design space.** Even 2D pixelated unit cells at modest resolution explode into $2^N$ topologies; 3D voxel or graph representations push this further. Truss metamaterials alone admit over 1.8 million topologies in a single systematic database [14], and triply periodic minimal surface (TPMS) architectures yield on the order of $7^{27}$ design possibilities [15].
2. **Expensive forward solvers.** FEM, FDTD, Bloch-wave, or atomistic simulations can take minutes to hours per candidate, making exhaustive search or classical gradient-based optimisation impractical for high-dimensional designs.
3. **Ill-posed inverse maps.** Many architectures can satisfy the same target property, so purely discriminative regression is fundamentally inadequate [16], [17], [18].
4. **Multi-scale, multi-physics coupling and manufacturability.** Real devices must satisfy stiffness *and* stability *and* dispersion *and* 3D-printability constraints simultaneously, and must survive the fidelity loss between simulation and fabrication [19], [20].

### 1.2 Why "advanced" AI, and why now

Recent reviews converge on the observation that the dominant AI paradigms in metamaterials are no longer simple surrogates. [21] frames a bidirectional relationship between intelligent metamaterials and metamaterials intelligence; [22] speaks of a "transformational dive" of nanophotonics inverse design from deep learning toward artificial general intelligence; [17] and [18] provide holistic cross-domain surveys that foreground generative, physics-informed, and foundation-model methods. Four forcing functions underlie this shift:

- **Data scale.** Systematic databases such as the 1.8-million-entry truss corpus of [14] and the benchmark suite of [20] now support pretraining at a scale comparable to natural-image domains.
- **Generative modelling.** VAEs, GANs, normalizing flows, and especially diffusion models [23], [24], [25], [1] convert the ill-posed inverse problem into *sampling* from a learned, conditional prior.
- **Transformers and foundation models.** Attention-based architectures [26], [27], [28] and autoregressive language / multimodal models [29], [30], [31], [32] bring universal sequence modelling and pretraining to metamaterial design.
- **Agentic systems.** Multi-LLM frameworks such as MechAgents [33], SciAgents [34], MetaScientist [35], CrossMatAgent [36], [37], and the photonics agent of [38] turn LLMs into orchestrators of tools, surrogates, simulators, and experiments.

### 1.3 Scope and structure

We deliberately concentrate on methods that have emerged or matured in the last ~3 years. §2 fixes the taxonomy and representations. §3 is a single compact section on classical and early deep-learning baselines, included only to establish vocabulary and motivate the rest of the paper. §§4–7 form the technical core: variational and graph autoencoders (§4), transformers and attention models (§5), diffusion and flow-based generative models (§6), and LLMs / foundation models / agentic design (§7). §8 covers cross-cutting physics-aware and differentiable learning. §9 treats sim-to-hardware pipelines, §10 domain-specific applications, §11 benchmarks and datasets, §12 open challenges, and §13 an outlook.

We make no attempt to be historical or exhaustive on pre-2021 literature; readers looking for that can consult [39], [40], [41], [42], and [43], which cover the classical inverse-design wave in nanophotonics.

---

## 2. Background and taxonomy

### 2.1 Physical domains and what AI is asked to optimise

The `new ML + Metamaterial` subcollection spans five broad physical regimes:

- **Mechanical metamaterials** — effective stiffness, Poisson's ratio, nonlinear stress–strain response, energy absorption, fracture, toughness, and programmable deformation [10], [11], [16], [24], [25]. Truss, plate, shell, TPMS, spinodoid, and origami/kirigami unit cells dominate [14], [15], [44].
- **Phononic and acoustic metamaterials** — band gaps, dispersion engineering, absorption, sound cloaking, and elastic wave control [45], [46], [47], [48], [49], [50], [51], [52].
- **Electromagnetic / photonic metamaterials and metasurfaces** — absorption spectra, polarization, holography, beam steering, chirality, and biosensing [21], [53], [54], [55], [56], [57], [58], [59].
- **Thermal metamaterials and coupled multiphysics** — cloaks, thermal rectifiers, radiative cooling, heat-enhanced transport [60], [61], [62], [63].
- **Intelligent / active metamaterials** — reconfigurable, programmable, and computing media; wave-based neural networks and in-physics logic [21], [64], [65], [66].

### 2.2 The inverse problem, restated

A unit cell is a topology $x \in \mathcal{X}$ whose forward map $f_\theta: \mathcal{X} \to \mathcal{Y}$ yields a physical response $y \in \mathcal{Y}$ (scalar moduli, full dispersion curves, spectra, stress–strain paths, or field distributions). The inverse problem seeks $x$ such that $f(x) \approx y^\star$ for a target $y^\star$, subject to manufacturability and symmetry constraints. This map is (a) non-unique — many topologies can realise the same target, (b) non-differentiable in standard representations, and (c) expensive to evaluate pointwise. The central move of advanced AI in this field is to replace the deterministic inversion $y^\star \mapsto x$ with *conditional sampling* from $p(x \mid y^\star)$ under a learned generative prior [16], [17], [67].

### 2.3 Unit-cell representations

Representation choice strongly constrains which AI models are applicable. The subcollection exhibits five dominant families, summarised here because they recur through §§3–7:

- **Pixel / voxel grids** — natural for CNNs and image-style diffusion models [68], [25], [24], [69].
- **Graph representations** — nodes as joints, edges as struts, used for GNNs, graph transformers, and graph-VAEs [70], [71], [72], [73], [74].
- **Implicit neural representations and signed distance functions** — smooth boundaries amenable to FE and 3D printing [28], [75], [76], [77], [78], [79], [80].
- **Parametric / algebraic tokenisations** — compact "mathematical sentences" that describe 3D shells or lattices as sequences of algebraic operations, enabling direct application of transformers and diffusion-transformer hybrids [1], [81].
- **Text / natural-language descriptions** — treated as tokens by LLMs for direct inverse design or for orchestrating downstream generators [29], [82], [83], [84], [74].

### 2.4 A taxonomy of AI approaches

We organise the methods reviewed below into four layers, weighted heavily toward the advanced-AI core:

| Layer | Representative methods | Sections |
|---|---|---|
| Legacy / foundational | MLP, CNN, GNN, tandem networks, vanilla GAN, classical RL | §3 (brief) |
| Advanced generative core | VAEs, (V)GAEs, Transformers, Diffusion, Flow matching | §§4–6 |
| Foundation & agentic layer | LLMs, vision–language models, multi-agent systems, foundation models | §7 |
| Cross-cutting | Physics-informed NNs, neural operators, differentiable simulators, equivariant nets, Bayesian / active learning | §8 |

---

## 3. Classical and early deep-learning baselines *(brief)*

The pre-2022 literature on ML for metamaterials is vast but, from the standpoint of this review, largely serves to motivate the advanced methods. We compress it into three bullets.

### 3.1 Forward surrogates

MLP and CNN property predictors for effective moduli, dispersion, absorption, and scattering appear in both mechanical [85], [70], [86], [87], and photonic [39], [40], [41], [88], [89], [90] domains. Graph neural networks quickly supplanted CNNs for truss and lattice topologies because they respect permutation and topological symmetries [70], [85], [71], [73]. Neural operators (FNO, DeepONet) then generalised surrogates from fixed-resolution to parametric PDEs, later re-appearing as the forward component of diffusion-based inverse pipelines [91], [92], [93], [94].

### 3.2 Early inverse design

The earliest inverse-design pipelines were direct regression from target property to structure, often unstable because of the one-to-many mapping [95], [96]. Tandem networks partially addressed this by routing an inverse network through a frozen forward surrogate [89], [90]. Vanilla (conditional) GANs produced the first truly generative inverse designs for freeform metasurfaces and auxetics [97], [98], [99], [100], [101], [102]. Direct CNN-as-optimiser approaches predicted optimal 2D microstructures in a single forward pass [68], [103]. Classical RL was used for sequential placement of scatterers [104], [105] and layer stacks [106].

### 3.3 Why these methods hit a ceiling

Four limitations motivated the generative turn: (a) brittle extrapolation beyond the training distribution, especially in nonlinear mechanics [24]; (b) failure to express *multiple valid solutions* to an ill-posed inverse, which tandem networks only paper over [16]; (c) topology lock-in to one lattice family or parameterisation [70]; and (d) absence of a reusable *prior* that could be fine-tuned across tasks, representations, or physical domains — the gap now being filled by foundation models [22], [29], [107].

---

# Part II — Advanced AI for metamaterial design

## 4. Variational and graph autoencoders: latent-space design

### 4.1 Why VAEs fit metamaterial design

A variational autoencoder provides three ingredients that map almost one-to-one onto metamaterial design: (i) a *continuous, differentiable* latent space usable as a design variable; (ii) a *structured* prior that regularises interpolation and extrapolation; and (iii) a natural coupling to *property regressors* so that geometry and physics share a representation. The review [16] explicitly identifies VAE-based latent design as one of three dominant deep-learning paradigms in mechanical metamaterials, alongside direct prediction and GAN-based inverse design.

### 4.2 The canonical VAE + property-regressor pattern

The canonical pattern — train a VAE over microstructure images or voxels jointly with a property predictor on the latent — is demonstrated across physical domains:

- **Mechanical.** [67] develops a probabilistic representation and inverse-design framework for metamaterials based on a deep generative model with a semi-supervised learning strategy, showing that the latent space supports uncertainty-aware generation. [108] extends this to *interpretable* generative inverse design for functional responses.
- **Photonic.** [109] uses manifold learning on the latent space of photonic nanostructures to expose smooth property landscapes and enable knowledge discovery.
- **Nonlinear / disentangled.** [110] employs disentangled generative models to investigate static and dynamic behaviours in 3D chiral mechanical metamaterials — separating rotation, size, and chirality axes in latent coordinates.

### 4.3 Graph autoencoders for lattice and porous metamaterials

Where unit cells are naturally graphs (trusses, porous networks), a variational *graph* autoencoder (VGAE) is a strictly better inductive bias than image-VAEs. [74] / [111] reconstructs and generates porous metamaterial units via a VGAE, and couples it to an LLM that converts unit cells into a text-based format, bridging geometry and language. [112] uses a generative GNN to design *connectivity-guaranteed* porous units — injecting a manufacturability constraint directly into generation. [72] designs metamaterials with programmable nonlinear responses and geometric constraints in *graph space*.

### 4.4 Latent-space operations as design tools

The real payoff of the VAE pattern is operational: once geometry and physics share a latent space, design becomes *latent arithmetic*. Interpolation along latent axes produces graded metamaterial families; gradient ascent on a latent property regressor performs inverse design without retraining; latent-space graph search composes unit cells into compatible multiscale assemblies. This paradigm is generalised by the unified modal-alignment framework of [113] (UniMate), which aligns three metamaterial modalities — 3D topology, density condition, and mechanical property — in a shared latent, and couples it to a synergetic diffusion generator; UniMate reports up to 80 % gains on topology generation relative to baselines, 5 % on property prediction, and 50 % on condition confirmation.

### 4.5 Limitations of VAEs and the handoff to diffusion

VAE samples are notoriously blurry; mode coverage is partial; and physical constraints are enforced only indirectly through the regressor. This is precisely why the field has increasingly stacked diffusion models *on top of* a VAE-style latent (latent diffusion, §6). We return to this handoff explicitly in §4→§6.

---

## 5. Transformers and attention-based models

### 5.1 Why attention matters for metamaterials

Transformers are attractive for three metamaterial-specific reasons. First, unit cells exhibit *long-range dependencies* — a defect or node in one corner can alter the global modulus — that CNN receptive fields struggle to capture at inference time. Second, attention naturally *conditions on heterogeneous inputs*: target spectra, nonlinear stress–strain curves, manufacturability masks, or even text can all be stacked as token streams. Third, [114] formalises the view that transformers *are* graph neural networks over a fully-connected token graph, unifying the graph and sequence perspectives on lattice representations.

### 5.2 Vision / spectrum transformers as forward surrogates

[26] introduces a **metamaterial spectrum transformer (MST)** for broadband solar absorber design, exploiting attention over wavelength-indexed tokens to capture the long-range couplings that defeated prior CNN surrogates. [27] develops **Metaformer**, a transformer-based explainable model for Q-BIC all-dielectric metasurface sensors; a spectrum-splitting scheme achieves 99 % prediction accuracy with 99 % fewer parameters, and the *meta-attention* maps provide physics insight into resonance formation. Broader deep-learning reviews [55], [57], [58], [115] now single out transformer backbones as the de facto standard for metasurface surrogates.

### 5.3 Neural-operator transformers

An important hybrid is the *neural operator transformer (NOT)* of [28], which handles signed-distance-function (SDF) metamaterial representations. The NOT predicts homogenised stress–strain curves and local field distributions on irregular query meshes, acting as a mesh-free surrogate coupled to a downstream diffusion model for inverse design. This paper is also the first in the subcollection to close the loop: SDF representation → neural-operator transformer forward → conditional diffusion inverse → FE / 3D printing ready.

### 5.4 Sequence, token, and algebraic-language models

A parallel track tokenises geometry itself and treats design as sequence generation:

- [1] (DiffuMeta) encodes 3D shell metamaterials as **algebraic language expressions** ("mathematical sentences") and applies a diffusion *transformer* to generate novel shells with targeted stress–strain responses under large deformations, including buckling and contact. Crucially, the transformer handles the one-to-many inverse map by generating diverse solutions, and supports simultaneous control over multiple mechanical objectives including nonlinear responses beyond the training domain.
- [29] (OptoGPT) is a **decoder-only transformer** for optical multilayer thin-film inverse design, trained to generate layer sequences under varying targets, materials, angles, and polarizations — an LLM-style foundation model within a narrow physical domain.
- [116] demonstrates autoregressive LLMs for crystal-structure generation, and [117] (AtomGPT) applies the same paradigm as a forward / inverse transformer for materials.

### 5.5 Graph and equivariant transformers

[73] introduces *similarity-equivariant* GNNs for metamaterial homogenization, and graph transformer variants extend this further to handle periodic boundary conditions and symmetry groups. These are increasingly being adopted as the backbone surrogate in closed-loop agentic systems (§7) because they generalise across lattice families without retraining.

### 5.6 Open questions

Data scaling laws for metamaterial transformers are essentially unknown; tokenisation of 3D geometry remains a bottleneck (voxel, graph, SDF, or algebraic language); and injecting hard physical constraints into attention layers is an active area, currently handled via classifier-free guidance (§6) or agentic post-checks (§7).

---

## 6. Diffusion and flow-based generative models

### 6.1 Why diffusion is a step-change

Denoising diffusion probabilistic models (DDPMs) and their score-based siblings address three limitations that bottlenecked earlier GANs and VAEs: (i) higher sample fidelity and mode coverage; (ii) stable training without adversarial pathologies; and (iii) a native mechanism for conditioning via classifier-free guidance. [69] makes this argument explicitly for metasurface inverse design, and shows that a diffusion probabilistic model "avoids the model instability introduced by the adversarial training process of GANs and ensures more accurate and high-quality generation results." In the space of 2–3 years, diffusion has become the dominant generative engine in the subcollection — spanning voxel, SDF, graph, and algebraic-sentence representations.

### 6.2 Diffusion on voxel and density-based microstructures

[25] (Guided Diffusion) generates voxel microstructures at 128³ resolution to approach specified homogenized tensor targets *in three seconds*, enabling rapid exploration of extreme metamaterials, sequence interpolation, and multi-scale microstructure families. [118] (GLU3D) generates complex lattice unit cells conditioned on desired mechanical properties, explicitly targeting additive-manufacturing compatibility. [119] demonstrates a denoising-diffusion algorithm for inverse design of microstructures with fine-tuned nonlinear material properties. [120] applies a 3D conditional diffusion model with data augmentation to metamaterial inverse design. [121] scales atom-by-atom inverse design with nano-topology optimisation and diffusion models.

### 6.3 Video and full-field diffusion for nonlinear response

A striking example of the diffusion paradigm's reach is [24], which trains **video denoising diffusion models** on full-field data of periodic stochastic cellular structures to inversely design nonlinear mechanical metamaterials under large strain — buckling and contact included. The key methodological insight: rather than map property directly to design, the model also predicts the expected *deformation path* and full-field internal stress distribution, providing physical interpretability and closely matching FE simulations. [122] reports a parallel study on the same topic.

### 6.4 SDF and algebraic-sentence diffusion

Binary pixel representations produce jagged edges that hinder both FE analysis and 3D printing. Two complementary solutions appear in the subcollection. [28] pairs a classifier-free guided diffusion over signed-distance functions with a neural-operator transformer forward surrogate, producing smooth, fabricable geometries. [1] (DiffuMeta) instead abstracts the geometry into an algebraic "sentence" and runs a diffusion transformer over the sentence tokens, reaching simultaneous control over multiple mechanical objectives and robustness under large deformations. These two papers together arguably define the current state of the art in mechanical-metamaterial inverse design.

### 6.5 Physics-guided and active-learning-augmented diffusion

Diffusion naturally admits *guidance*: one can steer sampling with classifier or classifier-free gradients, or plug a differentiable simulator in as a likelihood. [123] demonstrates an **active-learning-augmented diffusion model** for high-asymmetry terahertz metasurfaces, closing the loop between diffusion sampling, simulation, and dataset expansion. [124] couples Bayesian active learning to diffusion for broadband polarization-insensitive metasurfaces. [125] introduces a hybrid *conditional diffusion–DeepONet* framework for high-fidelity stress prediction in hyperelastic materials — a template for diffusion-plus-neural-operator stacks.

### 6.6 Diffusion beyond crystals: amorphous and disordered media

[23] presents a generative diffusion model for *amorphous* materials — a regime where crystal-based generative baselines fail. The model generates amorphous structures up to three orders of magnitude faster than molecular dynamics, recovers short- and medium-range order, reproduces macroscopic properties, and discovers a ductile-to-brittle transition in silica under conditional sampling. Parallel work on disordered metamaterials uses self-supervised learning to decode and design disordered architectures [126].

### 6.7 Flow matching as an alternative

[127] applies flow matching to accelerated simulation of atomic transport — a competing framework that trades the Markov-chain derivation of diffusion for continuous vector-field learning, with potentially better sample efficiency and exact likelihoods. Its uptake in metamaterial design is nascent but likely.

### 6.8 Open questions

High-resolution 3D diffusion at engineering scale, equivariant diffusion over crystallographic symmetry groups, physical-constraint injection beyond classifier-free guidance, and the compute cost of diffusion relative to latent-VAE and transformer alternatives all remain open.

---

## 7. LLMs, foundation models, and agentic design

§7 is, in our view, the most transformative of the advanced-AI layers, and occupies the largest share of the 2024–2026 primary literature in the subcollection. We divide it into five tracks.

### 7.1 LLMs as text-conditioned forward / inverse models

The simplest use of an LLM is as an architecture-agnostic, text-conditioned regressor or generator. [84] / [128] fine-tunes ChatGPT on up to 40,000 all-dielectric metamaterial data points and shows that the fine-tuned LLM *matches* a deep neural network at predicting absorptivity spectra from text-described geometry, and can also be prompted for inverse design. [82] uses LLMs for inverse design of optical multilayer thin films and metasurfaces. [83] extends this to acoustic metamaterials for sound absorption, explicitly motivating the approach as a way to eliminate the need for specialised ML/coding expertise — a "universal user-friendly strategy" for the domain. [129] fine-tunes a language model that directly generates stable inorganic materials as text, pointing toward a universal text-first materials design workflow.

The scaling of this track is illustrated by **OptoGPT** [29], which treats nanophotonic multilayer stacks as token sequences and trains a decoder-only transformer that adapts quickly to new targets, angles, and polarizations — the first paper in the subcollection to explicitly claim a "foundation model" for a metamaterial domain. [107] (MatterGen), though broader than metamaterials, provides the best-established general-purpose materials foundation model for stable inorganic crystals, with property-conditioned fine-tuning that can be imported wholesale into metamaterial pipelines.

### 7.2 Domain-specialised mechanics language models

Beyond thin wrappers around frontier LLMs, a line of research trains genuine mechanics foundation models:

- **MeLM** [30] is a multimodal mechanics language model handling instructions, numbers, and microstructure data, applied to hierarchical honeycomb design, carbon nanotube mechanics, and protein unfolding. Its autoregressive attention effectively represents a "multi-particle system" whose interaction potentials emerge from self-attention.
- **MechGPT** [31] / [130] is a fine-tuned LLM on multiscale materials failure that supports question-answering, knowledge retrieval, hypothesis generation, and ontological knowledge graph construction, with up to 70 B parameters and 10 k-token context.
- **Cephalo** [32] is a multimodal vision-language model for bio-inspired materials analysis and design, with an image-to-text-to-3D pipeline and demonstrated generation of pollen-inspired architected materials and microstructures from a photograph of a solar eclipse.

### 7.3 Agents: LLMs orchestrating tools, simulators, and experiments

The most ambitious line of work replaces the *single-LLM-as-generator* with *multi-agent systems* that orchestrate generative models, surrogates, simulators, topology optimisers, and literature retrieval.

- **MechAgents** [33]: two-to-many LLM agents collaborate to write, execute, and self-correct FE code to solve classical elasticity problems in various geometries, boundary conditions, small/finite deformation, and linear/hyperelastic regimes.
- **SciAgents** [34]: combines ontological knowledge graphs, a suite of LLMs, and multi-agent systems with in-situ learning to autonomously generate and refine hypotheses for biologically inspired materials, revealing hidden interdisciplinary connections.
- **MetaScientist** [35] / [131]: a human-in-the-loop system with two phases — (i) hypothesis generation via domain-specific foundation models and literature-derived inductive biases, (ii) 3D structure synthesis via a novel 3D diffusion model conditioned on textual hypotheses, with LLM-based refinement. Explicitly integrates expert validation at each step.
- **CrossMatAgent** [37] / [36]: a hierarchical multi-agent framework combining GPT-4o multimodal reasoning, DALL-E 3, and fine-tuned Stable Diffusion XL; agents specialise in pattern analysis, architectural synthesis, prompt engineering, and supervisory feedback to produce simulation- and 3D-printing-ready metamaterial patterns, with evaluation via CLIP alignment, SHAP interpretability, and mechanical simulations.
- **Autonomous photonic metamaterial agent** [38]: an agent that, given a desired optical spectrum, autonomously proposes and develops a forward DL surrogate, invokes external APIs for optimisation, uses memory, and generates a final design via a deep inverse method.
- **Agentic graph reasoning** [132] yields self-organising knowledge networks — positioning agent systems as knowledge-building as well as design-building tools.

[133] illustrates a complementary pattern: the LLM does not *generate the design* but *controls the optimiser*. A GPT-based agent acts as an online adaptive controller for SIMP topology optimisation, reading structured observations (compliance, grayness, stagnation, checkerboard measure) and outputting continuation parameters in real time, beating fixed-schedule baselines by 5.7–18.1 % on benchmark problems. A second LLM pass meta-optimises the agent's call frequency. This is plausibly a template for many more "LLM-as-controller" hybrids.

### 7.4 Knowledge graphs, retrieval, and literature-scale reasoning

MechGPT and SciAgents both underline the value of ontological knowledge graphs as first-class intermediate representations. [132] pushes this further with agentic deep graph reasoning, and general materials-science perspectives [134], [135], [136] argue that retrieval-augmented generation grounded in curated literature will be indispensable for any trustworthy metamaterial LLM. Automation of data ingestion from papers — exactly the task Cephalo is built for — is a prerequisite for scaling this track.

### 7.5 Toward a metamaterial foundation model

Across these five threads, a concrete research agenda emerges: pretrain a multimodal (geometry + spectrum/response + text) foundation model on the unified datasets of [20] and [14], with graph, SDF, and algebraic-sentence tokenisers as complementary front-ends, and expose it to downstream fine-tuning for mechanical, acoustic, thermal, and photonic specialisations. OptoGPT [29] and UniMate [113] are the closest extant approximations in photonics and mechanics respectively; a truly cross-domain "metamaterial GPT" does not yet exist.

### 7.6 Risks and limitations

LLM agents hallucinate; they generate unphysical geometries when their tool grounding is weak; their evaluation gaps are severe (few benchmarks measure *scientific correctness* rather than formatting); and their compute and carbon costs are non-trivial. Every paper in §7.3 that reports quantitative gains also reports human-in-the-loop validation, suggesting that fully autonomous metamaterial design remains aspirational.

---

## 8. Physics-aware and differentiable learning *(cross-cutting)*

Physics-aware methods cut across §§4–7 rather than competing with them. The subcollection shows them in three flavours.

### 8.1 Physics-informed neural networks

PINNs enforce PDE residuals as soft penalties during training. In metamaterials they have been applied to inverse problems in nano-optics [137], optical metamaterials [138], and artificial electromagnetic materials more broadly [139]. The general PIML survey [140] remains the standard reference. [141] introduces a *Lorentz* neural network that learns the physics of all-dielectric metamaterials by building the Lorentz oscillator model into the architecture.

### 8.2 Neural operators and differentiable surrogates

Neural operators [91], [92], [93], [142], [143] generalise surrogates from fixed-resolution grids to parametric PDEs and are increasingly used as the *forward* component of diffusion-based inverse pipelines (e.g. [28]). Differentiable FEM / FDTD simulators enable gradient-based topology optimisation through learned priors and slot naturally beneath any of the generative methods in §§4–6. Reliable, efficient, scalable photonic inverse design [144] is an exemplar of this stack.

### 8.3 Equivariance, symmetry, and constitutive hypernetworks

Equivariant networks [73] respect the symmetry group of periodic unit cells and improve data efficiency dramatically. Hypernetworks [145], [146], [147] parameterise *model weights as functions of the metamaterial design*, enabling a single master network to instantiate many constitutive laws. **HyperCAN** [148] is the leading metamaterial-specific example: a hypernetwork dynamically adjusts the parameters of an input-convex neural network that models the nonlinear stress–strain map of a truss lattice, generalising to unseen lattice topologies and loading scenarios and providing large computational savings in multiscale simulation. The hypernetwork literature in the subcollection [149], [150], [151], [152], [153], [154], [155] suggests this paradigm is poised for rapid adoption.

### 8.4 Active learning and Bayesian optimisation

Although not "advanced AI" in the foundation-model sense, Bayesian active learning provides the *data-efficient* scaffolding for all the generative methods above. [156] demonstrates high-dimensional BO for metamaterial design; [157] couples multi-fidelity BO to architected materials design; [158] combines deep neural networks with BO for on-demand inverse design. [15] pairs Monte Carlo Tree Search with CNNs and FEM to explore a $7^{27}$ TPMS design space starting from only 100 initial samples, beating Bayesian optimisation on both final stiffness (+30 %) and strength (+20 %).

---

## 9. From simulation to hardware: closing the loop

### 9.1 Manufacturability as a learning target

Generative models that ignore manufacturability produce beautiful FE-ready geometries that cannot be printed. The subcollection shows increasing emphasis on baking AM constraints directly into training or sampling: [19] (manufacturability-aware deep generative design of 3D metamaterial units for AM), [112] (connectivity-guaranteed porous units via generative GNNs), [159] (TPMS metamaterial via DL plus topology optimization), and [28]'s SDF + diffusion pipeline (smooth boundaries for 3D printing).

### 9.2 End-to-end workflows

[160] enhances high-degree-of-freedom meta-atom design precision and speed with a tandem generative network. [161] combines ML and evolutionary algorithms to design 4D-printed active composite structures. [162] discusses 3D and 4D printing of electromagnetic metamaterials. On the mechanics side, [87] supports rare-event rational design of 3D-printed multi-material mechanical metamaterials, and [163] reviews deep-learning-assisted design of mechanical metamaterials end-to-end.

### 9.3 Autonomous / self-driving labs

Agentic systems from §7 are natural orchestrators for self-driving fabrication and measurement. [37] / [36] already produce simulation- *and* 3D-printing-ready outputs; coupling these to robotic printers and measurement rigs is a near-term extension. [38] provides a template for API-mediated agent control of external tools.

### 9.4 The sim-to-real gap

Few papers in the subcollection report systematic experimental validation at scale; those that do — e.g. [1] (DiffuMeta, experimentally validated) and [164] in the parent collection — consistently report strong qualitative but imperfect quantitative agreement with simulation, pointing to domain adaptation and uncertainty quantification as the next research frontier.

---

## 10. Applications across physical domains

### 10.1 Mechanical metamaterials

Auxetics, tunable stiffness, energy absorption, toughness, programmable deformation, and soft robotics are all represented. Highlights include the DiffuMeta pipeline for shell metamaterials [1], the MCTS-AL TPMS pipeline [15], the truss-property atlas and database [14], and applications to chiral mechanical metamaterials [110], kirigami [44], [165], rigid torque transmission in soft robots [166], double-network-inspired metamaterials [167], and granular entanglement [168]. Reviews [10], [16], [163], [169] provide cross-cutting surveys.

### 10.2 Acoustic and phononic metamaterials

[45] and [46] provide focused reviews. Primary work covers acoustic absorption [49], structure-borne noise [50], phononic band gaps [47], [51], 2D phononic crystal dispersion engineering [52], elastic wave control via digital materials [170], dispersion prediction and inverse design of elastic metamaterials [171], and ultrathin acoustic absorbing metasurfaces [48]. [83] is the first LLM-driven acoustic-metamaterial design paper in the subcollection.

### 10.3 Electromagnetic metamaterials and metasurfaces

The largest application slice. Reviews [54], [53], [55], [57], [58], [115], [172], [173], [56] give comprehensive entry points. Primary work spans broadband absorbers [26], biosensing [59], multi-functional absorbers [174], structural colour [100], chiral plasmonics [175], radiative cooling [63], terahertz resonance [123], metasurface antennas and holography [115], [58], and reconfigurable surfaces [66], [65].

### 10.4 Thermal and multiphysics metamaterials

[61] provides a ML-focused review; [60] and [62] cover deep-learning-driven design of thermal and active thermal metamaterials. [12] surveys thermal camouflaging metamaterials. Multiphysics coupling is exemplified by [176] (frequency transfer and inverse design under multi-physics coupling).

### 10.5 Biomedical, wearable, and programmable applications

[177] develops inverse co-design of mechanical and sensory properties in soft lattice foams for multifunctional wearables; [178] designs shape-conformal porous frameworks for high-resolution neural organoid electrophysiology; [179] explores multi-mechanical regulation of 3D-printed TPMS surfaces via Fourier synthesis.

---

## 11. Benchmarks, datasets, and reproducibility

The honest state of benchmarking is still poor. [20] (MetamatBench) is the first serious attempt at a unified framework: it integrates five heterogeneous multimodal datasets, adapts 17 state-of-the-art ML methods, ships 12 novel performance metrics with FE-based validation, and provides a visual-interactive UI targeting both ML and non-ML researchers. Its authors explicitly frame three challenges — *data heterogeneity*, *model complexity*, and *dual black-box human–AI collaboration* — that the community must solve collectively.

Complementary large-scale datasets include the 1.8-million-entry truss database of [14], the METASET framework [180] for exploring shape and property spaces, and the architected-material databases underlying [181], [182], and the cubic-symmetric extreme-property atlas of [183].

What is still missing: standard splits for generative inverse design (not just prediction), task families that test out-of-distribution generalisation, manufacturability scoring as a first-class metric, and shared pretrained checkpoints that advance the foundation-model agenda of §7.

---

## 12. Open challenges and research agenda

1. **Generalisation across topology, scale, and physics.** Most models are tied to one lattice family or one physical domain. Pretraining on cross-domain corpora (metamaterials of different physics sharing a geometry language) is the most promising path; [74], [113], [29], [107] hint at the form.
2. **Multi-objective, multi-physics trade-offs.** [176] and [92] begin the conversation, but few works report Pareto-aware generative sampling.
3. **Interpretability and physical insight.** Attention maps in [27], latent-space arithmetic in [110], and ontological knowledge graphs in [31], [34] all offer partial windows into the learned physics. Symbolic regression on latent features remains essentially unexplored.
4. **Data efficiency.** Self-supervised [126], active-learning-augmented diffusion [123], and MCTS-AL [15] point the way; combined with pretrained backbones they may bring the per-design data cost into the hundreds of samples.
5. **Manufacturability and sim-to-real gap.** Guarantees at sampling time, uncertainty quantification, and integration with in-situ inspection must graduate from add-ons to primary design variables.
6. **Benchmarks.** MetamatBench [20] is a start; a community effort is needed.
7. **A metamaterial foundation model.** The community should settle on a geometry tokenisation, a pretraining corpus, and a cross-domain evaluation suite. OptoGPT, UniMate, and MatterGen provide feasibility evidence from adjacent subdomains.
8. **Agent evaluation.** [184] (CORE: full-path evaluation of LLM agents) is an early template for evaluating multi-step agent behaviour; similar rigour must reach §7 metamaterial agents before they can be trusted unsupervised.
9. **Risk and robustness.** Hallucinated designs, unphysical outputs, and adversarial prompts all need dedicated defences. The LLM-attack literature ([185], in the parent LLM collection) is a warning.
10. **Human–AI co-design.** The practical interface between domain experts and agents — sketch in, constraints in, structure out — is still ad hoc; [186]'s human–AI schema discovery and the UI layer of MetamatBench are early entries.

---

## 13. Conclusion and outlook

The literature in `new ML + Metamaterial` tells a coherent story: the dominant design paradigm for metamaterials has shifted, in ~3 years, from discriminative surrogates to generative latent design, and now toward multimodal, physics-aware, agentic systems. VAEs and graph-VAEs provided the first *reusable* priors (§4). Transformers and attention models brought universal sequence modelling and explainable long-range dependencies (§5). Diffusion models supplied state-of-the-art sample quality, principled conditioning, and naturally handled one-to-many inverse maps (§6). And LLMs and multi-agent systems are collapsing the distance between *concept*, *design*, *simulation*, *fabrication*, and *literature* into a single orchestrated loop (§7).

Near term (1–3 years): (a) mechanical metamaterial inverse design will be dominated by SDF / algebraic-sentence diffusion pipelines in the [1] / [28] style; (b) metasurface design will increasingly run on transformer surrogates + active-learning-augmented diffusion [123], [26], [27]; (c) LLM-orchestrated agent frameworks [33], [34], [35], [37], [38] will become standard infrastructure for research workflows rather than research outputs in themselves; (d) [133]-style LLM-as-controller patterns will reach classical topology optimisers in production.

Long term (5–10 years): the community should target a genuine *metamaterial foundation model* — pretrained on unified cross-domain corpora with geometry, spectrum, field, and text modalities — that specialises via light fine-tuning to each physical application. The clearest prerequisites are: a shared benchmark (§11), manufacturability as a first-class modality (§9), and a cross-domain geometry tokeniser robust to trusses, shells, porous networks, and metasurface meta-atoms alike.

Call to action: open datasets, shared pretrained checkpoints, agent-evaluation benchmarks, and cross-domain collaboration between mechanics, photonics, acoustics, and ML communities.

---

## Appendix A — Anchor papers by section (with Zotero keys)

| §    | Focus                           | Anchor Zotero keys                                                                                                                                                                                                                                                                                                                                                                                            |
| ---- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1, 2 | Scope / background              | [2], [21], [17], [18], [16], [22], [173], [187], [54], [55], [57], [188], [163], [169], [189]                                                                                                                                                                                                                                                                                                                 |
| 3    | Classical baselines             | [39], [40], [41], [89], [90], [95], [96], [97], [98], [99], [100], [101], [102], [68], [103], [104], [105], [106], [85], [70], [86], [87], [88]                                                                                                                                                                                                                                                               |
| 4    | VAEs / graph autoencoders       | [67], [109], [110], [108], [74], [111], [112], [72], [113]                                                                                                                                                                                                                                                                                                                                                    |
| 5    | Transformers                    | [26], [27], [114], [28], [1], [29], [117], [116], [73]                                                                                                                                                                                                                                                                                                                                                        |
| 6    | Diffusion & flows               | [69], [25], [118], [119], [120], [24], [122], [28], [1], [123], [124], [125], [23], [126], [121], [127], [190], `DiffMat 9YE5DQHD`                                                                                                                                                                                                                                                                            |
| 7    | LLMs / foundation / agents      | [30], [31], [130], [32], [33], [34], [35], [131], [37], [36], [38], [132], [133], [84], [128], [82], [83], [29], [107], [117], [116], [129], [134], [135], [136]                                                                                                                                                                                                                                              |
| 8    | Physics-aware / ops / hypernets | [137], [138], [139], [140], [141], [91], [92], [93], [142], [143], [73], [148], [145], [146], [147], [149], [150], [151], [152], [153], [154], [155], [156], [157], [158], [15], [124]                                                                                                                                                                                                                        |
| 9    | Sim-to-hardware                 | [19], [112], [159], [28], [160], [161], [162], [87], [1], [38]                                                                                                                                                                                                                                                                                                                                                |
| 10   | Applications                    | Mechanical: [1], [15], [14], [110], [44], [165], [166], [167], [168], [10], [16], [163], [169]. Acoustic/phononic: [45], [46], [49], [50], [47], [51], [52], [170], [171], [48], [83]. EM/photonic: [54], [53], [55], [57], [58], [115], [172], [173], [56], [26], [59], [174], [100], [175], [63], [123], [66], [65]. Thermal/multiphysics: [61], [60], [62], [12], [176]. Bio/wearable: [177], [178], [179] |
| 11   | Benchmarks/datasets             | [20], [14], [180], [183], [181], [182]                                                                                                                                                                                                                                                                                                                                                                        |
| 12   | Open challenges                 | [74], [113], [29], [107], [176], [92], [27], [110], [31], [34], [126], [123], [15], [20], [184], [186]                                                                                                                                                                                                                                                                                                        |

---

## Appendix B — Outstanding tasks for the next revision

- [ ] Fetch and incorporate abstracts for the ~20 papers cited by key alone (no abstract yet read), especially [188], [45], [55], [57], [54], [191], [173], [53], [192], [193], [194], [195], [163], [196], [46], [169], [61], [58], [115], [189].
- [ ] Add explicit tables comparing: (a) generative model families across representations; (b) agent frameworks by orchestration pattern; (c) diffusion conditioning strategies.
- [ ] Pull figures where licensing allows — particularly from DiffuMeta, UniMate, MetamatBench, MechGPT, Cephalo.
- [ ] Decide on target venue and tighten §§1–3 to match its word budget.
- [ ] Draft §1 Introduction at full length once venue is chosen; current §1 is a framing sketch.
- [ ] Cross-reference this draft against the companion file `AI and Large Language Models in Metamaterial Design A Comprehensive Review (2020–2025).md` for overlap / complementarity.

---

## References

[1] DiffuMeta: Algebraic Language Models for Inverse Design of Metamaterials via Diffusion Transformers. (2025). [https://doi.org/10.48550/arXiv.2507.15753](https://doi.org/10.48550/arXiv.2507.15753)

[2] Mechanical metamaterials and beyond. *Nature Communications* (2023). [https://doi.org/10.1038/s41467-023-41679-8](https://doi.org/10.1038/s41467-023-41679-8)

[3] Negative refraction makes a perfect lens. *Physical Review Letters* (2000). [https://doi.org/10.1103/PhysRevLett.85.3966](https://doi.org/10.1103/PhysRevLett.85.3966)

[4] Metamaterials and Negative Refractive Index. *Science* (2004). [https://doi.org/10.1126/science.1096796](https://doi.org/10.1126/science.1096796)

[5] Flat optics with designer metasurfaces. *Nature Materials* (2014). [https://doi.org/10.1038/nmat3839](https://doi.org/10.1038/nmat3839)

[6] Acoustic metamaterials: From local resonances to broad horizons. *Science Advances* (2016). [https://doi.org/10.1126/sciadv.1501595](https://doi.org/10.1126/sciadv.1501595)

[7] Controlling sound with acoustic metamaterials. *Nature Reviews Materials* (2016). [https://doi.org/10.1038/natrevmats.2016.1](https://doi.org/10.1038/natrevmats.2016.1)

[8] Acoustic band structure of periodic elastic composites. *Physical Review Letters* (1993). [https://doi.org/10.1103/PhysRevLett.71.2022](https://doi.org/10.1103/PhysRevLett.71.2022)

[9] Flexible mechanical metamaterials. *Nature Reviews Materials* (2017). [https://doi.org/10.1038/natrevmats.2017.66](https://doi.org/10.1038/natrevmats.2017.66)

[10] Rational Designs of Mechanical Metamaterials: Formulations, Architectures, Tessellations and Prospects. *Materials Science & Engineering R: Reports* (2023). [https://doi.org/10.1016/j.mser.2023.100755](https://doi.org/10.1016/j.mser.2023.100755)

[11] Programmable Multi-Physical Mechanics of Mechanical Metamaterials. *Materials Science & Engineering R: Reports* (2023). [https://doi.org/10.1016/j.mser.2023.100745](https://doi.org/10.1016/j.mser.2023.100745)

[12] Thermal camouflaging metamaterials. *Materials Today* (2021). [https://doi.org/10.1016/j.mattod.2020.11.013](https://doi.org/10.1016/j.mattod.2020.11.013)

[13] Shape Morphing Metamaterials. (2025). [https://doi.org/10.48550/arXiv.2501.14804](https://doi.org/10.48550/arXiv.2501.14804)

[14] Mining extreme properties from a large metamaterial database. *Nature Communications* (2025). [https://doi.org/10.1038/s41467-025-64745-9](https://doi.org/10.1038/s41467-025-64745-9)

[15] Highly Efficient Discovery of 3D Mechanical Metamaterials via Monte Carlo Tree Search. *Advanced Science* (2025). [https://doi.org/10.1002/advs.202513771](https://doi.org/10.1002/advs.202513771)

[16] Deep learning in mechanical metamaterials: From prediction and generation to inverse design. *Advanced Materials* (2023). [https://doi.org/10.1002/adma.202302530](https://doi.org/10.1002/adma.202302530)

[17] Data-driven design for metamaterials and multiscale systems: A review. *Advanced Materials* (2024). [https://doi.org/10.1002/adma.202305254](https://doi.org/10.1002/adma.202305254)

[18] Machine intelligence in metamaterials design: a review. *Oxford Open Materials Science* (2024). [https://doi.org/10.1093/oxfmat/itae001](https://doi.org/10.1093/oxfmat/itae001)

[19] Manufacturability-aware deep generative design of 3D metamaterial units for additive manufacturing. *Structural and Multidisciplinary Optimization* (2024). [https://doi.org/10.1007/s00158-023-03732-4](https://doi.org/10.1007/s00158-023-03732-4)

[20] MetamatBench: Integrating Heterogeneous Data, Computational Tools, and Visual Interface for Metamaterial Discovery. (2025). [https://doi.org/10.48550/arXiv.2505.20299](https://doi.org/10.48550/arXiv.2505.20299)

[21] A guidance to intelligent metamaterials and metamaterials intelligence. *Nature Communications* (2025). [https://doi.org/10.1038/s41467-025-56122-3](https://doi.org/10.1038/s41467-025-56122-3)

[22] The transformational dive of nanophotonics inverse design from deep learning to artificial general intelligence. *APL Photonics* (2024). [https://doi.org/10.1063/5.0226592](https://doi.org/10.1063/5.0226592)

[23] A generative diffusion model for amorphous materials. *npj Computational Materials* (2025). [https://doi.org/10.1038/s41524-025-01901-1](https://doi.org/10.1038/s41524-025-01901-1)

[24] Inverse design of nonlinear mechanical metamaterials via video denoising diffusion models. *Nature Machine Intelligence* (2023). [https://doi.org/10.1038/s42256-023-00762-x](https://doi.org/10.1038/s42256-023-00762-x)

[25] Guided Diffusion for Fast Inverse Design of Density-based Mechanical Metamaterials. *arXiv* (2024). [https://doi.org/10.48550/arXiv.2401.13570](https://doi.org/10.48550/arXiv.2401.13570)

[26] Broadband solar metamaterial absorbers empowered by transformer-based deep learning. *Advanced Science* (2023). [https://doi.org/10.1002/advs.202206718](https://doi.org/10.1002/advs.202206718)

[27] Meta-Attention Deep Learning for Smart Development of Metasurface Sensors. *Advanced Science* (2024). [https://doi.org/10.1002/advs.202405750](https://doi.org/10.1002/advs.202405750)

[28] Toward signed distance function based metamaterial design: Neural operator transformer for forward prediction and diffusion model for inverse design. *Computer Methods in Applied Mechanics and Engineering* (2025). [https://doi.org/10.1016/j.cma.2025.118316](https://doi.org/10.1016/j.cma.2025.118316)

[29] OptoGPT: A foundation model for inverse design in optical multilayer thin film structures. *Opto-Electronic Advances* (2024). [https://doi.org/10.29026/oea.2024.240062](https://doi.org/10.29026/oea.2024.240062)

[30] MeLM, a generative pretrained language modeling framework that solves forward and inverse mechanics problems. *Journal of the Mechanics and Physics of Solids* (2023). [https://doi.org/10.1016/j.jmps.2023.105454](https://doi.org/10.1016/j.jmps.2023.105454)

[31] MechGPT, a Language-Based Strategy for Mechanics and Materials Modeling That Connects Knowledge Across Scales, Disciplines, and Modalities. *Applied Mechanics Reviews* (2024). [https://doi.org/10.1115/1.4063843](https://doi.org/10.1115/1.4063843)

[32] Cephalo: Multi‐Modal Vision‐Language Models for Bio‐Inspired Materials Analysis and Design. *Advanced Functional Materials* (2024). [https://doi.org/10.1002/adfm.202409531](https://doi.org/10.1002/adfm.202409531)

[33] MechAgents: Large language model multi-agent collaborations can solve mechanics problems, generate new data, and integrate knowledge. *Extreme Mechanics Letters* (2024). [https://doi.org/10.1016/j.eml.2024.102131](https://doi.org/10.1016/j.eml.2024.102131)

[34] SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning. (2024). [https://doi.org/10.48550/arXiv.2409.05556](https://doi.org/10.48550/arXiv.2409.05556)

[35] MetaScientist: A Human-AI Synergistic Framework for Automated Mechanical Metamaterial Design. (2024). [https://doi.org/10.48550/arXiv.2412.16270](https://doi.org/10.48550/arXiv.2412.16270)

[36] CrossMatAgent: AI-assisted design of manufacturable metamaterial patterns via multi-agent generative framework. *Advanced Intelligent Discovery* (2025). [https://doi.org/10.1002/aidi.202500063](https://doi.org/10.1002/aidi.202500063)

[37] A Multi-Agent Framework Integrating Large Language Models and Generative AI for Accelerated Metamaterial Design. (2025). [https://doi.org/10.48550/arXiv.2503.19889](https://doi.org/10.48550/arXiv.2503.19889)

[38] An agentic framework for autonomous metamaterial modeling and inverse design. *ACS Photonics* (2025). [https://doi.org/10.1021/acsphotonics.5c01514](https://doi.org/10.1021/acsphotonics.5c01514)

[39] Deep learning enabled inverse design in nanophotonics. *Nanophotonics* (2020). [https://doi.org/10.1515/nanoph-2019-0474](https://doi.org/10.1515/nanoph-2019-0474)

[40] Deep learning for the design of photonic structures. *Nature Photonics* (2021). [https://doi.org/10.1038/s41566-020-0685-y](https://doi.org/10.1038/s41566-020-0685-y)

[41] Deep learning in nano-photonics: inverse design and beyond. *Photonics Research* (2021). [https://doi.org/10.1364/PRJ.415960](https://doi.org/10.1364/PRJ.415960)

[42] Inverse design in nanophotonics. *Nature Photonics* (2018). [https://doi.org/10.1038/s41566-018-0246-9](https://doi.org/10.1038/s41566-018-0246-9)

[43] A newcomer’s guide to deep learning for inverse design in nano‐photonics. *Nanophotonics* (2023). [https://doi.org/10.1515/nanoph-2023-0527](https://doi.org/10.1515/nanoph-2023-0527)

[44] Textile Hinges Enable Extreme Properties of Kirigami Metamaterials. *Advanced Functional Materials* (2024). [https://doi.org/10.1002/adfm.202415986](https://doi.org/10.1002/adfm.202415986)

[45] Machine learning and deep learning in phononic crystals and metamaterials – A review. *Materials Today Communications* (2022). [https://doi.org/10.1016/j.mtcomm.2022.104606](https://doi.org/10.1016/j.mtcomm.2022.104606)

[46] Machine learning and deep learning in phononic crystals and metamaterials - A review. *Materials Today Communications* (2022). [https://doi.org/10.1016/j.mtcomm.2022.104606](https://doi.org/10.1016/j.mtcomm.2022.104606)

[47] Deep learning for the design of phononic crystals and elastic metamaterials. *Journal of Computational Design and Engineering* (2023). [https://doi.org/10.1093/jcde/qwad013](https://doi.org/10.1093/jcde/qwad013)

[48] Ultrathin acoustic absorbing metasurface based on deep learning approach. *Smart Materials and Structures* (2021). [https://doi.org/10.1088/1361-665x/ac0675](https://doi.org/10.1088/1361-665x/ac0675)

[49] Bio-inspired acoustic metamaterials for traffic noise control with machine learning. *Communications Engineering* (2025). [https://doi.org/10.1038/s44172-025-00470-x](https://doi.org/10.1038/s44172-025-00470-x)

[50] Deep-Learning-Based Acoustic Metamaterial Design for Attenuating Structure-Borne Noise. *Materials* (2023). [https://doi.org/10.3390/ma16051879](https://doi.org/10.3390/ma16051879)

[51] Deep-learning-based inverse design of phononic crystals for anticipated wave attenuation. *Journal of Applied Physics* (2022). [https://doi.org/10.1063/5.0111182](https://doi.org/10.1063/5.0111182)

[52] Deep learning of dispersion engineering in two-dimensional phononic crystals. *Engineering Optimization* (2021). [https://doi.org/10.1080/0305215x.2021.1988587](https://doi.org/10.1080/0305215x.2021.1988587)

[53] Unleashing the potential: AI empowered advanced metasurface research. *Nanophotonics* (2024). [https://doi.org/10.1515/nanoph-2023-0759](https://doi.org/10.1515/nanoph-2023-0759)

[54] AI-driven approaches in electromagnetic metamaterials design and application: a review. *EPJ Applied Metamaterials* (2025). [https://doi.org/10.1051/epjam/2025006](https://doi.org/10.1051/epjam/2025006)

[55] Advanced deep learning approaches in metasurface modeling and design: A review. *Progress in Quantum Electronics* (2025). [https://doi.org/10.1016/j.pquantelec.2025.100554](https://doi.org/10.1016/j.pquantelec.2025.100554)

[56] Empowering Metasurfaces with Inverse Design: Principles and Applications. *ACS Photonics* (2022). [https://doi.org/10.1021/acsphotonics.1c01850](https://doi.org/10.1021/acsphotonics.1c01850)

[57] Deep learning in metasurfaces: from automated design to adaptive metadevices. *Advanced Photonics* (2025). [https://doi.org/10.1117/1.AP.7.3.034005](https://doi.org/10.1117/1.AP.7.3.034005)

[58] Advances in Deep Learning-Driven Metasurface Design and Application in Holographic Imaging. *Photonics* (2025). [https://doi.org/10.3390/photonics12100947](https://doi.org/10.3390/photonics12100947)

[59] Deep-learning empowered customized chiral metasurface for calibration-free biosensing. *Advanced Materials* (2025). [https://doi.org/10.1002/adma.202411490](https://doi.org/10.1002/adma.202411490)

[60] Deep-learning-enabled intelligent design of thermal metamaterials. *Advanced Materials* (2023). [https://doi.org/10.1002/adma.202302387](https://doi.org/10.1002/adma.202302387)

[61] Machine Learning Aided Design and Optimization of Thermal Metamaterials. *Chemical Reviews* (2024). [https://doi.org/10.1021/acs.chemrev.3c00708](https://doi.org/10.1021/acs.chemrev.3c00708)

[62] Deep learning-assisted active metamaterials with heat-enhanced thermal transport. *Advanced Materials* (2024). [https://doi.org/10.1002/adma.202305791](https://doi.org/10.1002/adma.202305791)

[63] Machine Learning-Enabled Inverse Design of Radiative Cooling Film with On-Demand Transmissive Color. *ACS Photonics* (2023). [https://doi.org/10.1021/acsphotonics.2c01857](https://doi.org/10.1021/acsphotonics.2c01857)

[64] Integrated mechanical computing for autonomous soft machines. *Nature Communications* (2024). [https://doi.org/10.1038/s41467-024-47201-y](https://doi.org/10.1038/s41467-024-47201-y)

[65] Recent Advances in Reconfigurable Metasurfaces: Principle and Applications. *Nanomaterials* (2023). [https://doi.org/10.3390/nano13030534](https://doi.org/10.3390/nano13030534)

[66] Real-data-driven real-time reconfigurable microwave reflective surface. *Nature Communications* (2023). [https://doi.org/10.1038/s41467-023-43473-y](https://doi.org/10.1038/s41467-023-43473-y)

[67] Probabilistic representation and inverse design of metamaterials based on a deep generative model with semi-supervised learning strategy. *Advanced Materials* (2019). [https://doi.org/10.1002/adma.201901111](https://doi.org/10.1002/adma.201901111)

[68] Deep learning for topology optimization of 2D metamaterials. *Materials & Design* (2020). [https://doi.org/10.1016/j.matdes.2020.109098](https://doi.org/10.1016/j.matdes.2020.109098)

[69] Diffusion probabilistic model based accurate and high-degree-of-freedom metasurface inverse design. *Nanophotonics* (2023). [https://doi.org/10.1515/nanoph-2023-0292](https://doi.org/10.1515/nanoph-2023-0292)

[70] Graph-based metamaterials: Deep learning of structure-property relations. *Materials & Design* (2022). [https://doi.org/10.1016/j.matdes.2022.111175](https://doi.org/10.1016/j.matdes.2022.111175)

[71] Learning the nonlinear dynamics of mechanical metamaterials with graph networks. *International Journal of Mechanical Sciences* (2023). [https://doi.org/10.1016/j.ijmecsci.2022.107835](https://doi.org/10.1016/j.ijmecsci.2022.107835)

[72] Designing metamaterials with programmable nonlinear responses and geometric constraints in graph space. *Nature Machine Intelligence* (2025). [https://doi.org/10.1038/s42256-025-01067-x](https://doi.org/10.1038/s42256-025-01067-x)

[73] Similarity equivariant graph neural networks for homogenization of metamaterials. *Computer Methods in Applied Mechanics and Engineering* (2025). [https://doi.org/10.1016/j.cma.2025.117867](https://doi.org/10.1016/j.cma.2025.117867)

[74] Reconstruction and generation of porous metamaterial units via variational graph autoencoder and large language model. *Journal of Computing and Information Science in Engineering* (2025). [https://doi.org/10.1115/1.4066095](https://doi.org/10.1115/1.4066095)

[75] Implicit Neural Representations with Periodic Activation Functions (SIREN). *NeurIPS* (2020). [https://arxiv.org/abs/2006.09661](https://arxiv.org/abs/2006.09661)

[76] DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation. *CVPR* (2019). [https://arxiv.org/abs/1901.05103](https://arxiv.org/abs/1901.05103)

[77] Occupancy Networks: Learning 3D Reconstruction in Function Space. *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (2019). [https://doi.org/10.1109/CVPR.2019.00459](https://doi.org/10.1109/CVPR.2019.00459)

[78] From Data to Functa: Your Data Point is a Function and You Can Treat It Like One. *ICML* (2022). [https://doi.org/10.48550/arXiv.2201.12204](https://doi.org/10.48550/arXiv.2201.12204)

[79] Heterogeneous Metamaterials via Multiscale Neural Implicit Representation. *arXiv* (2025). [https://doi.org/10.1115/detc2025-168619](https://doi.org/10.1115/detc2025-168619)

[80] NTopo: Mesh-free Topology Optimization using Implicit Neural Representations. *NeurIPS* (2021). [https://arxiv.org/abs/2102.10782](https://arxiv.org/abs/2102.10782)

[81] Novel Surface‐Based Bézier Metamaterials: A Higher Degree of Parametrization toward Tailoring the Effective Properties under Compression. *Advanced Engineering Materials* (2025). [https://doi.org/10.1002/adem.202500638](https://doi.org/10.1002/adem.202500638)

[82] Nanophotonic device design based on large language models: multilayer and metasurface examples. *Nanophotonics* (2025). [https://doi.org/10.1515/nanoph-2024-0674](https://doi.org/10.1515/nanoph-2024-0674)

[83] A data-driven design for sound absorption of acoustic metamaterials based on large language models. *Scientific Reports* (2026). [https://doi.org/10.1038/s41598-025-29930-2](https://doi.org/10.1038/s41598-025-29930-2)

[84] Learning Electromagnetic Metamaterial Physics With ChatGPT. (2025). [https://doi.org/10.48550/arXiv.2404.15458](https://doi.org/10.48550/arXiv.2404.15458)

[85] Inverting the structure-property map of truss metamaterials by deep learning. *Proceedings of the National Academy of Sciences* (2022). [https://doi.org/10.1073/pnas.2111505119](https://doi.org/10.1073/pnas.2111505119)

[86] Inverse-designed spinodoid metamaterials. *npj Computational Materials* (2020). [https://doi.org/10.1038/s41524-020-0341-6](https://doi.org/10.1038/s41524-020-0341-6)

[87] Deep learning for the rare-event rational design of 3D printed multi-material mechanical metamaterials. *Communications Materials* (2022). [https://doi.org/10.1038/s43246-022-00270-2](https://doi.org/10.1038/s43246-022-00270-2)

[88] Nanophotonic particle simulation and inverse design using artificial neural networks. *Science Advances* (2018). [https://doi.org/10.1126/sciadv.aar4206](https://doi.org/10.1126/sciadv.aar4206)

[89] A bidirectional deep neural network for accurate silicon color design. *Advanced Materials* (2019). [https://doi.org/10.1002/adma.201905467](https://doi.org/10.1002/adma.201905467)

[90] Deep-Learning-Enabled On-Demand Design of Chiral Metamaterials. *ACS Nano* (2018). [https://doi.org/10.1021/acsnano.8b03569](https://doi.org/10.1021/acsnano.8b03569)

[91] Neural Operator-Based Surrogate Solver for Free-Form Electromagnetic Inverse Design. *ACS Photonics* (2023). [https://doi.org/10.1021/acsphotonics.3c00156](https://doi.org/10.1021/acsphotonics.3c00156)

[92] Deep neural operator enabled concurrent multitask design for multifunctional metamaterials under heterogeneous fields. *Advanced Optical Materials* (2024). [https://doi.org/10.1002/adom.202303087](https://doi.org/10.1002/adom.202303087)

[93] Characterization and inverse design of stochastic mechanical metamaterials using neural operators. *Advanced Materials* (2025). [https://doi.org/10.1002/adma.202420063](https://doi.org/10.1002/adma.202420063)

[94] One-shot learning for solution operators of partial differential equations. [https://doi.org/10.1038/s41467-025-63076-z](https://doi.org/10.1038/s41467-025-63076-z)

[95] Training deep neural networks for the inverse design of nanophotonic structures. *ACS Photonics* (2018). [https://doi.org/10.1021/acsphotonics.7b01377](https://doi.org/10.1021/acsphotonics.7b01377)

[96] Machine learning inverse problem for topological photonics. *Communications Physics* (2018). [https://doi.org/10.1038/s42005-018-0058-8](https://doi.org/10.1038/s42005-018-0058-8)

[97] Free-Form Diffractive Metagrating Design Based on Generative Adversarial Networks. *ACS Nano* (2019). [https://doi.org/10.1021/acsnano.9b02371](https://doi.org/10.1021/acsnano.9b02371)

[98] Generative model for the inverse design of metasurfaces. *Nano Letters* (2018). [https://doi.org/10.1021/acs.nanolett.8b03171](https://doi.org/10.1021/acs.nanolett.8b03171)

[99] Simulator-based training of generative neural networks for the inverse design of metasurfaces. *Nanophotonics* (2020). [https://doi.org/10.1515/nanoph-2019-0330](https://doi.org/10.1515/nanoph-2019-0330)

[100] Inverse design of structural color: finding multiple solutions via conditional GANs. *Nanophotonics* (2022). [https://doi.org/10.1515/nanoph-2022-0095](https://doi.org/10.1515/nanoph-2022-0095)

[101] Controllable inverse design of auxetic metamaterials using deep learning. *Materials & Design* (2021). [https://doi.org/10.1016/j.matdes.2021.110178](https://doi.org/10.1016/j.matdes.2021.110178)

[102] Generative adversarial networks for high degree of freedom metasurface designs. *Advanced Composites and Hybrid Materials* (2024). [https://doi.org/10.1007/s42114-024-01190-0](https://doi.org/10.1007/s42114-024-01190-0)

[103] Accelerating auxetic metamaterial design with deep learning. *Advanced Engineering Materials* (2020). [https://doi.org/10.1002/adem.202070018](https://doi.org/10.1002/adem.202070018)

[104] Reinforcement learning applied to metamaterial design. *The Journal of the Acoustical Society of America* (2021). [https://doi.org/10.1121/10.0005545](https://doi.org/10.1121/10.0005545)

[105] Reinforcement learning optimisation for graded metamaterial design using a physical-based constraint on the state representation and action space. *Scientific Reports* (2023). [https://doi.org/10.1038/s41598-023-48927-3](https://doi.org/10.1038/s41598-023-48927-3)

[106] Deep reinforcement learning empowers automated inverse design and optimization of photonic crystals for nanoscale laser cavities. *Nanophotonics* (2023). [https://doi.org/10.1515/nanoph-2022-0692](https://doi.org/10.1515/nanoph-2022-0692)

[107] MatterGen: A generative model for inorganic materials design. *Nature* (2025). [https://doi.org/10.1038/s41586-025-08628-5](https://doi.org/10.1038/s41586-025-08628-5)

[108] Generative Inverse Design of Metamaterials with Functional Responses by Interpretable Learning. *arXiv* (2024). [https://doi.org/10.48550/arXiv.2401.00003](https://doi.org/10.48550/arXiv.2401.00003)

[109] Manifold Learning for Knowledge Discovery and Intelligent Inverse Design of Photonic Nanostructures. *ACS Photonics* (2022). [https://doi.org/10.1021/acsphotonics.1c01888](https://doi.org/10.1021/acsphotonics.1c01888)

[110] Investigating static and dynamic behaviors in 3D chiral mechanical metamaterials by disentangled generative models. *Advanced Functional Materials* (2025). [https://doi.org/10.1002/adfm.202412901](https://doi.org/10.1002/adfm.202412901)

[111] Reconstruction and Generation of Porous Metamaterial Units Via Variational Graph Autoencoder and Large Language Model. *Journal of Computing and Information Science in Engineering* (2025). [https://doi.org/10.1115/1.4066095](https://doi.org/10.1115/1.4066095)

[112] Designing connectivity-guaranteed porous metamaterial units using generative graph neural networks. *Journal of Mechanical Design* (2024). [https://doi.org/10.1115/1.4066128](https://doi.org/10.1115/1.4066128)

[113] UniMate: A Unified Model for Mechanical Metamaterial Generation, Property Prediction, and Condition Confirmation. *ICML* (2025). [https://doi.org/10.48550/arXiv.2506.15722](https://doi.org/10.48550/arXiv.2506.15722)

[114] Transformers are Graph Neural Networks. (2025). [https://doi.org/10.48550/arXiv.2506.22084](https://doi.org/10.48550/arXiv.2506.22084)

[115] Innovations in metamaterial and metasurface antenna design: The role of deep learning. *Materials Today Electronics* (2025). [https://doi.org/10.1016/j.mtelec.2025.100162](https://doi.org/10.1016/j.mtelec.2025.100162)

[116] Crystal structure generation with autoregressive large language modeling. *Nature Communications* (2024). [https://doi.org/10.1038/s41467-024-54639-7](https://doi.org/10.1038/s41467-024-54639-7)

[117] AtomGPT: Atomistic generative pretrained transformer for forward and inverse materials design. *The Journal of Physical Chemistry Letters* (2024). [https://doi.org/10.1021/acs.jpclett.4c01126](https://doi.org/10.1021/acs.jpclett.4c01126)

[118] Generative lattice units with 3D diffusion for inverse design: GLU3D. *Advanced Functional Materials* (2024). [https://doi.org/10.1002/adfm.202404165](https://doi.org/10.1002/adfm.202404165)

[119] Denoising diffusion algorithm for inverse design of microstructures with fine-tuned nonlinear material properties. *Computer Methods in Applied Mechanics and Engineering* (2023). [https://doi.org/10.1016/j.cma.2023.116126](https://doi.org/10.1016/j.cma.2023.116126)

[120] Optimizing metamaterial inverse design with 3D conditional diffusion model and data augmentation. *Advanced Materials Technologies* (2025). [https://doi.org/10.1002/admt.202500293](https://doi.org/10.1002/admt.202500293)

[121] Scaling atom-by-atom inverse design with nano-topology optimization and diffusion models. (2026). [https://doi.org/10.48550/arXiv.2604.03276](https://doi.org/10.48550/arXiv.2604.03276)

[122] Inverse design of nonlinear mechanical metamaterials via video denoising diffusion models. *Nature Machine Intelligence* (2023). [https://doi.org/10.1038/s42256-023-00762-x](https://doi.org/10.1038/s42256-023-00762-x)

[123] High‐Asymmetry Metasurface: A New Solution for Terahertz Resonance via Active Learning‐Augmented Diffusion Model. *Advanced Science* (2025). [https://doi.org/10.1002/advs.202508610](https://doi.org/10.1002/advs.202508610)

[124] Bayesian active learning for accelerated design of broadband polarization-insensitive metasurfaces. *Intelligent Computing* (2025). [https://doi.org/10.34133/icomputing.0135](https://doi.org/10.34133/icomputing.0135)

[125] A Hybrid Conditional Diffusion-DeepONet Framework for High-Fidelity Stress Prediction in Hyperelastic Materials. (2026). [https://doi.org/10.48550/ARXIV.2603.18225](https://doi.org/10.48550/ARXIV.2603.18225)

[126] Self-supervised AI for decoding and designing disordered metamaterials. *Science AdvAnceS* (2026). [https://doi.org/10.1126/sciadv.adx7389](https://doi.org/10.1126/sciadv.adx7389)

[127] Flow Matching for Accelerated Simulation of Atomic Transport in Materials. (2025). [https://doi.org/10.48550/arXiv.2410.01464](https://doi.org/10.48550/arXiv.2410.01464)

[128] Learning Electromagnetic Metamaterial Physics With ChatGPT. (2025). [https://doi.org/10.48550/arXiv.2404.15458](https://doi.org/10.48550/arXiv.2404.15458)

[129] Fine-Tuned Language Models Generate Stable Inorganic Materials as Text. (2024). [https://doi.org/10.48550/ARXIV.2402.04379](https://doi.org/10.48550/ARXIV.2402.04379)

[130] MechGPT, a Language-Based Strategy for Mechanics and Materials Modeling. *Applied Mechanics Reviews* (2024). [https://doi.org/10.1115/1.4063843](https://doi.org/10.1115/1.4063843)

[131] MetaScientist: A Human-AI Synergistic Framework for Automated Mechanical Metamaterial Design. (2025). [https://doi.org/10.18653/v1/2025.naacl-demo.34](https://doi.org/10.18653/v1/2025.naacl-demo.34)

[132] Agentic deep graph reasoning yields self-organizing knowledge networks. *Journal of Materials Research* (2025). [https://doi.org/10.1557/s43578-025-01652-1](https://doi.org/10.1557/s43578-025-01652-1)

[133] Large Language Models as Optimization Controllers: Adaptive Continuation for SIMP Topology Optimization. (2026). [https://doi.org/10.48550/ARXIV.2603.25099](https://doi.org/10.48550/ARXIV.2603.25099)

[134] Materials science in the era of large language models: a perspective. *Digital Discovery* (2024). [https://doi.org/10.1039/d4dd00074a](https://doi.org/10.1039/d4dd00074a)

[135] 14 examples of how LLMs can transform materials science and chemistry: a reflection on a large language model hackathon. *Digital Discovery* (2023). [https://doi.org/10.1039/D3DD00113J](https://doi.org/10.1039/D3DD00113J)

[136] Leveraging large language models for predictive chemistry. *Nature Machine Intelligence* (2024). [https://doi.org/10.1038/s42256-023-00788-1](https://doi.org/10.1038/s42256-023-00788-1)

[137] Physics-informed neural networks for inverse problems in nano-optics and metamaterials. *Optics Express* (2020). [https://doi.org/10.1364/OE.384875](https://doi.org/10.1364/OE.384875)

[138] Physics-Informed Machine Learning for Inverse Design of Optical Metamaterials. *Advanced Photonics Research* (2023). [https://doi.org/10.1002/adpr.202300158](https://doi.org/10.1002/adpr.202300158)

[139] Physics-informed learning in artificial electromagnetic materials. *Applied Physics Reviews* (2025). [https://doi.org/10.1063/5.0232675](https://doi.org/10.1063/5.0232675)

[140] Physics-informed machine learning. *Nature Reviews Physics* (2021). [https://doi.org/10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)

[141] Learning the physics of all-dielectric metamaterials with deep Lorentz neural networks. *Advanced Optical Materials* (2022). [https://doi.org/10.1002/adom.202200097](https://doi.org/10.1002/adom.202200097)

[142] Digitalizing metallic materials from image segmentation to multiscale solutions via physics informed operator learning. *npj Computational Materials* (2025). [https://doi.org/10.1038/s41524-025-01718-y](https://doi.org/10.1038/s41524-025-01718-y)

[143] Deep neural networks for parameterized homogenization in concurrent multiscale structural optimization. *Structural and Multidisciplinary Optimization* (2023). [https://doi.org/10.1007/s00158-022-03471-y](https://doi.org/10.1007/s00158-022-03471-y)

[144] Reliable, efficient, and scalable photonic inverse design empowered by physics‐inspired deep learning. *Nanophotonics* (2025). [https://doi.org/10.1515/nanoph-2024-0504](https://doi.org/10.1515/nanoph-2024-0504)

[145] HyperNetworks. *ICLR* (2017). [https://doi.org/10.48550/arXiv.1609.09106](https://doi.org/10.48550/arXiv.1609.09106)

[146] A Brief Review of Hypernetworks in Deep Learning. *Artificial Intelligence Review* (2024). [https://doi.org/10.1007/s10462-024-10862-8](https://doi.org/10.1007/s10462-024-10862-8)

[147] Continual Learning with Hypernetworks. *ICLR* (2020). [https://doi.org/10.48550/arXiv.1906.00695](https://doi.org/10.48550/arXiv.1906.00695)

[148] HyperCAN: Hypernetwork-driven deep parameterized constitutive models for metamaterials. *Extreme Mechanics Letters* (2024). [https://doi.org/10.1016/j.eml.2024.102243](https://doi.org/10.1016/j.eml.2024.102243)

[149] HyperNet Fields: Efficiently Training Hypernetworks without Ground Truth by Learning Weight Trajectories. *arXiv* (2024). [https://doi.org/10.1109/cvpr52734.2025.02061](https://doi.org/10.1109/cvpr52734.2025.02061)

[150] Zhyper: Factorized Hypernetworks for Efficient Generation of Neural Fields. *arXiv* (2025). [https://doi.org/10.48550/arXiv.2503.08231](https://doi.org/10.48550/arXiv.2503.08231)

[151] HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models. *CVPR* (2024). [https://doi.org/10.1109/cvpr52733.2024.00624](https://doi.org/10.1109/cvpr52733.2024.00624)

[152] Hyper-Transforming Latent Diffusion Models for Generating Implicit Neural Representations. *ICML* (2025). [https://arxiv.org/abs/2504.16580](https://arxiv.org/abs/2504.16580)

[153] D2NWG: Diffusion-Based Neural Network Weights Generation. *ICLR* (2025). [https://doi.org/10.48550/arXiv.2402.18153](https://doi.org/10.48550/arXiv.2402.18153)

[154] Foundation Models Secretly Understand Neural Network Weights: Enhancing Hypernetwork Architectures with Foundation Models. *ICLR* (2025). [https://doi.org/10.48550/arXiv.2503.00838](https://doi.org/10.48550/arXiv.2503.00838)

[155] HyperAlign: Hypernetwork for Efficient Test-Time Alignment of Diffusion Models. (2026). [https://doi.org/10.48550/ARXIV.2601.15968](https://doi.org/10.48550/ARXIV.2601.15968)

[156] High-dimensional Bayesian optimization for metamaterial design. *Materials Genome Engineering Advances* (2024). [https://doi.org/10.1002/mgea.79](https://doi.org/10.1002/mgea.79)

[157] Accelerated Design of Architected Materials with Multifidelity Bayesian Optimization. *Journal of Engineering Mechanics* (2023). [https://doi.org/10.1061/jenmdt.emeng-7033](https://doi.org/10.1061/jenmdt.emeng-7033)

[158] On-Demand Inverse Design of Metamaterials Using Deep Neural Networks with Bayesian Optimization. *Intelligent Computing* (2025). [https://doi.org/10.34133/icomputing.0139](https://doi.org/10.34133/icomputing.0139)

[159] Designing a TPMS metamaterial via deep learning and topology optimization. *Frontiers in Mechanical Engineering* (2024). [https://doi.org/10.3389/fmech.2024.1417606](https://doi.org/10.3389/fmech.2024.1417606)

[160] Enhancing high-degree-of-freedom meta-atom design precision and speed with a tandem generative network. *ACS Photonics* (2025). [https://doi.org/10.1021/acsphotonics.4c02352](https://doi.org/10.1021/acsphotonics.4c02352)

[161] Machine learning-evolutionary algorithm enabled design for 4D-printed active composite structures. *Advanced Functional Materials* (2022). [https://doi.org/10.1002/adfm.202109805](https://doi.org/10.1002/adfm.202109805)

[162] 3D and 4D Printing of Electromagnetic Metamaterials. *Engineering* (2025). [https://doi.org/10.1016/j.eng.2024.10.017](https://doi.org/10.1016/j.eng.2024.10.017)

[163] Deep Learning-Assisted Design of Mechanical Metamaterials. *Advanced Intelligent Discovery* (2025). [https://doi.org/10.1002/aidi.202500084](https://doi.org/10.1002/aidi.202500084)

[164] Designing architected materials for mechanical compression via simulation, deep learning, and experimentation. *npj Computational Materials* (2023). [https://doi.org/10.1038/s41524-023-01036-1](https://doi.org/10.1038/s41524-023-01036-1)

[165] Kirigami-inspired parachutes with programmable reconfiguration. *Nature* (2025). [https://doi.org/10.1038/s41586-025-09515-9](https://doi.org/10.1038/s41586-025-09515-9)

[166] Bridging hard and soft: Mechanical metamaterials enable rigid torque transmission in soft robots. *Science Robotics* (2025). [https://doi.org/10.1126/scirobotics.ads0548](https://doi.org/10.1126/scirobotics.ads0548)

[167] Double-network-inspired mechanical metamaterials. *Nature Materials* (2025). [https://doi.org/10.1038/s41563-025-02219-5](https://doi.org/10.1038/s41563-025-02219-5)

[168] Programmable Entanglement of Granular Mechanical Metamaterials. *Advanced Functional Materials* (2025). [https://doi.org/10.1002/adfm.202516484](https://doi.org/10.1002/adfm.202516484)

[169] Computational microstructure design for mechanical property optimization: a review. *Science and Technology of Advanced Materials: Methods* (2025). [https://doi.org/10.1080/27660400.2025.2581359](https://doi.org/10.1080/27660400.2025.2581359)

[170] Machine-learning based design of digital materials for elastic wave control. *Extreme Mechanics Letters* (2021). [https://doi.org/10.1016/j.eml.2021.101372](https://doi.org/10.1016/j.eml.2021.101372)

[171] Dispersion relation prediction and structure inverse design of elastic metamaterials via deep learning. *Materials Today Physics* (2022). [https://doi.org/10.1016/j.mtphys.2022.100616](https://doi.org/10.1016/j.mtphys.2022.100616)

[172] AI-Assisted Metasurface Antennas Design/Optimization and Performance Enhancement Techniques: A Comprehensive Survey. *IEEE Access* (2026). [https://doi.org/10.1109/ACCESS.2026.3667812](https://doi.org/10.1109/ACCESS.2026.3667812)

[173] Advances in artificial intelligence for artificial metamaterials. *APL Materials* (2024). [https://doi.org/10.1063/5.0247369](https://doi.org/10.1063/5.0247369)

[174] Deep learning-based inverse design of multi-functional metasurface absorbers. *Optics Letters* (2024). [https://doi.org/10.1364/ol.518786](https://doi.org/10.1364/ol.518786)

[175] Neural-Network-Enabled Design of a Chiral Plasmonic Nanodimer for Target-Specific Chirality Sensing. *ACS Nano* (2023). [https://doi.org/10.1021/acsnano.2c08867](https://doi.org/10.1021/acsnano.2c08867)

[176] Frequency transfer and inverse design for metasurface under multi-physics coupling by Euler latent dynamic and data-analytical regularizations. *Nature Communications* (2025). [https://doi.org/10.1038/s41467-025-57516-z](https://doi.org/10.1038/s41467-025-57516-z)

[177] Inverse Co‐Design of Mechanical And Sensory Properties in Soft Lattice Foams for Multifunctional Wearables. *Advanced Science* (2025). [https://doi.org/10.1002/advs.202507102](https://doi.org/10.1002/advs.202507102)

[178] Shape-conformal porous frameworks for full coverage of neural organoids and high-resolution electrophysiology. *Nature Biomedical Engineering* (2026). [https://doi.org/10.1038/s41551-026-01620-y](https://doi.org/10.1038/s41551-026-01620-y)

[179] Multi‐Mechanical Regulation of 3D Printed Triply Periodic Hyperbolic Surfaces via Fourier Synthesis‐Based Free Modeling. *Advanced Science* (2025). [https://doi.org/10.1002/advs.202503694](https://doi.org/10.1002/advs.202503694)

[180] METASET: Exploring Shape and Property Spaces for Data-Driven Metamaterials Design. *Journal of Mechanical Design* (2021). [https://doi.org/10.1115/1.4048629](https://doi.org/10.1115/1.4048629)

[181] Topology Optimization for Architected Materials Design. *Annual Review of Materials Research, Vol 46* (2016). [https://doi.org/10.1146/annurev-matsci-070115-031826](https://doi.org/10.1146/annurev-matsci-070115-031826)

[182] Plate-nanolattices at the theoretical limit of stiffness and strength. *Nature Communications* (2020).

[183] An Atlas of Extreme Properties in Cubic Symmetric Metamaterials. (2026). [https://doi.org/10.48550/arXiv.2603.10934](https://doi.org/10.48550/arXiv.2603.10934)

[184] CORE: Full-Path Evaluation of LLM Agents Beyond Final State. (2025). [https://doi.org/10.48550/arXiv.2509.20998](https://doi.org/10.48550/arXiv.2509.20998)

[185] 9CDXAHU9 — (not found in local Zotero cache)

[186] Human-AI Schema Discovery and Application for Creative Problem Solving. (2025). [https://doi.org/10.48550/arXiv.2508.05045](https://doi.org/10.48550/arXiv.2508.05045)

[187] Applied Artificial Intelligence in Materials Science and Material Design. *Advanced Intelligent Systems* (2025). [https://doi.org/10.1002/aisy.202400986](https://doi.org/10.1002/aisy.202400986)

[188] Artificial Intelligence in the Design of Innovative Metamaterials: A Comprehensive Review. *International Journal of Precision Engineering and Manufacturing* (2024). [https://doi.org/10.1007/s12541-023-00857-w](https://doi.org/10.1007/s12541-023-00857-w)

[189] Metamaterials and smart structures: Leveraging AI for design, optimization and adaptive engineering solutions. *Global Journal of Engineering and Technology Advances* (2025). [https://doi.org/10.30574/gjeta.2025.24.3.0260](https://doi.org/10.30574/gjeta.2025.24.3.0260)

[190] Structural topology optimization based on diffusion generative adversarial networks. *Engineering Applications of Artificial Intelligence* (2024). [https://doi.org/10.1016/j.engappai.2024.109444](https://doi.org/10.1016/j.engappai.2024.109444)

[191] Symbiotic evolution of photonics and artificial intelligence: a comprehensive review. *Advanced Photonics* (2025). [https://doi.org/10.1117/1.ap.7.2.024001](https://doi.org/10.1117/1.ap.7.2.024001)

[192] Recent advances in metasurface design and quantum optics applications with machine learning, physics-informed neural networks, and topology optimization methods. *Light: Science & Applications* (2023). [https://doi.org/10.1038/s41377-023-01218-y](https://doi.org/10.1038/s41377-023-01218-y)

[193] Review of generative models for the inverse design of nanophotonic metasurfaces. *Applied Science and Convergence Technology* (2023). [https://doi.org/10.5757/ASCT.2023.32.6.141](https://doi.org/10.5757/ASCT.2023.32.6.141)

[194] Machine Learning‐Based Inverse Design for Functional Materials: Methods, Challenges, and Engineering Applications. *Advanced Functional Materials* (2026). [https://doi.org/10.1002/adfm.75070](https://doi.org/10.1002/adfm.75070)

[195] Machine Learning‐Based Inverse Design for Functional Materials: Methods, Challenges, and Engineering Applications. *Advanced Functional Materials* (2026). [https://doi.org/10.1002/adfm.75070](https://doi.org/10.1002/adfm.75070)

[196] Topology Optimization via Machine Learning and Deep Learning: A Review. *Journal of Computational Design and Engineering* (2023). [https://doi.org/10.1093/jcde/qwad072](https://doi.org/10.1093/jcde/qwad072)

