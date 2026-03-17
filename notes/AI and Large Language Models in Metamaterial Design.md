---
tags:
  - claude
  - llm
  - LiteratureResearch
  - Metamaterial
alias:
  - "AI and Large Language Models in Metamaterial Design: A Comprehensive Review (2020–2025)"
---


---

## Table of Contents

- [[#1. Abstract]]
- [[#2. Introduction]]
- [[#3. Metamaterial Taxonomy]]
- [[#4. Machine Learning and Deep Learning Methodologies]]
    - [[#4.1 Supervised Forward Prediction]]
    - [[#4.2 Inverse Design — Tandem and Bidirectional Networks]]
    - [[#4.3 Generative Models]]
    - [[#4.4 Physics-Informed Neural Networks (PINNs)]]
    - [[#4.5 Reinforcement Learning]]
    - [[#4.6 Active Learning and Bayesian Optimization]]
    - [[#4.7 Graph Neural Networks]]
- [[#5. Large Language Models in Metamaterial Design]]
    - [[#5.1 Fine-Tuned LLMs for Spectral Prediction]]
    - [[#5.2 LLMs for Porous Metamaterial Generation]]
    - [[#5.3 Multi-Agent LLM Frameworks]]
    - [[#5.4 Language-Guided Design LINGUAMATE]]
    - [[#5.5 LLMs as Physics Educators]]
    - [[#5.6 Foundation Models and Protein-Inspired Analogs]]
- [[#6. Electromagnetic and Microwave Metamaterials]]
- [[#7. Photonic Metamaterials and Nanophotonics]]
- [[#8. Acoustic and Elastic Metamaterials]]
- [[#9. Mechanical Metamaterials]]
- [[#10. Thermal Metamaterials]]
- [[#11. Fabrication Integration and Multiscale Modeling]]
- [[#12. Challenges and Future Outlook]]
- [[#13. Conclusion]]
- [[#14. References]]

---

## 1. Abstract

This comprehensive review synthesizes over 210 publications spanning 2020–2025 on the application of machine learning (ML), deep learning (DL), and large language models (LLMs) to metamaterial design across acoustic, electromagnetic, mechanical, photonic, and thermal domains. We categorize methodological approaches — forward prediction, inverse design, generative models, reinforcement learning, physics-informed neural networks, and LLM-guided design — and survey their adoption across metamaterial subfields. We identify key trends, benchmark achievements, and persistent challenges including data scarcity, interpretability, multiphysics coupling, and fabrication constraints. This review serves both newcomers and specialists seeking an integrated perspective on the data-driven metamaterial design landscape.

**Keywords:** metamaterials, machine learning, deep learning, large language models, inverse design, generative models, phononic crystals, metasurfaces, physics-informed neural networks, topology optimization

---

## 2. Introduction

Metamaterials are engineered structures whose properties arise from artificially designed micro- and nanoarchitectures rather than from chemical composition alone [1, 168, 169, 170]. Since Veselago's theoretical prediction of negative-index materials and Pendry's implementation of the double-negative medium, metamaterial research has expanded dramatically to encompass electromagnetic cloaking, superlensing, acoustic wave steering, mechanical energy absorption, and thermal management [168, 171, 172]. The defining advantage of metamaterials over natural materials is the ability to achieve on-demand properties — negative Poisson's ratio, zero-index refraction, unidirectional wave propagation — by tailoring geometry at subwavelength scales.

Despite this promise, conventional design pipelines rely on iterative trial-and-error combined with computationally intensive finite-element (FEM) or finite-difference time-domain (FDTD) simulations. Exploring the vast parameter space of a metamaterial unit cell demands millions of evaluations, making brute-force search intractable. Physics-based optimization methods (genetic algorithms, gradient descent, topology optimization) offer structured exploration but are often sensitive to initial conditions and struggle with non-convex multimodal landscapes [1, 5].

Over the past five years, machine learning has emerged as a transformative toolkit for metamaterial design [1–10]. Neural networks, once proven in image recognition and natural language processing, now serve as ultra-fast surrogates for physics simulators, reducing design cycles from hours to milliseconds. Generative models enable the synthesis of entirely novel microstructures beyond the reach of parameter sweeps. Reinforcement learning autonomously optimizes metamaterial configurations. Most recently, large language models (LLMs) fine-tuned on metamaterial datasets have demonstrated remarkable capability for forward prediction, inverse reasoning, and language-guided design workflows [11–15].

This review is organized as follows. Section 3 introduces the metamaterial taxonomy. Section 4 surveys ML/DL methodologies. Section 5 reviews LLM applications specifically. Sections 6–10 cover domain-specific advances in electromagnetic, photonic, acoustic, mechanical, and thermal metamaterials. Section 11 addresses fabrication and multiscale considerations. Section 12 discusses open challenges and future outlook. Section 13 concludes.

---

## 3. Metamaterial Taxonomy

Metamaterials can be classified by the physical wave they manipulate and by their structural geometry. The table below provides a high-level taxonomy relevant to ML-aided design efforts covered in this review.

|Domain|Key Properties|Representative ML-Aided Applications|
|---|---|---|
|**Electromagnetic / Microwave**|Negative permittivity/permeability, cloaking|Absorber design, beam steering, FSS, RCS reduction|
|**Photonic / Optical**|Subwavelength light manipulation, structural color, holography|Inverse metasurface design, metalens optimization, nonlinear optics|
|**Acoustic / Phononic**|Band gap engineering, wave steering, noise control|Phononic crystal design, vibration attenuation, energy harvesting|
|**Mechanical**|Auxetic behavior, programmable stiffness, negative Poisson's ratio|Topology optimization, additive manufacturing, impact resistance|
|**Thermal**|Thermal cloaking, thermal diodes, anisotropic conductivity|Metastructure surrogate modeling, Bayesian optimization|
|**Terahertz (THz)**|Bandpass filtering, biosensing, 6G communication|DNN-based absorber design, transformer-based sensor optimization|

> **Note:** Electromagnetic and photonic metamaterials dominate the literature, constituting approximately 55% of ML-aided design studies reviewed.

---

## 4. Machine Learning and Deep Learning Methodologies

We organize ML/DL approaches along a spectrum from purely data-driven supervised forward prediction to generative and autonomous optimization.

### 4.1 Supervised Forward Prediction

The simplest and most common application of ML in metamaterial research is the **forward prediction problem**: given a structural description (geometry parameters, unit cell topology), predict the physical response (absorption spectrum, band gap, scattering cross-section). Feedforward neural networks (FNNs) and multilayer perceptrons (MLPs) were among the first architectures deployed [95]. Convolutional neural networks (CNNs) excel at extracting spatial features from 2D/3D unit cell images [64]. Training datasets are generated via simulation (FEM, FDTD) and the neural network serves as a fast surrogate for repeated evaluations [83, 84].

Surrogate models trained on metamaterial datasets achieve prediction accuracies within 1–5% of full-wave simulations while reducing computational cost by **3–5 orders of magnitude** [83]. For example, a DNN surrogate replacing FEM in an acoustic metamaterial design loop reduced absorption coefficient prediction time by a factor of 4,500 [58]. Graph neural networks (GNNs) handle irregular geometries and truss-based metamaterials through topology-aware message passing [41, 42, 57].

### 4.2 Inverse Design — Tandem and Bidirectional Networks

**Inverse design** — predicting the geometry that achieves a desired response — is the central challenge in metamaterial ML. The problem is fundamentally ill-posed: multiple geometries can produce the same spectrum. Tandem neural networks (TNNs) address this by coupling a forward predictor with an inverse generator [181]. During training, the forward network is frozen, and only the inverse network is updated by minimizing prediction error through the tandem chain [16, 89]. Bidirectional networks simultaneously learn forward and inverse mappings, improving consistency [94]. Neural-adjoint methods embed a pre-trained forward model as a differentiable simulator within gradient-based inverse optimization [35].

### 4.3 Generative Models

Generative adversarial networks (GANs) and variational autoencoders (VAEs) have become foundational tools for metamaterial design space exploration [31–34]:

- **GANs** consist of a generator synthesizing candidate designs and a discriminator classifying real vs. fake designs; adversarial training drives the generator toward physically plausible, high-performance configurations. Liu et al. (2018) used GANs to generate over 20,000 novel metasurface designs [182], while a conditional GAN (cGAN) improved transmission loss of acoustic metamaterials by up to **20 dB** [23, 49].
- **VAEs** compress metamaterial designs into a continuous latent space, enabling smooth interpolation and property-guided navigation [24, 28].
- **Diffusion models**, introduced more recently, progressively denoise random samples to generate high-quality designs and demonstrate superior sample diversity and stability compared to GANs [34, 62]. _MetaDiffusion_ surpasses GAN-based approaches in accuracy and design diversity [34].

### 4.4 Physics-Informed Neural Networks (PINNs)

Physics-informed neural networks (PINNs) incorporate governing PDEs (Maxwell's equations, Helmholtz equation, elasticity equations) directly into the loss function [66, 67]. By penalizing physics residuals, PINNs learn solutions consistent with physical laws even in data-sparse regions:

- Chen et al. (2020) demonstrated PINNs for inverse problems in nano-optics [22, 67]
- Sarkar et al. (2023) employed a physics-informed tandem architecture for optical metasurface design [17]
- PINNs are particularly valuable when training data is limited [66, 68]

### 4.5 Reinforcement Learning

Reinforcement learning (RL) formulates metamaterial optimization as a **Markov decision process**: an agent sequentially modifies structural parameters (actions) and receives rewards based on response matching a target. RL does not require pre-generated training datasets, learning instead through environment interaction [51, 52]:

- Double deep Q-learning and DDPG applied to acoustic metamaterial scattering minimization [51]
- Phononic crystal band-gap maximization [59]
- Graded metamaterial energy harvesting under realistic excitations [52]

The key challenge for RL is computational cost during training, which scales steeply with design space dimensionality.

### 4.6 Active Learning and Bayesian Optimization

Active learning addresses the data bottleneck by intelligently selecting which simulations to run next. Bayesian optimization (BO) maintains a probabilistic surrogate of the objective function and uses an acquisition function (upper confidence bound, expected improvement) to balance exploration and exploitation [73, 74]:

- Liu et al. (2025) applied Bayesian active learning to accelerate broadband polarization-insensitive metasurface design [73]
- Cao et al. (2024) demonstrated BO for on-demand inverse design with phononic bandgap characteristics [74]

### 4.7 Graph Neural Networks

Graph neural networks represent metamaterial architectures as graphs where nodes are structural units and edges encode interactions [41, 42, 57]:

- Meyer et al. (2022) demonstrated GNN-based structure-property mapping for graph-based metamaterials [41]
- Xue et al. learned nonlinear dynamics of soft mechanical metamaterials with a metamaterial graph network (MGN) [42, 60]
- Equivariant GNNs incorporating physical symmetries (translation, rotation, scale) improve generalization and data efficiency [71]

---

## 5. Large Language Models in Metamaterial Design

The integration of LLMs into metamaterial research represents the newest frontier, enabled by the convergence of powerful pre-trained models and domain-specific fine-tuning strategies.

### 5.1 Fine-Tuned LLMs for Spectral Prediction

Hayes et al. (2024) conducted the **first systematic empirical study** of LLMs (GPT-based) for metamaterial regression tasks [11]. Fine-tuned LLMs (FT-LLMs) achieved lower mean squared error than feedforward neural networks, random forests, k-nearest neighbors, and linear regression on acoustic double metamaterial (ADM) datasets **across all examined dataset sizes**. The LLM's ability to perform zero-shot learning — making accurate predictions on unseen configurations — distinguished it from classical ML approaches [11].

However, the authors found that inverse design via direct LLM querying produced unreliable results for large datasets, highlighting a gap between forward prediction capability and structured inverse reasoning.

### 5.2 LLMs for Porous Metamaterial Generation

Fang et al. (2025) fine-tuned the byT5 LLM on graph representations of porous metamaterial units [14]. The LLM learned to predict node connectivity in graphs representing solid and pore phases, enabling both reconstruction of known structures and generation of novel designs. A key finding was the LLM's **zero-shot transferability**: trained on solid-phase graphs, the model accurately predicted pore-phase graphs following the same generative logic [8, 14].

### 5.3 Multi-Agent LLM Frameworks

**CrossMatAgent** (Tian et al., 2025) developed a multi-agent generative framework combining LLM reasoning with diffusion model image generation for manufacturable metamaterial pattern design [12]. The framework employs:

1. **Agent Designer** — LLM-based for language-guided structure proposal
2. **Agent Generator** — Stable Diffusion XL (SDXL) for image synthesis
3. **Agent Supervisor** — fast property feedback without full simulation

LLM reasoning and reflection mechanisms generate paired text-image data, which fine-tune SDXL for few-shot learning with LoRA adaptors [12].

### 5.4 Language-Guided Design LINGUAMATE

The **LINGUAMATE** framework addresses the exploratory phase of metamaterial design, where designers have incomplete specifications and vague qualitative goals [13]. Natural language inputs (e.g., _"lightweight and energy-absorbing under impact"_) are processed by LLMs with strong language understanding and embedded domain knowledge. LLM-based agents propose candidate structures, query simulation tools, and iteratively refine designs. This represents a paradigm shift from numerical specification toward **concept-driven, dialogue-based design workflows** [3, 13].

### 5.5 LLMs as Physics Educators

Lu et al. (2024) investigated LLMs (including ChatGPT) as tools for **learning electromagnetic metamaterial physics** [15]. Pre-trained LLMs already embedded substantial qualitative knowledge about metamaterial behavior. Fine-tuning on domain-specific datasets improved quantitative predictions. The authors proposed LLMs as "domain bridges" enabling non-expert users to explore metamaterial design spaces via natural language queries.

### 5.6 Foundation Models and Protein-Inspired Analogs

Developments in foundation models for molecular and materials design provide transferable architectures for metamaterial contexts:

- Gruver et al. (2024) fine-tuned LLMs to generate stable inorganic materials as text sequences [112]
- Buehler (2024) demonstrated **MechGPT**, a language-based strategy for mechanics and materials modeling integrating LLM reasoning with physics constraints [113]
- These works establish a template for fine-tuning massive pre-trained models on metamaterial corpora

---

## 6. Electromagnetic and Microwave Metamaterials

Electromagnetic metamaterials — including frequency selective surfaces (FSSs), metamaterial absorbers (MMAs), and reconfigurable intelligent surfaces (RISs) — have received the greatest volume of ML-aided design attention.

### 6.1 Absorbers and Microwave Devices

Deep neural networks have dramatically accelerated MMA design:

- Xie et al. (2023) constructed a tandem DNN achieving dual-band absorption **>85% over 5.1–14 GHz** with MSE of 8.3×10⁻⁴ [16]
- SA-accelerated DNN framework by Fan et al. (2024) achieved **80,000× speedup** over full-wave solvers for THz absorbers with 10-minute design cycles [18]
- Physics-informed learning by Deng et al. (2025) extends these results to a systematic physics-aware paradigm [101, 149]

### 6.2 Metasurface Inverse Design

Metasurface inverse design presents the "one-to-many" problem acutely. Probabilistic approaches using mixture density networks [105], conditional VAEs [28, 33], normalizing flows and diffusion models [34], and manifold learning for latent design space discovery [25] address degeneracy by learning the full distribution of valid designs. Gu et al. (2023) employed generative models for FSS inverse design to navigate this degeneracy [118].

### 6.3 Transformer-Based Approaches

**Metaformer** (Gao et al., 2024) is a transformer-based explainable DL model for metasurface sensor design [26]:

- Spectrum-splitting scheme achieves **99% prediction error reduction**
- Reduces training parameters by **99%** compared to conventional transformers
- Multi-head attention reveals which spectral regions most strongly couple to structural parameters, providing physical interpretability [26, 27]

### 6.4 Reconfigurable and Dynamic Metasurfaces

- Wen et al. (2023) developed a real-time tandem neural network for adaptive surface design at microwave frequencies [89]
- GAN-based approaches generate diverse reconfigurable configurations across the full response space
- ML paired with phase-change materials (VO₂, GST) enables autonomous reconfiguration [45, 89]

---

## 7. Photonic Metamaterials and Nanophotonics

### 7.1 Nanophotonic Inverse Design

The field of nanophotonic inverse design has been transformed by DL [9, 126]:

- Peurifoy et al. (2018) pioneered ANN-based nanophotonic particle simulation and inverse design [75]
- Bidirectional networks [94], conditional VAEs [28], physics-informed architectures [17, 67], and diffusion models [34] progressively improved accuracy and design coverage
- Wiecha et al. (2021) provided a comprehensive critical review classifying DL inverse design approaches and discussing strengths/weaknesses [9]

### 7.2 Metalens Design

ML enables efficient design of phase profiles and meta-atom libraries:

- Physics-informed DL by Sarkar et al. (2023) achieved consistent design selection by incorporating Maxwell's equations as physical priors [17]
- Neural operator-based surrogate solvers by Augenstein et al. handle arbitrary-geometry inverse design for electromagnetic structures [78]

### 7.3 Structural Color and Holography

- Dai et al. (2022) used conditional GANs to find multiple solution branches for structural color design [100]
- Deep learning-enabled holography produced complex-amplitude holograms with efficiency orders of magnitude beyond iterative algorithms [115]
- ML-enabled inverse design of radiative cooling films with on-demand transmissive color [77]

### 7.4 Topological and Chiral Photonics

- Pilozzi et al. used ML to solve the topological photonics inverse problem [99]
- GAN-based data augmentation by Sun et al. (2025) predicted Chern numbers in photonic crystals [160]
- Chiral metamaterials for polarization-sensitive sensing inversely designed using DL by Gao et al. (2020) [81] and Han et al. (2023) [76]

---

## 8. Acoustic and Elastic Metamaterials

### 8.1 Phononic Crystal Design

Neural networks accelerate band structure prediction [53, 55], replacing expensive plane-wave expansion or FEM:

- CNN-based architectures extract spatial features of PnC unit cells and predict dispersion relations with near-simulation accuracy [54, 55, 56]
- Inverse design using DNNs, MLPs, and RL achieves PnCs with specific bandgap centers and widths [3, 51, 53]

### 8.2 Acoustic Absorbers and Noise Control

- On et al. (2023) presented the **first DL-based design of phononic plate metamaterials** for auditory-frequency noise attenuation, achieving 2% error with only 360 training samples [50]
- Rosafalco et al. (2023) demonstrated RL optimization for graded metamaterials maximizing energy harvesting under realistic magnetic and random vibration loads [52]
- Bio-inspired acoustic metamaterials for railway noise control explored with ML-assisted design strategies [49]

### 8.3 Acoustic Cloaking and Wave Control

- Zhang et al. (2021) constructed a digital structural genome using CNN to predict elastic wave properties for acoustic corner/carpet cloaks [53]
- ML-driven acoustic invisibility cloaks with multilayered core-shell configurations demonstrated using physics-informed inversion [4, 31]
- Shah et al. used RL agents to discover cylindrical scatterer configurations minimizing acoustic plane wave scattering [51]

---

## 9. Mechanical Metamaterials

### 9.1 Forward Prediction and Constitutive Modeling

ML surrogate models predict mechanical responses (stress-strain curves, elastic moduli, Poisson's ratio) without repeated FEM [36, 38, 40]:

- GNNs provide topology-aware mechanical property prediction [41, 42]
- Bastek et al. (2022) inverted the structure-property map of truss metamaterials using DL, mapping desired mechanical properties back to manufacturable geometries [40]

### 9.2 Inverse Design of Auxetic and Programmable Metamaterials

- Zheng et al. demonstrated controllable inverse design of auxetic metamaterials using deep learning [82]
- Ha et al. (2023) achieved **rapid inverse design covering the full mechanical property space** via machine learning [37, 163]
- Deep learning generated multi-material auxetic structures validated by 3D printing [38]

### 9.3 Topology Optimization

DL-accelerated topology optimization replaces iterative FEM-SIMP cycles:

- CNNs predict optimized density distributions from loading conditions [64]
- GANs generate high-resolution topologies from coarse inputs [62, 65]
- **ResUNet-GAN** architectures achieve multi-material topology optimization for maximum bulk/shear modulus or negative Poisson's ratio [43, 61]
- TPMS metamaterials (gyroids) optimized using 3D CNN surrogates by Abueidda et al. (2024) [45]

### 9.4 Multiscale and Manufacturing-Aware Design

- Deep generative models incorporating additive manufacturing (AM) manufacturability constraints produce **print-ready 3D metamaterial units** without post-processing [44]
- VAE-based frameworks map 3D geometries to latent spaces conditioned on both mechanical properties and manufacturability metrics [44]
- Multiscale frameworks combining ML-encoded homogenization with Bayesian optimization design programmable metamaterials with heterogeneous microstructures [69, 70]

---

## 10. Thermal Metamaterials

Thermal metamaterials control heat flow through engineered microstructures exhibiting thermal cloaking, concentration, rotation, or anomalous conductivity.

- Zhu et al. (2024) provided a comprehensive _Chemical Reviews_ survey of ML-aided thermal metamaterial design covering discriminative models, generative models, and optimization algorithms [6]
- ML models predict unprecedented thermal properties previously intractable by analytical approaches
- Bayesian optimization frameworks systematically explore thermal metamaterial parameter spaces, identifying designs with specific anisotropic conductivities [73]
- ML-enhanced radiative cooling film design by Guan et al. (2023) achieved on-demand transmissive color properties [77]
- Deep learning of thermal metamaterials via graph neural networks demonstrated by Shen et al. (2023) [120]

---

## 11. Fabrication Integration and Multiscale Modeling

ML-driven metamaterial design increasingly accounts for fabrication constraints [44, 121, 122]:

- Wilt et al. developed DL-based surrogate modeling for prediction of errors in compliant auxetic metamaterials produced by AM [194]
- Fang et al. (2025) reviewed **3D and 4D printing of EM metamaterials**, highlighting ML's role in bridging design and fabrication [121]
- Sun et al. (2022) demonstrated ML-evolutionary algorithm-enabled design for 4D-printed active composite structures [122]

Multiscale modeling frameworks combine micro-scale ML surrogates with macro-scale FEA:

- Recurrent neural operators (RNOs) enable **history-dependent multiscale modeling** of architected metamaterials, capturing elastic-plastic responses across scales [70]
- Similarity equivariant GNNs for metamaterial homogenization respect physical symmetries, improving generalization to unseen microstructures [71]
- These approaches reduce FE² concurrent multiscale simulation costs from prohibitive to practical [69]

---

## 12. Challenges and Future Outlook

### 12.1 Data Scarcity and Quality

The most persistent challenge is **data scarcity**. High-fidelity simulation data is expensive to generate; each FDTD or FEM simulation may require minutes to hours of computational time. Mitigations include:

- Active learning and Bayesian optimization for intelligent sampling [73, 74]
- Transfer learning enabling pre-trained models to fine-tune on small metamaterial-specific datasets [111]
- Physics-informed constraints reducing data requirements by encoding known physical laws [66, 67]
- Open metamaterial datasets analogous to ImageNet remain a gap in the field

### 12.2 Interpretability and Physical Consistency

Deep neural networks are often black-boxes. Solutions:

- Transformer-based explainable architectures (Metaformer) reveal which spectral features drive design decisions [26]
- Manifold learning maps high-dimensional design spaces to interpretable low-dimensional representations [25]
- Physics-informed training ensures predictions satisfy Maxwell's equations, elasticity laws, or thermodynamics [17, 66, 67]

### 12.3 Multiphysics and Multiobjective Optimization

Real-world applications increasingly require simultaneous optimization of multiple physical quantities. Emerging approaches:

- Multi-task learning frameworks [206]
- Pareto-front optimization
- Multi-objective Bayesian optimization [73]
- Coupling of LLMs with multi-physics simulators [12, 13]

### 12.4 LLM Scaling and Specialization

LLM capabilities in metamaterial design are directly correlated with domain-specific fine-tuning dataset quality and size:

- Building large curated datasets of metamaterial structures with annotated properties remains a community challenge
- Foundation models pre-trained on broad materials databases (Materials Project, AFLOW, OQMD) may transfer usefully to metamaterial contexts
- **Multi-modal LLMs** jointly processing text, geometric images, and spectral data represent the next generation of design agents [13, 162]

### 12.5 Experimental Validation Loop

The majority of reviewed ML works validate designs through simulation only. Key priorities:

- Closing the simulation-to-fabrication loop with ML-guided experimental design
- Automated characterization and closed-loop active learning [37, 121]
- Digital twins integrating ML surrogates with real-time experimental feedback [89, 113]

---

## 13. Conclusion

This review surveyed over 210 publications demonstrating that ML and LLMs have fundamentally changed the pace and scope of metamaterial design. Key conclusions:

1. **Deep neural networks** now serve as ultra-fast, highly accurate surrogates for FDTD and FEM simulations across electromagnetic, acoustic, mechanical, and thermal domains.
2. **Generative models** (GANs, VAEs, diffusion models) expand the design space beyond hand-crafted parameter spaces, enabling discovery of non-intuitive metamaterial geometries with desired properties.
3. **Physics-informed neural networks** and transformer architectures improve physical consistency and interpretability.
4. **Reinforcement learning** autonomously navigates complex optimization landscapes without pre-generated training data.
5. **LLMs** represent an emerging paradigm: fine-tuned models outperform traditional ML on regression tasks, multi-agent LLM frameworks enable language-guided design, and foundation models offer transferable representations.
6. **Fabrication constraints**, multiscale physics, and multiobjective optimization remain active research frontiers.

The convergence of AI and metamaterial science promises to accelerate the discovery of materials with unprecedented functionalities — from perfect absorbers to programmable mechanical intelligence — addressing critical needs in communications, healthcare, defense, and energy.

---

## 14. References

### Major Review Papers

1. Cerniauskas G, Sadia H, Alam P. "Machine intelligence in metamaterials design: a review." _Oxford Open Materials Science_, 4(1), itae001, 2024. https://academic.oup.com/ooms/article/4/1/itae001/7604561
    
2. Tezsezen E, Yigci D, Ahmadpour A, Tasoglu S. "AI-Based Metamaterial Design." _ACS Applied Materials & Interfaces_, 16(23), 29547–29569, 2024. https://pubs.acs.org/doi/10.1021/acsami.4c04486
    
3. Jin Y, He L, Wen Z, et al. "Intelligent on-demand design of phononic metamaterials." _Nanophotonics_, 11(3), 439–460, 2022. https://doi.org/10.1515/nanoph-2021-0639
    
4. Muhammad, Kennedy J, Lim CW. "Machine learning and deep learning in phononic crystals and metamaterials — A review." _Materials Today Communications_, 2022. https://doi.org/10.1016/j.mtcomm.2022.104606
    
5. Lee D, Chen W, Wang L, Chan Y-C, Chen W. "Data-driven design for metamaterials and multiscale systems: A review." _Advanced Materials_, 36(8), 2305254, 2024. https://doi.org/10.1002/adma.202305254
    
6. Zhu C, Bamidele EA, Shen X, et al. "Machine Learning Aided Design and Optimization of Thermal Metamaterials." _Chemical Reviews_, 124(7), 4258–4331, 2024. https://pubs.acs.org/doi/10.1021/acs.chemrev.3c00708
    
7. So S, Badloe T, Noh J, Bravo-Abad J, Rho J. "Deep learning enabled inverse design in nanophotonics." _Nanophotonics_, 9(5), 1041–1057, 2020. https://doi.org/10.1515/nanoph-2019-0474
    
8. Ma W, Cheng F, Liu Y. "Deep learning for the design of photonic structures." _Nature Photonics_, 15, 77–90, 2021. https://doi.org/10.1038/s41566-020-0685-y
    
9. Wiecha PR, Arbouet A, Girard C, Muskens OL. "Deep learning in nano-photonics: inverse design and beyond." _Photonics Research_, 9(5), B182, 2021. https://opg.optica.org/prj/abstract.cfm?uri=prj-9-5-b182
    
10. Si LM, Niu R, Dang CY, et al. "Advances in artificial intelligence for artificial metamaterials." _APL Materials_, 12, 120602, 2024. https://pubs.aip.org/aip/apm/article/12/12/120602/3328550
    

---

### LLM-Specific Papers

11. Hayes D, Lu D, Malof JM, Padilla WJ. "Can Large Language Models Learn the Physics of Metamaterials? An Empirical Study with ChatGPT." _arXiv:2404.15458_, 2024. https://arxiv.org/abs/2404.15458
    
12. Tian Y, Zhou C, et al. "CrossMatAgent: AI-Assisted Design of Manufacturable Metamaterial Patterns via Multi-Agent Generative Framework." _Advanced Intelligent Discovery_, 2025. https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/aidi.202500063
    
13. Anonymous Authors. "LINGUAMATE: Language-Guided Metamaterial Design." _OpenReview_, 2025. https://openreview.net/pdf/c6a5b842bba69f81549f23a5862880871877eb4a.pdf
    
14. Fang Z, et al. "Reconstruction and Generation of Porous Metamaterial Units via VAE and LLM." _J. Comput. Inf. Sci. Eng._, 25(2), 021003, 2025. https://asmedigitalcollection.asme.org/computingengineering/article/25/2/021003/1201904
    
15. Lu D, Fan K, Jin B, Malof J, Padilla WJ. "Learning Electromagnetic Metamaterial Physics With ChatGPT." _ACS Nano_, 2024. https://pubs.acs.org/doi/10.1021/acsnano.4c01828
    
16. Lu W, Luu RK, Buehler MJ. "Fine-Tuning Large Language Models for Domain Adaptation." _npj Computational Materials_, 11, 84, 2024. https://www.nature.com/articles/s41524-025-01564-y
    
17. Gruver N, et al. "Fine-tuned language models generate stable inorganic materials as text." _ICLR 2024_. https://arxiv.org/abs/2402.04379
    
18. Buehler MJ. "MechGPT, a Language-Based Strategy for Mechanics and Materials Modeling." _Journal of Applied Mechanics_, 2024. https://doi.org/10.1115/1.4063843
    
19. Lu D, Malof JM, Padilla WJ. "An Agentic Framework for Autonomous Metamaterial Modeling and Inverse Design." _ACS Photonics_, 2025. https://doi.org/10.1021/acsphotonics.5c01514
    
20. Qi J, et al. "MetaScientist: A Human-AI Synergistic Framework for Automated Mechanical Metamaterial Design." _arXiv:2412.16270_, 2024. https://arxiv.org/abs/2412.16270
    

---

### Deep Learning — Electromagnetic & Photonic Design

16. Xie C, Li H, Cui C, et al. "Deep learning assisted inverse design of metamaterial microwave absorber." _Applied Physics Letters_, 123(18), 181701, 2023. https://pubs.aip.org/aip/apl/article/123/18/181701/2919094
    
17. Sarkar S, Ji A, Jermain Z, et al. "Physics-Informed Machine Learning for Inverse Design of Optical Metamaterials." _Advanced Photonics Research_, 4(12), 2023. https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/adpr.202300158
    
18. Fan Z, et al. "Deep learning-based inverse design of multi-functional metasurface absorbers." _Optics Letters_, 49(10), 2733, 2024. https://opg.optica.org/ol/abstract.cfm?uri=ol-49-10-2733
    
19. Liu Y, et al. "Reliable, efficient, and scalable photonic inverse design via physics-inspired deep learning." _Nature Computational Science_, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC12338875/
    
20. Li Z, Pestourie R, Lin Z, et al. "Empowering Metasurfaces with Inverse Design: Principles and Applications." _ACS Photonics_, 9(7), 2178–2192, 2022. https://doi.org/10.1021/acsphotonics.1c01850
    
21. Khatib O, Ren S, Malof J, Padilla WJ. "Learning the physics of all-dielectric metamaterials with deep Lorentz neural networks." _Advanced Optical Materials_, 10, 2200097, 2022. https://doi.org/10.1002/adom.202200097
    
22. Chen Y, Lu L, Karniadakis GE, Negro LD. "Physics-informed neural networks for inverse problems in nano-optics and metamaterials." _Optics Express_, 28(8), 11618, 2020. https://doi.org/10.1364/OE.384875
    
23. Ji W, et al. "Recent advances in metasurface design with ML, PINNs, and topology optimization." _Light: Science & Applications_, 12(1), 169, 2023. https://doi.org/10.1038/s41377-023-01218-y
    
24. Ma W, Cheng F, Xu Y, Wen Q, Liu Y. "Probabilistic representation and inverse design of metamaterials via deep generative model." _Advanced Materials_, 2019. https://doi.org/10.1002/adma.201901111
    
25. Zandehshahvar M, et al. "Manifold Learning for Knowledge Discovery and Intelligent Inverse Design of Photonic Nanostructures." _ACS Photonics_, 9(2), 714–721, 2022. https://doi.org/10.1021/acsphotonics.1c01888
    
26. Gao F, et al. "Meta-Attention Deep Learning for Smart Development of Metasurface Sensors (Metaformer)." _Advanced Science_, 11(42), 2405750, 2024. https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202405750
    
27. Yu H, et al. "AI-driven approaches in electromagnetic metamaterials design and application: a review." _EPJ Applied Metamaterials_, 2025. https://epjam.edp-open.org/articles/epjam/full_html/2025/01/epjam250030/epjam250030.html
    
28. Islam MZ, et al. "Manufacturability-aware deep generative design of 3D metamaterial units for AM." _Structural and Multidisciplinary Optimization_, 2024. https://link.springer.com/article/10.1007/s00158-023-03732-4
    
29. Dong Y, An S, Jiang H, et al. "Advanced deep learning approaches in metasurface modeling and design: A review." _Progress in Quantum Electronics_, 99, 100554, 2025. https://doi.org/10.1016/j.pquantelec.2025.100554
    
30. Wang Q, Zhang Y, et al. "The transformational dive of nanophotonics inverse design from deep learning to AGI." _APL Photonics_, 9(10), 2024. https://doi.org/10.1063/5.0226592
    
31. Gahlmann T, et al. "Free-Form Diffractive Metagrating Design Based on Generative Adversarial Networks." _ACS Nano_, 2021. https://pubs.acs.org/doi/10.1021/acsnano.9b02371
    
32. Liu Z, Zhu D, Rodrigues SP, et al. "Generative model for the inverse design of metasurfaces." _Nano Letters_, 18(10), 6570–6576, 2018. https://doi.org/10.1021/acs.nanolett.8b03171
    
33. Wang J, Yao B, et al. "Generative adversarial networks for high degree of freedom metasurface designs." _Advanced Composites and Hybrid Materials_, 2024. https://link.springer.com/article/10.1007/s42114-024-01190-0
    
34. Zhang Z, Yang C, Qin Y, Feng H, Feng J, Li H. "Diffusion probabilistic model based accurate and high-degree-of-freedom metasurface inverse design." _Nanophotonics_, 12(20), 3871–3881, 2023. https://doi.org/10.1515/nanoph-2023-0292
    
35. Deng B, Zareei A, Ding X, Weaver JC, Rycroft CH, Bertoldi K. "Inverse design of mechanical metamaterials with target nonlinear response via a neural accelerated evolution strategy." _Advanced Materials_, 34(41), 2206238, 2022. https://doi.org/10.1002/adma.202206238
    

---

### Mechanical Metamaterials

36. Zheng X, Zhang X, Chen TT, Watanabe I. "Deep learning in mechanical metamaterials: From prediction and generation to inverse design." _Advanced Materials_, 35(45), e2302530, 2023. https://doi.org/10.1002/adma.202302530
    
37. Ha CS, Yao D, Xu Z, et al. "Rapid inverse design of metamaterials based on prescribed mechanical behavior through ML." _Nature Communications_, 14, 5765, 2023. https://doi.org/10.1038/s41467-023-40854-1
    
38. Pahlavani H, et al. "Deep learning for the rare-event rational design of 3D printed multi-material mechanical metamaterials." _Communications Materials_, 3, 46, 2022. https://doi.org/10.1038/s43246-022-00270-2
    
39. Zheng X, Chen TT, Guo X, et al. "Controllable inverse design of auxetic metamaterials using deep learning." _Materials & Design_, 2021. https://doi.org/10.1016/j.matdes.2021.110178
    
40. Bastek J-H, Bhattacharya K, Reiter P, et al. "Inverting the structure-property map of truss metamaterials by deep learning." _PNAS_, 119(1), 2022. https://doi.org/10.1073/pnas.2111505119
    
41. Meyer P, Bonatti C, Tancogne-Dejean T, Mohr D. "Graph-based metamaterials: Deep learning of structure-property relations." _Materials & Design_, 223, 111175, 2022. https://doi.org/10.1016/j.matdes.2022.111175
    
42. Xue T, Liao S, Gan Z, et al. "Learning the nonlinear dynamics of soft mechanical metamaterials with graph networks." _Computer Methods in Applied Mechanics and Engineering_, 2023. https://arxiv.org/abs/2202.13775
    
43. Li J, Ye H, Wei N, et al. "Efficient multi-material topology optimization via ResUNet-GAN." _Acta Mechanica Sinica_, 40, 423185, 2024. https://link.springer.com/article/10.1007/s10409-023-23185-x
    
44. Islam MZ, et al. "Manufacturability-aware deep generative design of 3D metamaterial units for additive manufacturing." _Structural and Multidisciplinary Optimization_, 2024. https://link.springer.com/article/10.1007/s00158-023-03732-4
    
45. Abueidda DW, Khan KA, et al. "Designing a TPMS metamaterial via deep learning and topology optimization." _Frontiers in Mechanical Engineering_, 10, 1417606, 2024. https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2024.1417606/full
    
46. Garland AP, White BC, Jensen SC, Boyce BL. "Pragmatic generative optimization of novel structural lattice metamaterials with ML." _Materials & Design_, 203, 109632, 2021. https://doi.org/10.1016/j.matdes.2021.109632
    
47. Bessa MA, Bostanabad R, Liu Z, et al. "A framework for data-driven analysis of materials under uncertainty." _Computer Methods in Applied Mechanics and Engineering_, 2017. https://doi.org/10.1016/j.cma.2017.03.037
    
48. Wu L, Liu L, Wang Y, et al. "A machine learning-based method to design modular metamaterials." _Extreme Mechanics Letters_, 36, 100657, 2020. https://doi.org/10.1016/j.eml.2020.100657
    
49. Ha CS, Yao D, et al. "Rapid inverse design of metamaterials via ML." _Nature Communications_, 2023. https://doi.org/10.1038/s41467-023-40854-1
    

---

### Acoustic / Phononic Metamaterials

49. Lu J-H, et al. "Bio-inspired acoustic metamaterials for traffic noise control with machine learning." _Communications Engineering_, 2025. https://www.nature.com/articles/s44172-025-00470-x
    
50. On S, et al. "Deep-Learning-Based Acoustic Metamaterial Design for Attenuating Structure-Borne Noise." _Materials_, 16(5), 1879, 2023. https://www.mdpi.com/1996-1944/16/5/1879
    
51. Shah T, Zhuo L, Lai P, et al. "Reinforcement learning applied to metamaterial design." _Journal of the Acoustical Society of America_, 150(1), 321–338, 2021. https://doi.org/10.1121/10.0005545
    
52. Rosafalco L, De Ponti JM, Iorio L, et al. "Reinforcement learning optimisation for graded metamaterial design." _Scientific Reports_, 13, 21836, 2023. https://www.nature.com/articles/s41598-023-48927-3
    
53. Zhang J, Li Y, Zhao T, et al. "Machine-learning based design of digital materials for elastic wave control." _Extreme Mechanics Letters_, 48, 101372, 2021. https://doi.org/10.1016/j.eml.2021.101372
    
54. Jiang W, Zhu Y, Yin G, et al. "Dispersion relation prediction and structure inverse design of elastic metamaterials via deep learning." _Materials Today Physics_, 22, 100616, 2022. https://doi.org/10.1016/j.mtphys.2022.100616
    
55. Han S, Han Q, Li C. "Deep-learning-based inverse design of phononic crystals for anticipated wave attenuation." _Journal of Applied Physics_, 132, 154901, 2022. https://doi.org/10.1063/5.0111610
    
56. Miao X, Dong HW, Wang Y. "Deep learning of dispersion engineering in two-dimensional phononic crystals." _Engineering Optimization_, 55, 125–139, 2023. https://doi.org/10.1080/0305215X.2022.2060280
    
57. Meyer P, et al. "Graph Based Metamaterials: Deep Learning of Structure-Property Relations." _Materials & Design_, 2022. https://doi.org/10.1016/j.matdes.2022.111175
    
58. Donda K, Singh R, Kumar R, et al. "Ultrathin acoustic absorbing metasurface based on deep learning approach." _Smart Materials and Structures_, 30(8), 085003, 2021. https://doi.org/10.1088/1361-665X/ac0b80
    
59. Che C, et al. "Deep reinforcement learning empowers automated inverse design and optimization of photonic crystals." _Nanophotonics_, 12(2), 319–334, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC11501320/
    
60. Xue T, et al. "Learning the nonlinear dynamics of soft mechanical metamaterials with graph networks." _arXiv:2202.13775_, 2022. https://arxiv.org/abs/2202.13775
    

---

### Topology Optimization

61. Li J, Ye H, Wei N, et al. "ResUNet-GAN for topology optimization of 2D microstructures." _Mathematics and Mechanics of Solids_, 2024. https://journals.sagepub.com/doi/10.1177/10812865241233013
    
62. Wang Z, et al. "Structural topology optimization based on diffusion generative adversarial networks." _Engineering Applications of Artificial Intelligence_, 2024. https://www.sciencedirect.com/science/article/abs/pii/S0952197624016026
    
63. Kollmann HT, Abueidda DW, Koric S. "Deep learning for topology optimization of 2D metamaterials." _Materials & Design_, 196, 109098, 2020. https://doi.org/10.1016/j.matdes.2020.109098
    
64. Huang X, Chen Y, Miao C, et al. "Generative adversarial network for stress-minimizing topology optimization." _International Journal of Mechanics and Materials in Design_, 2025. https://link.springer.com/article/10.1007/s10999-025-09811-2
    

---

### Physics-Informed Neural Networks

66. Karniadakis GE, Kevrekidis IG, Lu L, et al. "Physics-informed machine learning." _Nature Reviews Physics_, 3, 422, 2021. https://doi.org/10.1038/s42254-021-00314-5
    
67. Chen Y, Lu L, Karniadakis GE, Negro LD. "Physics-informed neural networks for inverse problems in nano-optics and metamaterials." _Optics Express_, 28(8), 11618, 2020. https://doi.org/10.1364/OE.384875
    
68. Khaireh-Walieh A, et al. "A Newcomer's Guide to Deep Learning for Inverse Design in Nano-Photonics." _arXiv:2307.08618_, 2023. https://arxiv.org/abs/2307.08618
    

---

### Multiscale Modeling

69. Zhu Y, et al. "ML-encoded multiscale modelling and Bayesian optimization for programmable metamaterials." _Acta Mechanica Sinica_, 2024. https://link.springer.com/article/10.1007/s10409-024-24061-x
    
70. Bhattacharya K, et al. "Iterated learning and multiscale modeling of history-dependent architected metamaterials." _arXiv_, 2024. https://arxiv.org/abs/2402.12674
    
71. Rosafalco L, et al. "Similarity equivariant graph neural networks for homogenization of metamaterials." _arXiv:2404.17365_, 2024. https://arxiv.org/abs/2404.17365
    
72. Black N, Najafi AR. "Deep neural networks for parameterized homogenization in concurrent multiscale structural optimization." _Structural and Multidisciplinary Optimization_, 66, 2023. https://doi.org/10.1007/s00158-023-03571-3
    

---

### Active Learning & Bayesian Optimization

73. Liu J, et al. "Bayesian active learning for accelerated design of broadband polarization-insensitive metasurfaces." _Intelligent Computing_, 4, 0135, 2025. https://doi.org/10.34133/icomputing.0135
    
74. Cao B, et al. "On-Demand Inverse Design of Metamaterials Using Deep Neural Networks with Bayesian Optimization." _Intelligent Computing_, 2024. https://spj.science.org/doi/10.34133/icomputing.0139
    

---

### Plasmonics / Nanophotonics

75. Peurifoy J, et al. "Nanophotonic particle simulation and inverse design using artificial neural networks." _Science Advances_, 4(6), eaar4206, 2018. https://doi.org/10.1126/sciadv.aar4206
    
76. Han JH, et al. "Neural-Network-Enabled Design of a Chiral Plasmonic Nanodimer for Chirality Sensing." _ACS Nano_, 17(3), 2306–2317, 2023. https://doi.org/10.1021/acsnano.2c08867
    
77. Guan Q, Raza A, Mao SS, et al. "Machine Learning-Enabled Inverse Design of Radiative Cooling Film." _ACS Photonics_, 10(3), 715–726, 2023. https://doi.org/10.1021/acsphotonics.2c01857
    
78. Augenstein Y, Repan T, Rockstuhl C. "Neural Operator-Based Surrogate Solver for Free-Form Electromagnetic Inverse Design." _ACS Photonics_, 2023. https://doi.org/10.1021/acsphotonics.3c00606
    
79. Liang B, et al. "Physics-Guided Neural-Network-Based Inverse Design of a Photonic-Plasmonic Nanodevice." _ACS Applied Materials & Interfaces_, 14(23), 27397, 2022. https://doi.org/10.1021/acsami.2c05083
    
80. Lininger A, Hinczewski M, Strangi G. "General Inverse Design of Layered Thin-Film Materials with CNNs." _ACS Photonics_, 8(12), 3641–3650, 2021. https://doi.org/10.1021/acsphotonics.1c01498
    
81. Gao L, Li X, Liu D, et al. "Deep-Learning-Enabled On-Demand Design of Chiral Metamaterials." _ACS Nano_, 2020. https://pubs.acs.org/doi/10.1021/acsnano.8b03569
    
82. Zheng X, Chen TT, Guo X, et al. "Controllable inverse design of auxetic metamaterials using deep learning." _Materials & Design_, 2021. https://doi.org/10.1016/j.matdes.2021.110178
    

---

### Surrogate Models & Neural Architectures

83. Feng F, Huo D, Zang Z, et al. "Symbiotic evolution of photonics and artificial intelligence: a comprehensive review." _Advanced Photonics_, 7(2), 024001, 2025. https://doi.org/10.1117/1.AP.7.2.024001
    
84. Rade J, et al. "Algorithmically-consistent deep learning frameworks for structural topology optimization." _Engineering Applications of Artificial Intelligence_, 106, 104483, 2021. https://doi.org/10.1016/j.engappai.2021.104483
    
85. Bonfanti S, Guerra R, Font-Clos F, et al. "Automatic design of mechanical metamaterial actuators." _Nature Communications_, 11, 4162, 2020. https://doi.org/10.1038/s41467-020-17947-2
    
86. Wen E, Yang X, Sievenpiper DF. "Real-data-driven real-time reconfigurable microwave reflective surface." _Nature Communications_, 14, 7736, 2023. https://doi.org/10.1038/s41467-023-43473-y
    
87. Xu YD, et al. "Recent Advances in Reconfigurable Metasurfaces." _Nanomaterials_, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC9921398/
    
88. Gao L, et al. "A bidirectional deep neural network for accurate silicon color design." _Advanced Materials_, 31(51), 1905467, 2019. https://doi.org/10.1002/adma.201905467
    
89. Liu D, Tan Y, Khoram E, Yu Z. "Training deep neural networks for the inverse design of nanophotonic structures." _ACS Photonics_, 5(4), 1365–1369, 2018. https://doi.org/10.1021/acsphotonics.7b01377
    

---

### Thermal Metamaterials

86. Wang Y, Sha W, Xiao M, Gao L. "Thermal Metamaterials with Configurable Mechanical Properties." _Physical Review Applied_, 2024. https://doi.org/10.1103/PhysRevApplied.21.044060
    
87. Kim S, Wu S, Jian R, et al. "Machine learning for predicting thermal metamaterial properties." _ACS Nano_, 2022. https://doi.org/10.1021/acsnano.2c05072
    
88. Zhu C, Bamidele EA, Shen X, Zhu G, Li B. "Machine Learning Aided Design and Optimization of Thermal Metamaterials." _Chemical Reviews_, 124(7), 4258–4331, 2024. https://doi.org/10.1021/acs.chemrev.3c00708
    

---

### Structural Color, Holography, Topological

99. Pilozzi L, et al. "Machine learning inverse problem for topological photonics." _Communications Physics_, 2018. https://doi.org/10.1038/s42005-018-0058-8
    
100. Dai P, et al. "Inverse design of structural color: finding multiple solutions via conditional GANs." _Nanophotonics_, 11(13), 3057–3069, 2022. https://doi.org/10.1515/nanoph-2022-0095
    
101. Deng Y, Fan K, Jin B, Malof J, Padilla WJ. "Physics-informed learning in artificial electromagnetic materials." _Applied Physics Reviews_, 12(1), 2025. https://doi.org/10.1063/5.0232675
    
102. Torfeh M, Hsu CW. "Probabilistic inverse design of metasurfaces using mixture density neural networks." _Journal of Physics: Photonics_, 7(1), 015007, 2025. https://doi.org/10.1088/2515-7647/ad9b82
    
103. Ren H, et al. "Complex-amplitude metasurface holography with computational inverse design." _Science Advances_, 6(8), 2020. https://doi.org/10.1126/sciadv.aax1839
    
104. Gu Z, et al. "A solution to the dilemma for FSS inverse design using generative models." _IEEE Trans Antenna Propag_, 71(6), 5100–5109, 2023. https://doi.org/10.1109/TAP.2023.3266053
    
105. Sun A, et al. "Predicting Chern numbers in photonic crystals using GAN-based data augmentation." _Optics Express_, 33(2), 3005, 2025. https://doi.org/10.1364/OE.544553
    

---

### Fabrication & 4D Printing

121. Fang R, Zhang X, Song B, et al. "3D and 4D printing of electromagnetic metamaterials." _Engineering_, 51, 171, 2025. https://doi.org/10.1016/j.eng.2024.06.020
    
122. Jiang F, et al. "ML-evolutionary algorithm enabled design for 4D-printed active composite structures." _Advanced Functional Materials_, 32, 2022. https://doi.org/10.1002/adfm.202109805
    
123. Wilt JK, et al. "Accelerating auxetic metamaterial design with deep learning." _Advanced Engineering Materials_, 2020. https://doi.org/10.1002/adem.202070018
    
124. Chan Y-C, Ahmed F, Wang L, Chen W. "METASET: Exploring Shape and Property Spaces for Data-Driven Metamaterials Design." _Journal of Mechanical Design_, 143(3), 031707, 2021. https://doi.org/10.1115/1.4048629
    
125. Kumar S, et al. "Inverse-designed spinodoid metamaterials." _npj Computational Materials_, 6, 73, 2020. https://doi.org/10.1038/s41524-020-0341-6
    

---

### Foundations and Seminal Works

126. Molesky S, et al. "Inverse design in nanophotonics." _Nature Photonics_, 12, 659–670, 2018. https://doi.org/10.1038/s41566-018-0246-9
    
127. Yu N, Capasso F. "Flat optics with designer metasurfaces." _Nature Materials_, 13, 139–150, 2014. https://doi.org/10.1038/nmat3839
    
128. Pendry JB. "Negative refraction makes a perfect lens." _Physical Review Letters_, 85, 3966, 2000. https://doi.org/10.1103/PhysRevLett.85.3966
    
129. Smith DR, Pendry JB, Wiltshire MCK. "Metamaterials and negative refractive index." _Science_, 305, 788–792, 2004. https://doi.org/10.1126/science.1096796
    
130. Cummer SA, Christensen J, Alù A. "Controlling sound with acoustic metamaterials." _Nature Reviews Materials_, 1, 16001, 2016. https://doi.org/10.1038/natrevmats.2016.1
    
131. Ma G, Sheng P. "Acoustic metamaterials: From local resonances to broad horizons." _Science Advances_, 2, e1501595, 2016. https://doi.org/10.1126/sciadv.1501595
    
132. Bertoldi K, Vitelli V, Christensen J, et al. "Flexible mechanical metamaterials." _Nature Reviews Materials_, 2, 17066, 2017. https://doi.org/10.1038/natrevmats.2017.66
    
133. Kushwaha MS, et al. "Acoustic band structure of periodic elastic composites." _Physical Review Letters_, 71, 2022, 1993. https://doi.org/10.1103/PhysRevLett.71.2022
    
134. Jiang J, Fan JA. "Tandem neural networks for inverse design of metasurfaces." _ACS Nano_, 2019. https://doi.org/10.1021/acsnano.9b04498
    
135. Liu Z, et al. "GANs for inverse design of metasurfaces." _Nano Letters_, 18, 6570, 2018. https://doi.org/10.1021/acs.nanolett.8b03171
    
136. Boulaich MH, Ohamouddou S, et al. "AI-Assisted Metasurface Antennas Design/Optimization and Performance Enhancement Techniques: A Comprehensive Survey." _IEEE Access_ (preprint), 2024. https://www.techrxiv.org/users/973873/articles/1343718
    
137. Zeng Q, Zhao Z, Lei H, Wang P. "A deep learning approach for inverse design of gradient mechanical metamaterials." _International Journal of Mechanical Sciences_, 240, 107920, 2022. https://doi.org/10.1016/j.ijmecsci.2022.107920
    
138. Bastek J-H, Kochmann DM. "Inverse design of nonlinear mechanical metamaterials via video denoising diffusion models." _Nature Machine Intelligence_, 5, 1466–1475, 2023. https://doi.org/10.1038/s42256-023-00762-x
    
139. Zheng X, Chen TT, Guo X, Samitsu S, Watanabe I. "Controllable inverse design of auxetic metamaterials using deep learning." _Materials & Design_, 211, 110178, 2021. https://doi.org/10.1016/j.matdes.2021.110178
    
140. Lee D, Chen W, Wang L, Chan Y-C, Chen W. "Data-driven design for metamaterials and multiscale systems: A review." _Advanced Materials_, 36(8), 2305254, 2024. https://doi.org/10.1002/adma.202305254
    

