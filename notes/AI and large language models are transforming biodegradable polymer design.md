---
tags:
 - claude
 - llm
 - LiteratureResearch
---


**The convergence of large language models, generative AI, and polymer informatics has created an unprecedented acceleration in the discovery and design of biodegradable polymers.** Between 2020 and 2025, over 75 papers have established the foundations of this interdisciplinary field — from transformer-based property predictors and graph neural networks to multi-objective Bayesian optimization of degradable formulations. The most consequential advances include polyBERT's ability to fingerprint 100 million polymers, multitask neural networks that identified 14 PHA bioplastics to replace petroleum-based commodity plastics, and the first regression model for predicting polymer biodegradation rates. Yet the field remains nascent for biodegradable systems specifically: most generative AI work has targeted dielectrics and electrolytes, leaving substantial whitespace for sustainability-focused inverse design. This review catalogues and analyzes all major contributions, organized thematically, to serve as a roadmap for future research at this critical intersection.

---

## 1. Transformer-Based Language Models for Polymer Property Prediction

The application of LLMs to polymer informatics represents the field's most rapid growth area. **polyBERT** (Kuenneth & Ramprasad, 2023) established the landmark: a DeBERTa-based chemical language model trained on 100 million hypothetical polymer SMILES strings that generates learned fingerprints predicting ~29 properties, outperforming handcrafted descriptors by two orders of magnitude in speed while preserving accuracy.![Fig. 1: Polymer informatics with polyBERT.|500](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-023-39868-6/MediaObjects/41467_2023_39868_Fig1_HTML.png)

**TransPolymer** (Xu et al., 2023) adopted a RoBERTa architecture with a chemically aware tokenizer, pre-trained via masked language modeling on ~5 million augmented polymer SMILES from the PI1M dataset. It demonstrated superior performance across 10 polymer property benchmarks including glass transition temperature (Tg), bandgap, and dielectric constant.![Fig. 1: Overview of TransPolymer.|500](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41524-023-01016-5/MediaObjects/41524_2023_1016_Fig1_HTML.png)

**PolyNC** (Qiu et al., 2024) pushed boundaries by combining natural language prompts with SMILES in a T5-based text-to-text framework, handling both regression and classification across 22,970 curated data points. This model treats polymer property prediction as a unified language task — a paradigm that may reshape how researchers interact with polymer data.![d3sc05079c-f2.gif|400](https://pubs.rsc.org/image/article/2024/sc/d3sc05079c/d3sc05079c-f2.gif)

General-purpose LLMs have also proven effective. Jablonka et al. (2024) demonstrated that fine-tuned GPT-3 performs comparably to or outperforms dedicated ML models for molecular property prediction, especially in low-data regimes, and even enables inverse design by simply inverting questions. Hatakeyama-Sato et al. (2023) showed that GPT-4 can assist with molecular descriptor selection for polymer informatics by leveraging its embedded domain knowledge to guide feature engineering. Most recently, Gupta et al. (2025) benchmarked fine-tuned LLaMA-3-8B and GPT-3.5 against polyBERT, polyGNN, and Polymer Genome pipelines for thermal property prediction, providing the first systematic comparison of general-purpose LLMs versus polymer-specific models.

[**Predicting polymerization reactions via transfer learning using chemical language models**](https://www.nature.com/articles/s41524-024-01304-8?fromPaywallRec=false)Polymers are candidate materials for a wide range of sustainability applications such as carbon capture and energy storage. However, computational polymer discovery lacks automated analysis of reaction pathways and stability assessment through retro-synthesis. Here, we report an extension of transformer-based language models to polymerization for both reaction and retrosynthesis tasks. To that end, we have curated a polymerization dataset for vinyl polymers covering reactions and retrosynthesis for representative homo-polymers and co-polymers. Overall, we obtain a forward model Top-4 accuracy of 80% and a backward model Top-4 accuracy of 60%. We further analyze the model performance with representative polymerization examples and evaluate its prediction quality from a materials science perspective. To enable validation and reuse, we have made our models and data available in public repositories.
![Fig. 1: Problem representation.](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41524-024-01304-8/MediaObjects/41524_2024_1304_Fig1_HTML.png)

Transfer learning and foundation models address polymer science's chronic data scarcity. Zhang et al. (2023) showed that transformer models pre-trained on large small-molecule datasets can be successfully fine-tuned on polymer tasks. Wang et al. (2024) introduced MMPolymer, a multimodal multitask pretraining framework combining structural and textual polymer representations.
[**On-demand reverse design of polymers with PolyTAO**](https://www.nature.com/articles/s41524-024-01466-5?fromPaywallRec=false)
![Fig. 1: Polymer Generation with PolyTAO.](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41524-024-01466-5/MediaObjects/41524_2024_1466_Fig1_HTML.png)


### References — Section 1

| #   | Title                                                                                            | Authors                      | Journal                     | Year | Link                                              |
| --- | ------------------------------------------------------------------------------------------------ | ---------------------------- | --------------------------- | ---- | ------------------------------------------------- |
| 1   | polyBERT: A chemical language model to enable fully machine-driven ultrafast polymer informatics | Kuenneth, C. & Ramprasad, R. | Nature Communications       | 2023 | [DOI](https://doi.org/10.1038/s41467-023-39868-6) |
| 2   | TransPolymer: A transformer-based language model for polymer property predictions                | Xu, C. et al.                | npj Computational Materials | 2023 | [DOI](https://doi.org/10.1038/s41524-023-01016-5) |
| 3   | PolyNC: A natural and chemical language model for unified polymer properties                     | Qiu, H. et al.               | Chemical Science            | 2024 | [DOI](https://doi.org/10.1039/D3SC05079C)         |
| 4   | Leveraging large language models for predictive chemistry                                        | Jablonka, K.M. et al.        | Nature Machine Intelligence | 2024 | [DOI](https://doi.org/10.1038/s42256-023-00788-1) |
| 5   | Using GPT-4 in parameter selection of polymer informatics                                        | Hatakeyama-Sato, K. et al.   | Digital Discovery           | 2023 | [DOI](https://doi.org/10.1039/D3DD00138E)         |
| 6   | Benchmarking large language models for polymer property predictions                              | Gupta, S. et al.             | arXiv preprint              | 2025 | [arXiv](https://arxiv.org/abs/2506.02129)         |
| 7   | Transferring a molecular foundation model for polymer property predictions                       |                              |                             | 2023 | [DOI](https://doi.org/10.1021/acs.jcim.3c01650)   |
| 8   | MMPolymer: A multimodal multitask pretraining framework for polymer property prediction          |                              |                             | 2024 | [DOI](https://doi.org/10.1145/3627673.3679590)    |

---

## 2. NLP Pipelines for Extracting Polymer Data from Literature

Automated extraction of polymer data from the scientific literature has matured into a scalable pipeline. **MaterialsBERT** (Shetty et al., 2023) — a BERT model pre-trained on 2.4 million materials science abstracts — powers a pipeline that extracted ~300,000 material property records from ~130,000 polymer abstracts in 60 hours, outperforming BioBERT, ChemBERT, and MatSciBERT on 3 of 5 NER benchmarks.![Fig. 1: Pipeline used for extracting material property records from a corpus of abstracts.](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41524-023-01003-w/MediaObjects/41524_2023_1003_Fig1_HTML.png)

This effort scaled dramatically when Gupta et al. (2024) combined GPT-3.5 and LLaMA-2 with MaterialsBERT to process full-text articles rather than abstracts alone, extracting over one million records for 24 properties of 106,000+ unique polymers from ~2.4 million articles. This represents the largest automated polymer data extraction effort to date.![Fig. 1: Overall workflow to extract polymer property data.](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs43246-024-00708-9/MediaObjects/43246_2024_708_Fig1_HTML.png)

Earlier foundational work by Shetty et al. (2021) demonstrated that Word2Vec embeddings trained on 500,000 polymer papers can capture latent materials science knowledge and predict new polymer-application pairings. The same group addressed the polymer name normalization challenge (Shetty & Ramprasad, 2021), achieving F1 = 0.85 for mapping diverse polymer names to canonical forms — a critical preprocessing step.

**MatSciBERT** (Gupta et al., 2022) established a materials-domain language model by continuing pre-training of SciBERT on a 285-million-word materials corpus, achieving state-of-the-art NER and relation classification. The **PolyIE** benchmark (Cheung et al., 2024) — the first scientific information extraction dataset for polymers, with expert annotations from 146 full-length articles — found that fine-tuned domain-specific models outperform GPT-3.5 in few-shot polymer NER.

### References — Section 2

|#|Title|Authors|Journal|Year|Link|
|---|---|---|---|---|---|
|9|A general-purpose material property data extraction pipeline from large polymer corpora using NLP|Shetty, P. et al.|npj Computational Materials|2023|[DOI](https://doi.org/10.1038/s41524-023-01003-w)|
|10|Data extraction from polymer literature using large language models|Gupta, S. et al.|Communications Materials|2024|[DOI](https://doi.org/10.1038/s43246-024-00708-9)|
|11|Automated knowledge extraction from polymer literature using NLP|Shetty, P. et al.|iScience|2021|[DOI](https://doi.org/10.1016/j.isci.2020.101922)|
|12|Machine-guided polymer knowledge extraction: named entity normalization|Shetty, P. & Ramprasad, R.|J. Chem. Inf. Model.|2021|[DOI](https://doi.org/10.1021/acs.jcim.1c00554)|
|13|MatSciBERT: A materials domain language model for text mining|Gupta, T. et al.|npj Computational Materials|2022|[DOI](https://doi.org/10.1038/s41524-022-00784-w)|
|14|PolyIE: A dataset of information extraction from polymer material scientific literature|Cheung, J. et al.|NAACL 2024|2024|[DOI](https://doi.org/10.18653/v1/2024.naacl-long.131)|

---

## 3. Graph Neural Networks and Learned Polymer Representations

Polymer representation has undergone a paradigm shift from handcrafted fingerprints to learned representations. **polyGNN** (Gurnani et al., 2023) introduced a GNN architecture that learns fingerprints directly from SMILES-based periodic graphs with invariance under translation and repeat-unit addition, predicting up to 36 properties for 13,388 polymers while accelerating feature extraction by 1–2 orders of magnitude.

The challenge that polymers are molecular ensembles rather than single molecules was addressed by Aldeghi & Coley (2022), who developed a weighted directed message-passing neural network (wD-MPNN) capturing chain architecture, monomer stoichiometry, and degree of polymerization. Queen et al. (2023) built PolymerGNN for multitask property prediction of complex heterogeneous polyesters.

Self-supervised learning has emerged as a key strategy. Gao et al. (2024) demonstrated that ensemble self-supervised pre-training reduces RMSE by up to 28.4% for polymer electronic properties. Volgin et al. (2022) showed transfer learning from synthetic datasets of ~50,000 samples dramatically boosts GNN prediction for polymer families with limited experimental data.

Traditional ML benchmarking remains important. Tao et al. (2021) systematically compared 79 ML models for Tg prediction, finding random forests with Morgan fingerprints yielded the best generalization. Park et al. (2022) and Hu et al. (2023) demonstrated graph convolutional networks achieving R² ~0.9 for Tg with enhanced interpretability. Chen et al. (2024) assembled a >900-polymer Tg dataset and deployed it as a public web application. Li et al. (2024) introduced Lieconv-Tg using Lie group equivariant neural networks for orientation-invariant Tg prediction from 3D structures. Ru & Li (2024) combined graph attention networks with XGBoost in GATBoost, mining important substructures.

### References — Section 3

|#|Title|Authors|Journal|Year|Link|
|---|---|---|---|---|---|
|15|Polymer informatics at scale with multitask graph neural networks|Gurnani, R. et al.|Chemistry of Materials|2023|[DOI](https://doi.org/10.1021/acs.chemmater.2c02991)|
|16|A graph representation of molecular ensembles for polymer property prediction|Aldeghi, M. & Coley, C.W.|Chemical Science|2022|[DOI](https://doi.org/10.1039/D2SC02839E)|
|17|Polymer graph neural networks for multitask property learning|Queen, O. et al.|npj Computational Materials|2023|[DOI](https://doi.org/10.1038/s41524-023-01034-3)|
|18|Self-supervised graph neural networks for polymer property prediction|Gao, Q. et al.|Mol. Syst. Des. Eng.|2024|[DOI](https://doi.org/10.1039/D4ME00088A)|
|19|ML with enormous synthetic data sets: predicting Tg of polyimides using GCNN|Volgin, I.V. et al.|ACS Omega|2022|[DOI](https://doi.org/10.1021/acsomega.2c04649)|
|20|Benchmarking ML models for polymer informatics: Tg example|Tao, L. et al.|J. Chem. Inf. Model.|2021|[DOI](https://doi.org/10.1021/acs.jcim.1c01031)|
|21|Prediction and interpretation of polymer properties using GCN|Park, J. et al.|ACS Polymers Au|2022|[DOI](https://doi.org/10.1021/acspolymersau.1c00050)|
|22|Prediction and interpretability of Tg by data-augmented GCNN|Hu, J. et al.|ACS Appl. Mater. Interfaces|2023|[DOI](https://doi.org/10.1021/acsami.3c12649)|
|23|ML analysis of a large set of homopolymers to predict Tg|Chen, M. et al.|Communications Chemistry|2024|[DOI](https://doi.org/10.1038/s42004-024-01305-0)|
|24|Large-scale Tg prediction with equivariant neural network (Lieconv-Tg)|Li, Z. et al.|ACS Omega|2024|[DOI](https://doi.org/10.1021/acsomega.3c06843)|
|25|GATBoost: Mining GAT-based important substructures for property prediction|Ru, Y. & Li, D.|Materials Today Communications|2024|[DOI](https://doi.org/10.1016/j.mtcomm.2023.107645)|
|26|Predicting polymers' Tg by a chemical language processing model|Chen, G. et al.|Polymers (MDPI)|2021|[DOI](https://doi.org/10.3390/polym13111898)|

---

## 4. Generative AI and Reinforcement Learning for De Novo Polymer Design

Generative models for polymer inverse design have advanced from variational autoencoders to diffusion transformers. Batra et al. (2020) pioneered syntax-directed VAEs combined with Gaussian process regression to generate polymers robust under extreme conditions, validated by DFT. **polyG2G** (Gurnani et al., 2021) introduced graph-to-graph translation that learns subtle chemical differences between high- and low-performing polymers, generating thousands of novel dielectric candidates confirmed by DFT.

More recent architectures have expanded capability. Jiang et al. (2024) developed a VAE framework for generating topologically diverse polymers (linear, cyclic, branched, dendritic) with target dilute-solution properties. **PolyTAO** (Qiu & Sun, 2024) is a transformer-based autoregressive model trained on ~1 million structure-property pairs for on-demand polymer reverse design. Yang et al. (2024) compared GPT-based (minGPT) and diffusion models for polymer electrolyte design, finding 17 of 46 MD-tested candidates exhibited superior ionic conductivity.

**Graph DiT** (Liu et al., 2024) — accepted as an Oral at NeurIPS 2024 — represents the state-of-the-art in multi-conditional molecular generation using a diffusion transformer with a graph-dependent noise model. Liu et al. (2023) demonstrated invertible normalizing flow models for molecular graph generation of high-temperature polymer dielectrics. Das et al. (2025) introduced the first CVAE for thermoset shape memory polymers. Vogel et al. (2024) developed a graph-to-string VAE for copolymer inverse design including monomer stoichiometry and chain architecture. Khajeh et al. (2025) presented a self-improvable polymer discovery framework using conditional generative models.![Fig. 1: Depiction of the workflow of the work reported here.|500](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41524-024-01470-9/MediaObjects/41524_2024_1470_Fig1_HTML.png)

Yue et al. (2025) provided the first comprehensive benchmark of six deep generative models (VAE, AAE, ORGAN, CharRNN, REINVENT, GraphINVENT) for de novo polymer design.

**Reinforcement learning** has emerged as a powerful tool for property-targeted polymer generation. RLPolyG (2025) generated 4,991 novel polymers with 45.2% improvement in average yield strength. **PolyRL** (Li et al., 2025) established the first standardized benchmark for RL-based polymer generation, integrating GPT-2, LLaMA-2, and multiple RL algorithms (REINVENT, REINFORCE, DPO, PPO).
![](https://pubs.rsc.org/image/article/2026/dd/d5dd00272a/d5dd00272a-f1.gif)
### References — Section 4

|#|Title|Authors|Journal|Year|Link|
|---|---|---|---|---|---|
|27|Polymers for extreme conditions designed using syntax-directed VAEs|Batra, R. et al.|Chemistry of Materials|2020|[DOI](https://doi.org/10.1021/acs.chemmater.0c03332)|
|28|polyG2G: A novel ML algorithm for generative design of polymer dielectrics|Gurnani, R. et al.|Chemistry of Materials|2021|[DOI](https://doi.org/10.1021/acs.chemmater.1c02061)|
|29|Property-guided generation of complex polymer topologies using VAEs|Jiang, S. et al.|npj Computational Materials|2024|[DOI](https://doi.org/10.1038/s41524-024-01328-0)|
|30|PolyTAO: On-demand reverse design of polymers|Qiu, H. & Sun, Z.-Y.|npj Computational Materials|2024|[DOI](https://doi.org/10.1038/s41524-024-01466-5)|
|31|De novo design of polymer electrolytes using GPT-based and diffusion-based models|Yang, Z. et al.|npj Computational Materials|2024|[DOI](https://doi.org/10.1038/s41524-024-01470-9)|
|32|Graph diffusion transformers for multi-conditional molecular generation|Liu, G. et al.|NeurIPS 2024|2024|[arXiv](https://arxiv.org/abs/2401.13858)|
|33|High-temperature polymer dielectrics using invertible molecular graph generative model|Liu, D.-F. et al.|J. Chem. Inf. Model.|2023|[DOI](https://doi.org/10.1021/acs.jcim.3c01572)|
|34|Generative design of thermoset shape memory polymers: a CVAE approach|Das, A. et al.|J. Polymer Science|2025|[DOI](https://doi.org/10.1002/pol.20240649)|
|35|Inverse design of copolymers including stoichiometry and chain architecture|Vogel, N. et al.|arXiv preprint|2024|[arXiv](https://arxiv.org/abs/2410.02824)|
|36|A materials discovery framework based on conditional generative models|Khajeh, A. et al.|Digital Discovery|2025|[DOI](https://doi.org/10.1039/D4DD00293H)|
|37|Benchmarking deep generative models for inverse polymer design|Yue, T. et al.|Digital Discovery|2025|[DOI](https://doi.org/10.1039/D4DD00395K)|
|38|De novo design of polymers with specified properties using RL (RLPolyG)|—|Macromolecules|2025|[DOI](https://doi.org/10.1021/acs.macromol.5c00427)|
|39|PolyRL: RL-guided polymer generation for multi-objective discovery|Li, W. et al.|Digital Discovery|2025|[DOI](https://doi.org/10.1039/D5DD00272A)|
|40|Generating high-temperature polymer dielectrics with GCN and RL|—|J. Phys. Chem. C|2025|[DOI](https://doi.org/10.1021/acs.jpcc.5c02310)|
|41|Discovery of polymers with high bulk modulus and low thermal conductivity via hybrid GA-RL|—|Chem. Eng. J.|2025|[DOI](https://doi.org/10.1016/j.cej.2025.155998)|

---

## 5. LLM-Assisted Retrosynthesis for Polymer Synthesis Planning

A critical gap in AI-driven polymer design — connecting generated structures to feasible synthesis routes — is beginning to close. Ma et al. (2025) presented the first fully automated retrosynthesis planning agent for macromolecules, integrating LLMs with knowledge graphs and a novel Multi-branched Reaction Pathway Search Algorithm (MBRPS) to handle polymer-specific challenges where products decompose into multiple intermediates.

**Llamole** (Liu et al., 2025) represents the most ambitious integration to date: a multimodal LLM that interleaves text and graph generation for molecular inverse design with retrosynthetic planning. Using A* search with LLM-based cost functions, Llamole increased retrosynthesis success rates from 5% to 35% for drugs and 17.9% for polymers, outperforming 14 adapted LLMs across 12 metrics.

Park et al. (2023) introduced Chemical Markdown Language (CMDL) for flexible representation of polymer experimental data, enabling fine-tuning of Regression Transformer models for generative design of catalysts for ring-opening polymerization. Five generated catalysts were experimentally synthesized and validated, including novel thiourea and guanidine catalysts for polymerization of biodegradable polymers like polylactide and polycarbonates — making this one of the few studies with closed-loop experimental validation.

### References — Section 5

|#|Title|Authors|Journal|Year|Link|
|---|---|---|---|---|---|
|42|Automated retrosynthesis planning of macromolecules using LLMs and knowledge graphs|Ma, Q. et al.|Macromol. Rapid Commun.|2025|[DOI](https://doi.org/10.1002/marc.202500065)|
|43|Llamole: Multimodal LLMs for inverse molecular design with retrosynthetic planning|Liu, G. et al.|ICLR 2025|2025|[arXiv](https://arxiv.org/abs/2410.04223)|
|44|AI-driven design of catalysts and materials for ring opening polymerization using CMDL|Park, N.H. et al.|Nature Communications|2023|[DOI](https://doi.org/10.1038/s41467-023-39396-3)|

---

## 6. ML Models for Biodegradation Prediction and Multi-Objective Sustainable Design

Sustainability-focused AI/ML for biodegradable polymers is the area with the greatest growth potential. Huang et al. (2025) developed the first regression ML model specifically for polymer biodegradation, curating an aerobic biodegradation dataset of 74 polymers (1,779 data points). Using Morgan fingerprints and thermal decomposition temperature, they achieved R²_test = 0.66, with SHAP analysis revealing that polyether/polysaccharide substructures promote biodegradation while aromatic rings and side chains inhibit it.![](https://pubs.acs.org/cms/10.1021/acs.est.4c11282/asset/images/medium/es4c11282_0001.gif)

The most impactful contribution for bioplastic design is Kuenneth et al. (2022), which trained multitask deep neural networks on ~23,000 polymer chemistries and identified 14 PHA-based bioplastics from ~1.4 million candidates as potential replacements for seven petroleum-based commodity plastics that account for 75% of yearly plastic production.![Fig. 1: Bioplastic design using multitask deep learning predictors.](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs43246-022-00319-2/MediaObjects/43246_2022_319_Fig1_HTML.png)

Multi-objective optimization has proven essential for overcoming the classic toughness-degradability trade-off. Zamengo et al. (2025) applied multi-objective Bayesian optimization with Gaussian process regression to simultaneously optimize degradation rate, strain at break, and Young's modulus for multiblock polyamides.
![Fig. 1: Schematic representation of this work.](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41524-025-01696-1/MediaObjects/41524_2025_1696_Fig1_HTML.png)
Several studies target specific biodegradable formulations. Yamawaki et al. (2021) constructed ML models for PLA decomposition under composting. White et al. (2024) used heteroscedastic Gaussian process Bayesian optimization to optimize compostable thermoplastic starch films. Multi-objective XGBoost + NSGA-II optimized PLA/spent coffee ground composites (2025). A Bayesian optimization framework using CNN-extracted NMR features for PLA process optimization appeared in npj Materials Degradation (2025).

For recyclable and sustainable polymers more broadly, Atasi et al. (2024) combined genetic algorithms with virtual forward synthesis to design chemically recyclable ring-opening polymerization polymers. Kern et al. (2025) developed an informatics framework screening commercially available monomers to identify ~37,000 candidate recyclable polymers. Kim et al. (2021) demonstrated genetic algorithms generating 132 new polymers meeting extreme property criteria. Löfgren et al. (2022) applied Bayesian optimization to lignin valorization in green biorefineries.

Gormley & Webb et al. (2023) reviewed ML for polymeric biomaterials, noting that lack of standardized degradation and biocompatibility data remains a major barrier. Laycock et al. (2024) reviewed the Materials 4.0 framework combining multiscale simulations with AI for biodegradable and biobased polymer design.

### References — Section 6

|#|Title|Authors|Journal|Year|Link|
|---|---|---|---|---|---|
|45|Polymer biodegradation in aquatic environments: ML model informed by meta-analysis|Huang, K. et al.|Environ. Sci. Technol.|2025|[DOI](https://doi.org/10.1021/acs.est.4c11282)|
|46|Bioplastic design using multitask deep neural networks|Kuenneth, C. et al.|Communications Materials|2022|[DOI](https://doi.org/10.1038/s43246-022-00319-2)|
|47|A ML approach to designing tough, degradable polyamides|Zamengo, M. et al.|npj Computational Materials|2025|[DOI](https://doi.org/10.1038/s41524-025-01696-1)|
|48|Decomposition factor analysis via Bayesian optimization for compost-degradable polymers|Yamawaki, R. et al.|Applied Sciences|2021|[DOI](https://doi.org/10.3390/app11062820)|
|49|Optimizing thermoplastic starch film with heteroscedastic GP in Bayesian design|White, G.M. et al.|Materials|2024|[DOI](https://doi.org/10.3390/ma17215345)|
|50|Multi-objective optimization of PLA/SCG/silane composites|—|RSC Advances|2025|[DOI](https://doi.org/10.1039/D5RA06825H)|
|51|Bayesian optimization of biodegradable polymers via ML-driven NMR features|—|npj Materials Degradation|2025|[DOI](https://doi.org/10.1038/s41529-025-00613-7)|
|52|Design of recyclable plastics with ML and genetic algorithm|Atasi, C. et al.|J. Chem. Inf. Model.|2024|[DOI](https://doi.org/10.1021/acs.jcim.4c01530)|
|53|An informatics framework for sustainable, chemically recyclable polymers|Kern, J. et al.|npj Computational Materials|2025|[DOI](https://doi.org/10.1038/s41524-025-01683-6)|
|54|Polymer design using genetic algorithm and machine learning|Kim, C. et al.|Comput. Mater. Sci.|2021|[DOI](https://doi.org/10.1016/j.commatsci.2020.110067)|
|55|ML optimization of lignin properties in green biorefineries|Löfgren, J. et al.|ACS Sustain. Chem. Eng.|2022|[DOI](https://doi.org/10.1021/acssuschemeng.2c01895)|
|56|Applied ML as a driver for polymeric biomaterials design|Gormley, A.J. & Webb, M.A. et al.|Nature Communications|2023|[DOI](https://doi.org/10.1038/s41467-023-40459-8)|
|57|Computational approaches for biodegradable and biobased polymers (review)|Laycock, B. et al.|Prog. Polym. Sci.|2024|[DOI](https://doi.org/10.1016/j.progpolymsci.2024.100911)|
|58|Design of polymer blend electrolytes through ML|Shen, K.-H. et al.|Macromolecules|2021|[DOI](https://doi.org/10.1021/acs.macromol.0c01547)|
|59|Accelerated development of biomass-based polyurethane adhesives via ML|—|ACS Appl. Mater. Interfaces|2025|[DOI](https://doi.org/10.1021/acsami.4c20371)|

---

## 7. Polymer Informatics Databases and Platforms

The quality and scope of polymer informatics depends fundamentally on data infrastructure. **PI1M** (Ma & Luo, 2020) is a benchmark database of ~1 million polymers generated by a model trained on ~12,000 PolyInfo entries, introducing polymer embedding representations. **Polymer Genome** (Tran et al., 2020) serves as the foundational web-based platform for near-instantaneous polymer property prediction using hierarchical chemical fingerprints combined with Gaussian process regression and neural networks.

Multi-task learning has been critical. Kuenneth et al. (2021) demonstrated that multi-task models using a combined dataset of 36 properties for 13,000+ polymers outperform single-task models by exploiting inherent property correlations.

Newer databases address remaining gaps. **OpenPoly** (2025) offers 3,985 curated experimental data points spanning 26 properties with a multi-task benchmarking framework. **POINT2** (2025) introduces standardized protocols covering prediction accuracy, uncertainty quantification, interpretability, and synthesizability assessment.

### References — Section 7

| #   | Title                                                                               | Authors             | Journal                    | Year | Link                                                |
| --- | ----------------------------------------------------------------------------------- | ------------------- | -------------------------- | ---- | --------------------------------------------------- |
| 60  | PI1M: A benchmark database for polymer informatics                                  | Ma, R. & Luo, T.    | J. Chem. Inf. Model.       | 2020 | [DOI](https://doi.org/10.1021/acs.jcim.0c00726)     |
| 61  | Machine-learning predictions of polymer properties with Polymer Genome              | Tran, H.D. et al.   | J. Applied Physics         | 2020 | [DOI](https://doi.org/10.1063/5.0023759)            |
| 62  | Polymer informatics with multi-task learning                                        | Kuenneth, C. et al. | Patterns                   | 2021 | [DOI](https://doi.org/10.1016/j.patter.2021.100238) |
| 63  | OpenPoly: A polymer database empowering benchmarking and multi-property predictions | —                   | Chinese J. Polymer Science | 2025 | [DOI](https://doi.org/10.1007/s10118-025-3402-y)    |
| 64  | POINT2: A polymer informatics training and testing database                         | —                   | arXiv preprint             | 2025 | [arXiv](https://arxiv.org/abs/2503.23491)           |

---

## 8. Review Papers and Perspectives

The 2020–2025 period produced an unprecedented number of authoritative reviews. The two most comprehensive are Chen et al. (2021) — a seminal review of the polymer informatics ecosystem — and Tran et al. (2024), which covers AI-enabled polymer design for energy storage, separation membranes, and sustainable/biodegradable polymers.

Audus & de Pablo (2023) highlighted unique polymer ML challenges including stochastic structure and representation difficulties. Ge et al. (2025) provided a comprehensive interdisciplinary review emphasizing FAIR data principles. Xie et al. (2025) compiled over 4,000 original reports through April 2024. Yue et al. (2025) reviewed ML-assisted molecular polymer design covering databases, representations, prediction, and generation.

### References — Section 8

|#|Title|Authors|Journal|Year|Link|
|---|---|---|---|---|---|
|65|Polymer informatics: Current status and critical next steps|Chen, L. et al.|Mater. Sci. Eng. R|2021|[DOI](https://doi.org/10.1016/j.mser.2020.100595)|
|66|Design of functional and sustainable polymers assisted by AI|Tran, H. et al.|Nature Reviews Materials|2024|[DOI](https://doi.org/10.1038/s41578-024-00708-8)|
|67|Emerging trends in ML: A polymer perspective|Audus, D.J. & de Pablo, J.J.|ACS Polymers Au|2023|[DOI](https://doi.org/10.1021/acspolymersau.2c00053)|
|68|Machine learning in polymer research|Ge, W. et al.|Advanced Materials|2025|[DOI](https://doi.org/10.1002/adma.202413695)|
|69|ML approaches in polymer science: progress and fundamentals|Xie, Y. et al.|SmartMat|2025|[DOI](https://doi.org/10.1002/smm2.1320)|
|70|ML-assisted molecular design of innovative polymers|Yue, T. et al.|Accounts of Materials Research|2025|[DOI](https://doi.org/10.1021/accountsmr.5c00151)|
|71|A review on molecular descriptors and ML in polymer design|Zhao, Y. et al.|Polymer Chemistry|2023|[DOI](https://doi.org/10.1039/D3PY00395G)|
|72|Recent advances in experiment-oriented polymer informatics|Hatakeyama-Sato, K.|Polymer Journal|2023|[DOI](https://doi.org/10.1038/s41428-022-00734-9)|
|73|Machine learning in polymer informatics|Sha, W. et al.|InfoMat|2021|[DOI](https://doi.org/10.1002/inf2.12167)|
|74|New opportunity: ML for polymer materials design and discovery|Xu, L. et al.|Adv. Theory Simul.|2022|[DOI](https://doi.org/10.1002/adts.202100565)|
|75|ML-assisted design of advanced polymeric materials|Wang, L. et al.|Accounts of Materials Research|2024|[DOI](https://doi.org/10.1021/accountsmr.3c00288)|
|76|Data-driven methods for accelerating polymer design|Batra, R.|ACS Polymers Au|2022|[DOI](https://doi.org/10.1021/acspolymersau.1c00035)|
|77|A prospective on ML challenges in polymer science|Abu-Mualla, J. et al.|MRS Communications|2024|[DOI](https://doi.org/10.1557/s43579-024-00587-8)|
|78|Data-efficient ML for polymer informatics|Xu, X.-Y. et al.|Chinese J. Polymer Science|2025|[DOI](https://doi.org/10.1007/s10118-025-3401-z)|

---

## Conclusions and Outlook

This literature review catalogues **78 unique papers from 2020–2025** spanning all major dimensions of AI/ML-driven polymer design. Three key insights emerge:

**First, the toolbox is mature but unevenly applied.** Transformer-based models (polyBERT, TransPolymer, PolyNC), graph neural networks (polyGNN, wD-MPNN), and generative frameworks (VAEs, diffusion models, RL pipelines) have demonstrated remarkable capability — yet the overwhelming majority of applications target dielectrics, electrolytes, and gas separation membranes. Biodegradable polymers (PLA, PHA, PCL, PBAT, starch-based, cellulose-based systems) remain dramatically underserved by generative AI despite being among the most societally urgent materials challenges.

**Second, data remains the binding constraint for biodegradable polymer AI.** While NLP pipelines can now extract over one million polymer-property records, biodegradation-specific data is sparse and heterogeneous. The first regression model for aquatic biodegradation appeared only in 2025 (Huang et al.), achieving a modest R² of 0.66. The community needs standardized biodegradation datasets — covering composting, soil, marine, and anaerobic conditions — to unlock the full power of modern ML architectures.

**Third, the integration of generative design with retrosynthesis and experimental validation represents the frontier.** Llamole's retrosynthesis success rate of 17.9% for polymers and Park et al.'s experimental validation of AI-designed ROP catalysts demonstrate that end-to-end pipelines are feasible. Applying these integrated approaches to biodegradable polymer design, where synthesis feasibility and environmental degradation must be co-optimized, represents the most impactful direction for future research.

The field is poised for rapid advancement: foundation models pre-trained on millions of polymer structures can be fine-tuned on small biodegradation datasets via transfer learning, multi-objective Bayesian optimization can balance degradability against mechanical performance, and LLM-assisted retrosynthesis can ensure generated candidates are synthetically accessible.