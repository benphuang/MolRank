Benchmarking SimSon Embeddings vs Classical Cheminformatics Similarity Methods

This repository contains a reproducible notebook and exported metrics evaluating SimSon molecular embeddings against standard non–deep-learning similarity methods including ECFP4 fingerprints, Tanimoto similarity, and (optionally) MACCS keys.
It is intended to provide a clear, practitioner-friendly assessment of whether SimSon’s learned embedding space meaningfully aligns with or improves upon traditional cheminformatics approaches.

1. Purpose

The goal of this benchmark is to compare:

SimSon embeddings (dense vector representations)

with

ECFP4 circular fingerprints (radius=2, 2048 bits)

MACCS Keys (optional)

across multiple retrieval and enrichment tasks.
The analysis aims to determine how well SimSon captures molecular similarity, bioactivity locality, and scaffold-level generalization compared to classical, non–deep-learning methods.

2. Dataset

We use the BACE inhibitor dataset provided via:

deepchem.molnet.load_bace_classification(featurizer="Raw", split="scaffold")


The merged dataset is exported as bace_subset.csv, containing:

smiles, active


Invalid SMILES are removed via RDKit sanitization.

3. Representations Compared
3.1 Classical Fingerprints

ECFP4 (Morgan radius=2, 2048 bits)
– Computed using RDKit’s GetMorganFingerprintAsBitVect.

MACCS Keys
– Optional baseline; not required for primary metrics.

Similarity metric:
Tanimoto similarity using RDKit’s DataStructs.TanimotoSimilarity.

3.2 SimSon Embeddings

The notebook supports:

Loading simson_embeddings.npy, or

Computing embeddings through a user-provided simson.compute_embedding(mol) function.

Embeddings are L2-normalized, enabling cosine similarity comparisons.

Similarity metric:
Cosine similarity (equivalent to inner-product on normalized vectors).

4. Benchmark Tasks

The notebook evaluates multiple aspects of molecular similarity performance.

4.1 Nearest-Neighbour Agreement (Retrieval Overlap)

For 100 randomly selected query molecules:

Compute top-10 neighbors using:

ECFP4 + Tanimoto

SimSon + Cosine

Measure Jaccard overlap between the two top-k lists.

This quantifies how similarly each method ranks molecular neighbors.

Reported metrics:

Mean, median, std of Jaccard@10 across queries.

4.2 Activity Enrichment (Local Virtual Screening)

For each molecule:

Rank all molecules by similarity

Compute ROC-AUC distinguishing active vs inactive

Produces:

ECFP4 AUC distribution

SimSon AUC distribution

A Wilcoxon signed-rank test comparing the two

This reflects how well each similarity measure preserves bioactivity locality.

4.3 Scaffold-Based Cross-Validation

Compute Bemis–Murcko scaffolds

Perform an 80/20 split by scaffold

Query test scaffolds against train scaffolds

Compare ECFP4 vs SimSon via Jaccard@10

Tests whether SimSon generalizes similarity across new structural families.

5. Additional Analyses
5.1 Similarity Distributions

Histogram of sampled ECFP4 Tanimoto similarities

Histogram of sampled SimSon cosine similarities

Used to inspect the geometry and density of each feature space.

5.2 UMAP Visualization (Optional)

If umap-learn is installed:

2D projection of SimSon embeddings

Points colored by activity label

Provides qualitative insight into embedding structure.

5.3 FAISS Indexing (Optional)

If FAISS is available:

Build IndexFlatIP for SimSon embeddings

Report:

Index build time

Mean query latency (ms/query)

Benchmarks practical scalability.

6. Exported Results

The notebook exports the following files:

6.1 Human-Readable Summary

benchmark_report.txt
Includes:

Dataset size

RDKit, FAISS versions

Jaccard overlaps

AUC statistics

Scaffold-split results

Speed metrics

6.2 Machine-Readable Core Results

benchmark_metrics.csv
Contains row-level results for:

Query index

ECFP AUC

SimSon AUC

Jaccard overlaps

Scaffold-split metrics (if applicable)

6.3 Auxiliary Analyses

Saved as NumPy arrays:

File	Description
ecfp_similarity.npy	ECFP4 similarity matrix or sample
simson_cosine.npy	SimSon cosine matrix or sample
tanimoto_sample.npy	Sampled Tanimoto similarities
cosine_sample.npy	Sampled cosine similarities
jaccard_topk.npy	Query-by-query Jaccard overlaps
umap_projection.npy	2D embedding coordinates (if computed)

These support downstream statistical workflows or publication-quality figures.

7. Interpretation of Results

High Jaccard overlap → SimSon behaves similarly to ECFP4.

Higher enrichment AUC → SimSon better captures bioactivity locality.

Better scaffold-split results → SimSon generalizes beyond structural families.

Similarity distributions reveal sparsity vs. smoothness in feature spaces.

FAISS performance demonstrates suitability of embeddings for large-scale search.

8. Limitations

Evaluation is limited to BACE; future work should incorporate:

ChEMBL subsets

DUD-E

MUV

Large internal screening sets

AUC is a simple metric; EF1%, BEDROC, and Boltzmann-weighted measures may be more appropriate for virtual screening.

SimSon is treated as a fixed embedding model; quality depends on pretraining.

9. Reproducibility

All stochastic processes use fixed seeds

RDKit/FAISS/scikit-learn versions are logged

All exported arrays and CSV files allow reconstruction of all comparisons

10. Conclusion

This benchmark provides a comprehensive and reproducible comparison between SimSon embeddings and classical cheminformatics similarity measures.
The analyses include retrieval, enrichment, scaffold-based generalization, distributional structure, and indexing performance, and produce both practitioner-readable summaries and machine-readable outputs.

The resulting files can be integrated into screening pipelines or used for subsequent statistical evaluation, extension to larger datasets, or publication.