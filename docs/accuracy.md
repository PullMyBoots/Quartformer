# Accuracy Benchmarks on Simulated Datasets

This document presents comprehensive accuracy evaluation of QuartFormer on simulated phylogenetic datasets, comparing against state-of-the-art methods including ASTER, FastTree, and IQ-TREE.

## Test Environment

### Simulation Parameters
- **Simulation Software**: SimPhy / RAXML-NG
- **Evolutionary Model**: GTR (General Time Reversible)
- **Tree Generation**: Yule process / birth-death process
- **Sequence Lengths**: 100k, 1M, 10M sites
- **Taxon Counts**: 24, 48, 96, 192 species
- **Replicates**: X replicates per configuration
- **Missing Data**: [Optional: rate of missing data]

### Evaluation Metrics
- **Robinson-Foulds (RF) Distance**: Normalized RF distance (0-1, lower is better)
- **Quartet Distance**: Quartet concordance (0-1, higher is better)
- **Branch Score Distance (BSD)**: [Optional]

---

## Accuracy Results

### 100k Sequence Length

| Taxa | QuartFormer | ASTER | FastTree | IQ-TREE |
|------|-------------|-------|----------|---------|
| 24   | TBD         | TBD   | TBD      | TBD     |
| 48   | TBD         | TBD   | TBD      | TBD     |
| 96   | TBD         | TBD   | TBD      | TBD     |
| 192  | TBD         | TBD   | TBD      | TBD     |

### 1M Sequence Length

| Taxa | QuartFormer | ASTER | FastTree | IQ-TREE |
|------|-------------|-------|----------|---------|
| 24   | TBD         | TBD   | TBD      | TBD     |
| 48   | TBD         | TBD   | TBD      | TBD     |
| 96   | TBD         | TBD   | TBD      | TBD     |
| 192  | TBD         | TBD   | TBD      | TBD     |

### 10M Sequence Length

| Taxa | QuartFormer | ASTER | FastTree | IQ-TREE |
|------|-------------|-------|----------|---------|
| 24   | TBD         | TBD   | TBD      | TBD     |
| 48   | TBD         | TBD   | TBD      | TBD     |
| 96   | TBD         | TBD   | TBD      | TBD     |
| 192  | TBD         | TBD   | TBD      | TBD     |

---

## Analysis by Taxon Count

### Small Datasets (24 taxa)

[Description of accuracy patterns, advantages/disadvantages]

### Medium Datasets (48-96 taxa)

[Description of accuracy patterns, advantages/disadvantages]

### Large Datasets (192+ taxa)

[Description of accuracy patterns, advantages/disadvantages]

---

## Analysis by Sequence Length

### Short Alignments (100k sites)

[Discussion of accuracy with limited sequence data]

### Medium Alignments (1M sites)

[Discussion of accuracy with moderate sequence data]

### Long Alignments (10M sites)

[Discussion of accuracy with extensive sequence data]

---

## Comparison with Competing Methods

### vs ASTER

[Accuracy comparison, strengths and weaknesses]

### vs FastTree

[Accuracy comparison, strengths and weaknesses]

### vs IQ-TREE

[Accuracy comparison, strengths and weaknesses]

---

## Key Findings

1. **Overall Accuracy**: [Summary statement]

2. **Scalability**: [How accuracy scales with taxa/sequence length]

3. **Competitive Performance**: [Compared to other methods]

4. **Best Use Cases**: [When QuartFormer performs best]

---

## Accuracy vs Speed Trade-off

[Combined analysis of accuracy benchmarks.md with speed results]

### Performance-Accuracy Pareto Front

[Visualization or table showing which method offers best balance]

---

## Statistical Significance

[Optional: Statistical tests comparing methods]

---

## Conclusion

[Summary of accuracy performance and recommendations]



-----


下面我们从一些研究文献中提取数据，然后使用他们的数据集，将他们提供的所有的基因串联拼接起来，缺失物种的直接补充“-”，成为一个超级矩阵，然后给我们的模型进行推断。上面我们对比了他们文献所给出的参考数据和我们推断的结果，并且注意，推断结果仅供参考，就是推断结果，他注意参考一下就行了吧。也就是说，因为不同的推断方法以及不同的数据处理的方式，可能会对推断结果造成很大的出入。

数据集一：
Phylogenomic Analysis of Wolbachia Strains Reveals Patterns of Genome Evolution and Recombination



数据集二：Insecta
Phylogenomics of the major lineages of Bembidion and related ground beetles (Coleoptera: Carabidae: Bembidiini)


数据集三：Oakleaf butterfly dataset
《The evolution and diversification of oakleaf butterflies.》


数据集四：plant1
《Highly resolved papilionoid legume phylogeny based on plastid phylogenomics》


数据集五：fish
Phylogenomic Systematics of Ostariophysan Fishes: Ultraconserved Elements Support the Surprising Non-Monophyly of Characiformes


数据集六：
哺乳动物1：82
Genomic evidence reveals a radiation of placental mammals uninterrupted by the KPg boundary

数据集七：
哺乳动物2：
Ultraconserved elements improve the resolution of difficult nodes within the rapid radiation of neotropical sigmodontine rodents


数据集八：反刍
Large-scale ruminant genome sequencing provides insights into their evolution and distinct traits
