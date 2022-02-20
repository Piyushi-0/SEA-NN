# Improving Attribution Methods by Learning Submodular Functions 
### Accepted at the 25th International Conference on Artificial Intelligence and Statistics ([AISTATS '22](https://aistats.org/aistats2022/cfp.html)). 
#### [Piyushi Manupriya](https://piyushi-0.github.io), [Tarun Ram Menta](https://github.com/peppermenta), [J. Saketha Nath](https://www.iith.ac.in/~saketha/), [Vineeth N Balasubramanian](https://www.iith.ac.in/~vineethnb/index.html)

[Code in PyTorch](https://github.com/Piyushi-0/SEA-NN) | [Link to arxiv pre-print](https://arxiv.org/pdf/2104.09073.pdf) | [Link to the version accepted at ICML '20 workshops (XXAI, WHI)](http://interpretable-ml.org/icml2020workshop/pdf/29.pdf)

<img src="docs/SEA-NN.jpg">

*This work explores the novel idea of learning a submodular scoring function to **improve the specificity** of existing feature attribution methods. A **novel formulation for  learning a deep submodular set function** that is consistent with the real-valued attribution maps obtained by existing attribution methods is proposed. The final attribution    value of a feature is then defined as the marginal gain in the induced submodular score of the feature in the context of other highly attributed features, thus decreasing the  attribution of redundant yet discriminatory features. Experiments on multiple datasets illustrate that the proposed attribution method **achieves higher specificity along with   good discriminative power**.*

#### Acknowledgements
The first author is especially grateful to the mentorship of [Dr Bilal Alsallakh](https://scholar.google.com/citations?user=0TZaxxwAAAAJ&hl=en&oi=ao) and for being supported by the Google PhD Fellowship.
