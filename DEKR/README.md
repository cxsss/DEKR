# DEKR
This repository is the implementation of [DEKR](https://dl.acm.org/doi/10.1145/3404835.3462900):
> DEKR: Description Enhanced Knowledge Graph for Machine Learning Method Recommendation

> The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR2021)

DEKR is a description-enhanced machine learning (ML) knowledge graph-based approach to help
recommend appropriate ML methods for given ML datasets.

## Environment Requirement
The code has been tested running under Python 3.6.12. The required packages are as follows:
* torch == 1.6.0
* numpy == 1.19.1
* pandas == 1.1.3
* sklearn == 0.23.2

### Files in the folder

- `data/MachineLearning/`

    - `dm_ratings.csv` 
      - Interaction data for machine learning (ML) datasets and methods;
      - Each row is a `dataset ID` and its interacted `method ID` with an interaction `label` of 1, indicating a positive sample.
    
    - `entity_desc_embed_vector.npy`
      - Embedding representation of the `descriptive text` for the ML dataset or method, which is obtained from the GloVe pre-trained word vector.
    
    - `item_index2entity_id_rehashed.txt`
      - Mapping from `method ID` in raw interaction data to `entity ID` in machine learning knowledge graphs；
      - Since the current method is extracted directly from the knowledge graph, the two are equal at present.
    
    - `mlkg_rehashed.txt`
       - Machine Learning Knowledge Graph (MLKG) Data;
       - Each row is a triplet in the format `(h, r, t)`, denoting the existence of a relationship `r` between the head entity `h` and the tail entity `t`.
       - These knowledge graph data are triples that are retained after data cleaning and pre-processing to provide effective auxiliary information to the datasets and method entities in the interaction data. Specifically, low-frequency entities and relations are filtered out in order to reduce the noise in the knowledge graph as much as possible.
> Note: Due to the privacy policy of the unconcluded National Key Research and Development Plan project in which this data is located, this data only provides the portion that satisfies the reproduction requirements, which can verify the feasibility of the method recommendation based on the knowledge graph as well as the effectiveness of the proposed model.


- `src/`
    
    - Implementations of DEKR.

### Running the code
- The setup and details of the implementation are already stated clearly in the code.
  
  - ```
    $ cd src
    $ python main.py
    ```
## Acknowledgement
Any scientific publications that use our datasets or codes should cite the following paper as the reference:
```
@article{Cao2021DEKRDE,
  title={DEKR: Description Enhanced Knowledge Graph for Machine Learning Method Recommendation},
  author={Xianshuai Cao and Yuliang Shi and Han Yu and Jihu Wang and Xinjun Wang and Zhongmin Yan and Zhiyong Chen},
  journal={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```

Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:
* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.
