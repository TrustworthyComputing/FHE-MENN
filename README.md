# FHE-MENNs: Opportunities and Pitfalls for Accelerating Fully Homomorphic Private Inference with Multi-Exit Neural Networks

This work evaluates the use of multi-exit neural networks (MENNs) for private inference. Fully homomorphic encryption (FHE) is a technology that lets you perform computations, including neural networks, on encrypted data. Using this technology, users can encypt their data and send it to a machine learning cloud service for processing. The cloud is unable to decrypt the data, protecting user privacy, but can perform the computation and decrypted the result. Using MENNs allows the cloud to reduce the computational cost by giving the user the option to terminate the computation early if a neural network early exit is confident. However, using MENNs will leak one bit of information per exit as the user will decide whether to terminate the computation early (True/False). 


The first part of this repository provides several MENN examples for image classification, including training and evaluation scripts.([Code](https://github.com/TrustworthyComputing/FHE-MENN/MENNs))

The second part of this repository analyzes *what* user data can be extracted from the limited information provided from the user decison ([Code](https://github.com/TrustworthyComputing/FHE-MENN/tree/main/TorMENNt)).

## Cite this work
This work was introduced in the FHE-MENN ([Paper](https://eprint.iacr.org/2024/1099)). The work can be cited as:
```
@misc{folkerts2023FheMenn,
      author = {Lars Wolfgang Folkerts and Nektarios Georgios Tsoutsos},
      title = {{FHE}-{MENNs}: Opportunities and Pitfalls for Accelerating Fully Homomorphic Private Inference with Multi-Exit Neural Networks},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1099},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1099}
}
```

## Acknowledgments
This work was supported by the National Science Foundation (Award #2239334).
