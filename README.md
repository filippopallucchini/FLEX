# FLEX
## Self-explanatory and Retrieval-augmented LLMs for Financial Sentiment Analysis

FLEX (Financial Language Enhancement with Guided LLM Execution), is an automated system capable of retrieving information from a Large Language Model (LLM) to enrich financial sentences, making them more knowledge-dense and explicit. FLEX generates multiple potentially enhanced sentences and uses a new logic to determine the most suitable one. Since LLMs may introduce hallucinated answers, we have significantly reduced this risk by developing a new algorithm that selects the most appropriate sentences. This approach ensures that the meaning of the original sentence is preserved, avoids excessive syntactic similarity between versions, and achieves the lowest possible perplexity. These enhanced sentences are more interpretable and directly useful for downstream tasks like financial sentiment analysis (FSA).

# FLEX FRAMEWORK

<img width="948" alt="Diagram_new" src="https://github.com/user-attachments/assets/ed717d4a-66bd-42cd-8a01-0f4f752de620">

The framework of our proposed model, as sketched in Figure above, consists of two main phases. 
- **MAKEUP phase** - This phase generates enriched sentence candidates that maintain the exact semantics of the original text while clarifying specific financial concepts or making implicit propositions more explicit.
- **MAKEUP SELECTION phase** -  This phase consists of a function that selects the most appropriate candidate from those produced in the previous phase. Using an embedding model, the function encodes both the original sentence and each candidate into vector representations to assess semantic similarity. In addition to semantic similarity, we also consider it crucial to ensure the sentence is clear and self-explanatory. Thus, the function also incorporates a measure of perplexity, which reflects how well the sentence aligns with the language model's expectations, providing insight into its readability and naturalness.

## **Deployment**

Use Python 3.10.12

Example with Financial PhraseBank data

1. Run FPB/PREPROCESS/compute_subsentences_of_fpb_densex.py to compute propositions for the input dataset
2. Run FPB/main.py to perform the enrichment

After creating the enriched dataset, you can reproduce the ablation study by running the scripts located in the ABLATION folder. 
Additionally, you can replicate the evaluation experiments described in the original published article.

## References
```
@article{pallucchini2025self,
  title={Self-explanatory and Retrieval-augmented LLMs for Financial Sentiment Analysis},
  author={Pallucchini, Filippo, and Zhang, Xulang and Mao, Rui and Cambri, Erik},
  journal={The 40th ACM/SIGAPP Symposium On Applied Computing},
  year={2025},
  publisher={ACM/SIGAPP}
}
```

