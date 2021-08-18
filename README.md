<h1 align="center"> Awesome EHR Graph AI </h1>

A collaborative list of resources for making Electronic Health Records AI-friendly
 

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>
<a href='#' id='top'></a>
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-list"> ➤ About The List</a></li>
    <li><a href="#tools"> ➤ Tools</a></li>
    <li><a href="#umls"> ➤ Unified Medical Language System</a></li>
    <li><a href="#nlp"> ➤ Natural Language Processing</a></li>
    <li><a href="#graphs"> ➤ Graph Databases </a></li>
    <li><a href="#learning"> ➤ Learning </a></li>
    <li><a href="#transformers"> ➤ Transformers </a></li>
    <li><a href="#research"> ➤ Academic Research </a></li> 
  </ol>
</details>

<hr>

### About the list 
<a href='#' id='about-the-list'>Top</a>

This list is a collaborative effort initiated by the [Medical Intelligence Society's Graph Working Group](https://mis-graph-ai.github.io). In the course of meeting its objective, the group's members came across useful resources that they wished to share with the community, hence this curated list. Some of the items may appear in multiple categorizations<br>
If you have something that is related to the application of graphs and / or artificial intelligence on electronic health records, please send us a pull request.

### Tools
<a href='#' id='tools'>Top</a>

- [MetaMap](https://metamap.nlm.nih.gov/) - A tool for recognising UMLS concepts in text
- [Apache cTAKES](https://ctakes.apache.org/) - clinical Text Analysis Knowledge Extraction System: natural language processing system for extraction of information from electronic medical record clinical free-text
- [SpaCy](https://allenai.github.io/scispacy/) - SpaCy models for biomedical text processing: scispaCy is a Python package containing spaCy models for processing biomedical, scientific or clinical text.
- [SemEHR](https://pubmed.ncbi.nlm.nih.gov/29361077/) - A general-purpose semantic search system to surface semantic data from clinical notes for tailored care, trial recruitment, and clinical research
- [Transformers](https://huggingface.co/transformers/) - State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0
- [WISER](https://github.com/BatsResearch/wiser) - <details>
  <summary> a system for training sequence tagging models, particularly neural networks for named entity recognition </summary> WISER (_Weak and Indirect Supervision for Entity Recognition_), a system for training sequence tagging models, particularly neural networks for named entity recognition (NER) and related tasks. WISER uses weak supervision in the form of rules to train these models, as opposed to hand-labeled training data. </details>
- [Snorkel](https://www.snorkel.org/get-started/) - Snorkel is a system for _programmatically_ building and managing training datasets **without manual labeling**. In Snorkel, users can develop large training datasets in hours or days rather than hand-labeling them over weeks or months.

### Unified Medical Language System
<a href='#' id='umls'>Top</a>

- [MetaMap](https://metamap.nlm.nih.gov/) - A tool for recognising UMLS concepts in text
- [QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS) - QuickUMLS (Soldaini and Goharian, 2016) is a tool for fast, unsupervised biomedical concept extraction from medical text.
- [SemRep](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3517-7) - In this paper, we present an in-depth description of SemRep, an NLP system that extracts semantic relations from PubMed abstracts using linguistic principles and UMLS domain knowledge.

### Natural Language Processing
<a href='#' id='nlp'>Top</a>

- [MetaMap](https://metamap.nlm.nih.gov/) - A tool for recognising UMLS concepts in text
- [Apache cTAKES](https://ctakes.apache.org/) - clinical Text Analysis Knowledge Extraction System: natural language processing system for extraction of information from electronic medical record clinical free-text
- [12 open source tools for natural language learning](https://opensource.com/article/19/3/natural-language-processing-tools)
- [MedSpaCy](https://github.com/medspacy/medspacy) - library of tools for performing clinical NLP and text processing tasks with the popular spaCy framework
- [Treatment Relation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4419965/) - <details>
  <summary>a supervised learning system that is able to predict whether or not a treatment relation exists between any two medical concepts</summary>
  We describe a supervised learning system that is able to predict whether or not a treatment relation exists between any two medical concepts mentioned in clinical notes. Our approach for identifying treatment relations in clinical text is based on (a) the idea of exploring the contextual information in which medical concepts are described and (b) the idea of using predefined medication-indication pairs.</details>
- [BERT-CRel](https://arxiv.org/abs/2012.11808) - <details>
  <summary> a transformer model for fine-tuning biomedical word embeddings that are jointly learned along with concept embeddings </summary> BERT-CRel is a transformer model for fine-tuning biomedical word embeddings that are jointly learned along with concept embeddings using a pre-training phase with fastText and a fine-tuning phase with a transformer setup. The goal is to provide high quality pre-trained biomedical embeddings that can be used in any downstream task by the research community. This repository contains the code used to implement the BERT-CRel methods and generate the embeddings. The corpus used for BERT-CRel contains biomedical citations from PubMed and the concepts are from the Medical Subject Headings (MeSH codes) terminology used to index citations. </details>
- [UMLS-Bert](https://arxiv.org/abs/2010.10391) - <details>
  <summary>a contextual embedding model that integrates domain knowledge during the pre-training process</summary>
  Contextual word embedding models, such as BioBERT and Bio_ClinicalBERT, have achieved state-of-the-art results in biomedical natural language processing tasks by focusing their pre-training process on domain-specific corpora. However, such models do not take into consideration expert domain knowledge. In this work, we introduced UmlsBERT, a contextual embedding model that integrates domain knowledge during the pre-training process via a novel knowledge augmentation strategy.</details>
- [Publicly Available Clinical BERT Embeddings](https://www.aclweb.org/anthology/W19-1909/) - <details>
  <summary>a domain-specific language representation model pre-trained on large-scale biomedical corpora</summary>
  We introduce BioBERT (Bidirectional Encoder Representations from Transformers for Biomedical Text Mining), which is a domain-specific language representation model pre-trained on large-scale biomedical corpora. With almost the same architecture across tasks, BioBERT largely outperforms BERT and previous state-of-the-art models in a variety of biomedical text mining tasks when pre-trained on biomedical corpora. While BERT obtains performance comparable to that of previous state-of-the-art models, BioBERT significantly outperforms them on the following three representative biomedical text mining tasks: biomedical named entity recognition (0.62% F1 score improvement), biomedical relation extraction (2.80% F1 score improvement) and biomedical question answering (12.24% MRR improvement). Our analysis results show that pre-training BERT on biomedical corpora helps it to understand complex biomedical texts.</details>
- [Enhanced Clinical BERT Embedding Using Biological Knowledge Base](https://www.aclweb.org/anthology/2020.coling-main.57.pdf) - <details>
  <summary> a novel training method is introduced for adding knowledge base information from UMLS into language model pre-training</summary>Domain knowledge is important for building Natural Language Processing (NLP) systems forlow-resource settings, such as in the clinical domain. In this paper, a novel joint training method is introduced for adding knowledge base information from the Unified Medical Language Sys-tem (UMLS) into language model pre-training for some clinical domain corpus. We show that in three different downstream clinical NLP tasks, our pre-trained language model outperformsthe corresponding model with no knowledge base information and other state-of-the-art mod-els.</details>
- [SciFive](https://arxiv.org/abs/2106.03598) - <details>
  <summary> a domain-specific T5 model that has been pre-trained on large biomedical corpora</summary>
  In this report, we introduce SciFive, a domain-specific T5 model that has been pre-trained on large biomedical corpora. Our model outperforms the current SOTA methods (i.e. BERT, BioBERT, Base T5) on tasks in named entity relation, relation extraction, natural language inference, and question-answering. We show that text-generation methods have significant potential in a broad array of biomedical NLP tasks, particularly those requiring longer, more complex outputs. Our results support the exploration of more difficult text generation tasks and the development of new methods in this area </details>
- [Survey Paper Comparing Various Models](https://arxiv.org/abs/2105.00827) - <details>
  <summary>a survey paper that can provide a comprehensive survey of various transformer-based biomedical pretrained language models</summary>
  Transformer-based pretrained language models (PLMs) have started a new era in modern natural language processing (NLP). These models combine the power of transformers, transfer learning, and self-supervised learning (SSL). Following the success of these models in the general domain, the biomedical research community has developed various in-domain PLMs starting from BioBERT to the latest BioMegatron and CoderBERT models. We strongly believe there is a need for a survey paper that can provide a comprehensive survey of various transformer-based biomedical pretrained language models (BPLMs).  
  </details>
- [Hybrid Approach to Measure Semantic Relatedness in Biomedical Concepts](https://arxiv.org/abs/2101.10196) - <details>
  <summary>the effectiveness of a hybrid approach based on Sentence BERT model and retrofitting algorithm to compute relatedness between any two biomedical concepts</summary>
  Objective: This work aimed to demonstrate the effectiveness of a hybrid approach based on Sentence BERT model and retrofitting algorithm to compute relatedness between any two biomedical concepts. Materials and Methods: We generated concept vectors by encoding concept preferred terms using ELMo, BERT, and Sentence BERT models. We used BioELMo and Clinical ELMo. We used Ontology Knowledge Free (OKF) models like PubMedBERT, BioBERT, BioClinicalBERT, and Ontology Knowledge Injected (OKI) models like SapBERT, CoderBERT, KbBERT, and UmlsBERT. We trained all the BERT models using Siamese network on SNLI and STSb datasets to allow the models to learn more semantic information at the phrase or sentence level so that they can represent multi-word concepts better. Finally, to inject ontology relationship knowledge into concept vectors, we used retrofitting algorithm and concepts from various UMLS relationships.
  </details>

### Graph Databases
<a href='#' id='graphs'>Top</a>

- [GraphDB](https://www.ontotext.com/blog/graphdb-semantic-text-similarity-for-identifying-related-terms-documents/) - Semantic Text Similarity for Identifying Related Terms & Documents

### Learning
<a href='#' id='learning'>Top</a>

- [Guide to comprehensive adult H&P](https://med.ucf.edu/media/2018/08/Guide-to-the-Comprehensive-Adult-H-and-P-Write-up-2017-18.pdf)
- [12 open source tools for natural language learning](https://opensource.com/article/19/3/natural-language-processing-tools)
- [GraphDB](https://www.ontotext.com/blog/graphdb-semantic-text-similarity-for-identifying-related-terms-documents/) - Semantic Text Similarity for Identifying Related Terms & Documents
- [NLP + Graph Pipeline](https://towardsdatascience.com/nlp-and-graphs-go-hand-in-hand-with-neo4j-and-apoc-e57f59f46845) - NLP goes hand in hand with graphs: Learn how to set up an NLP pipeline and analyze its results with Neo4j
- [Tabular Data + Transformers](https://medium.com/georgian-impact-blog/how-to-incorporate-tabular-data-with-huggingface-transformers-b70ac45fcfb4) - How to Incorporate Tabular Data with HuggingFace Transformers
- 

### Transformers
<a href='#' id='transformers'>Top</a>

### Academic Research
<a href='#' id='research'>Top</a>

