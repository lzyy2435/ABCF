# AECF: An Adaptive Balanced Multimodal URL Classification Framework
![image](https://github.com/lzyy2435/AECF/assets/70757777/3ae00887-fee3-46b1-935a-92f4f7f33dc6)
## Abstract
Websites are a crucial medium for conveying multimedia information. However, malicious websites can lurk among them, posing a threat to the security of users' sensitive data and personal privacy. It is imperative to detect and classify these harmful websites. Various solutions have been proposed by researchers for this task, including extracting features from URLs and analyzing web page content using classification algorithms. However, little attention has been paid to the short lifespan and frequent updates characteristics of malicious websites. Existing methods often use single feature inputs and static models to process website data, regardless of the dynamic feature of multimedia scenarios or the Out-of-Distribution (OOD) problem of malicious websites in real-world scenarios. To address these problems, we propose the an adaptive balanced multimodal URL classification framework (AECF), a classification system for web page classification tasks. This includes a FAModule and FEAModule to effectively combine the various data modalities of the website and mitigate the impact of the OOD problem. We also employ incremental learning to continuously improve our model and adapt to the changing network environment. Additionally, we have designed a comprehensive architecture for storage-compute separation and employ a distributed database to ensure algorithm deployment. Experiments on three website datasets confirm the effectiveness of our modules in capturing website features, and demonstrate the superiority of our model compared to past malicious website classification systems. Simultaneously, a pragmatic data storage architecture, combined with a two-tier website filtering and categorization approach, renders our structure both efficient and effective.
## Code
The source code will be released after review.
