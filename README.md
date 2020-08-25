# AV-Janatahack-Independence-Day-2020-ML-Hackathon
<img src="https://img.shields.io/badge/Hugging%20face-Transformers-orange"> <img src="https://img.shields.io/badge/Python-3.7-blue"> <img src="https://img.shields.io/badge/torch-1.5.1-red"><br>
Code implemented during the hackathon

<p>
  <h2>Problem Statement :</h2>
  <h3>Topic Modeling for Research Articles</h3>
  <p>Researchers have access to large online archives of scientific articles. As a consequence, finding relevant articles has become more difficult. Tagging or topic modelling     provides a way to give token of identification to research articles which facilitates recommendation and search process. <br>
Given the abstract and title for a set of research articles, predict the topics for each article included in the test set. 
Note that a research article can possibly have more than 1 topic. The research article abstracts and titles are sourced from the following 6 topics: 
<ul>
  <li>Computer Science</li>
  <li>Physics</li>
  <li>Mathematics</li>
  <li>Statistics</li>
  <li>Quantitative Biology</li>
  <li>Quantitative Finance</li>
</ul>
</p>
<h2>Data :</h2>
<ul>
  <li><a href="https://datahack.analyticsvidhya.com/contest/janatahack-independence-day-2020-ml-hackathon/download/train-file">Train data</a></li>
  <li><a href="https://datahack.analyticsvidhya.com/contest/janatahack-independence-day-2020-ml-hackathon/download/test-file">Test data</a></li>
</ul>
<p>
  <h2>Models :</h2>
  <ul>
  <li>
    <h3>BERT</h3>
    <p>
      A language representation model, which stands for Bidirectional Encoder Representations from Transformers. BERT is a multi-layer bidirectional Transformer's encoder stack.
      <ul>
        <li>
          <h4>Architectures for multi-label classification:</h4>
          <ol>
            <li>Pooled outputs + Classification Layer</li>
            <li>Sequence outputs + Spatial dropout + Mean & Max pooling + Classification layer</li> 
          </ol>
        </li>
        <li>
          <h4>Code & Notebooks</h4>
          <ol>
          <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/bert-base">bert-base-uncased for multilabel classification</a></li>
          </ol>
        </li>
      </ul>
    </p>
  </li>
  <li>
    <h3>RoBERTa</h3>
     <p>
       RoBERTa: A Robustly Optimized BERT Pretraining Approach. It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.
     </p>
     <ul>
       <li>
         <h4>Architectures for multi-label classification:</h4>
          <ol>
            <li>Pooled outputs(roberta-base) + Classification Layer</li>
            <li>Pooled outputs(roberta-large) + Classification Layer</li>
            <li>Dual input + Single head + Concatenation + Classification Layer</li> 
          </ol>
       </li>
       <li>
         <h4>Code & Notebooks</h4>
         <ol>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/roberta-base">roberta-base for multi-label classification</a></li>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/roberta-large">roberta-large for multi-label classification</a></li>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/roberta-dual-input">Dual input roberta-base for multi-label classification</a></li>
         </ol>
       </li>
     </ul>
  </li>
  
  <li>
    <h3>ALBERT</h3>
     <p>
       ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. ALBERT uses repeating layers which results in a small memory footprint, however the computational cost remains similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same number of (repeating) layers.
     </p>
     <ul>
       <li>
         <h4>Architectures for multi-label classification:</h4>
          <ol>
            <li>Pooled outputs(albert-base-v2) + Classification Layer</li>
          </ol>
       </li>
       <li>
         <h4>Code & Notebooks</h4>
         <ol>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/albert-base">albert-base for multi-label classification</a></li>
         </ol>
       </li>
     </ul>
  </li>
  
  <li>
    <h3>Longformer</h3>
     <p>
       Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, the Longformer was introduced with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer.
     </p>
     <ul>
       <li>
         <h4>Architectures for multi-label classification:</h4>
          <ol>
            <li>Pooled outputs(allenai/longformer-base-4096) + Classification Layer</li>
          </ol>
       </li>
       <li>
         <h4>Code & Notebooks</h4>
         <ol>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/longformer">longformer-base for multi-label classification</a></li>
         </ol>
       </li>
     </ul>
  </li>
  
  <li>
    <h3>T5</h3>
     <p>
T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format. T5 works well on a variety of tasks out-of-the-box by prepending a different prefix to the input corresponding to each task, e.g.: for translation: translate English to German, etc.
     </p>
     <ul>
       <li>
         <h4>Architectures for multi-label classification:</h4>
          <ol>
            <li>Complete Text-to-Text Transformer(encoder stack + decoder stack)</li>
          </ol>
       </li>
       <li>
         <h4>Code & Notebooks</h4>
         <ol>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/blob/master/t5-base">T5 for multi-label classification</a></li>
         </ol>
       </li>
     </ul>
  </li>
  
   <li>
    <h3>XLNet</h3>
     <p>
XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization order.
     </p>
     <ul>
       <li>
         <h4>Architectures for multi-label classification:</h4>
          <ol>
            <li>Pooled outputs +Classification layer</li>
          </ol>
       </li>
       <li>
         <h4>Code & Notebooks</h4>
         <ol>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/xlnet-base">xlnet for multi-label classification</a></li>
         </ol>
       </li>
     </ul>
  </li>
  
  <li>
    <h3>SciBERT</h3>
     <p>
It is a BERT model trained on scientific text. SciBERT is trained on papers from the corpus of semanticscholar.org. Corpus size is 1.14M papers, 3.1B tokens
     </p>
     <ul>
       <li>
         <h4>Architectures for multi-label classification:</h4>
          <ol>
            <li>Pooled outputs + Classification layer</li>
            <li>Sequence outputs + Spatial dropout + BiLstm + Classification layer</li>
            <li>Siamese like architecture: Dual inputs(single head) + Pooled outputs + Avg pooling + Concatenation + Classification layer</li>
            <li>Siamese like architecture: Dual inputs(single head) + Sequence outputs + Bi-GRU + Classification layer</li>
            <li>Dual inputs(dual head) + Sequence outputs + Avg pooling + Concatenation + Classification layer</li>
            <li>Scibert embeddings + XGBoost</li>
            <li>Scibert embeddings + LGBM</li>
            <li>Scibert + XLNet</li>
          </ol>
       </li>
       <li>
         <h4>Code & Notebooks</h4>
         <ol>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/scibert-bilstm">scibert-bilstm for multi-label classification</a></li>
          <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/scibert-dual-input">scibert-dual inputs for multi-label classification</a></li>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/scibert">scibert-base for multi-label classification</a></li>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/scibert-gradient-boosting">scibert-gradient boosting for multi-label classification</a></li>
           <li><a href="https://github.com/shanayghag/AV-Janatahack-Independence-Day-2020-ML-Hackathon/tree/master/scibert-xlnet-ensemble">scibert-xlnet ensemble for multi-label classification</a></li>
         </ol>
       </li>
     </ul>
  </li>
  </ul>
</p>
</p>
