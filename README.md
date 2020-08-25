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
          <h4>Architecture for multi-label classification:</h4>
          <ol>
            <li>Pooled outputs + Classification Layer</li>
            <li>Sequence outputs + Spatial dropout + Mean & Max pooling + Classification layer</li> 
          </ol>
        </li>
      <ul>
    </p>
  </li>
  </ul>
</p>
</p>
