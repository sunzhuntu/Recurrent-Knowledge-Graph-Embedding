# Recurrent-Knowledge-Graph-Embedding
This is the code of a knowledge graph embedding framework – RKGE – with a novel recurrent network architecture for high-quality recommendation. RKGE [1] not only learns the semantic representation of different types of entities but also automatically captures entity relations encoded in KGs.


## Pre-requisits

- ### Running environment

  - Python 3
  
  - Pytorch (conda 4.5.11 - https://zhuanlan.zhihu.com/p/26854386)
  

- ### Datasets - MovieLens and Yelp. 

  - For the MoiveLens dataset, we crawl the corresponding IMDB dataset as movie auxiliary information, including genre, director, and actor. Note that we automatically remove the movies without auxilairy information. We then combined MovieLens and IMDB by movie title and released year. The combined data is save in a txt file (auxiliary.txt) and the format is as follows:

    ```
    id:1|genre:Animation,Adventure,Comedy|director:John Lasseter|actors:Tom Hanks,Tim Allen,Don Rickles,Jim Varney
    ```

  - For the original user-movie rating data, we remove all items without auxiliary information. The data is save in a txt file (rating-delete-missing-itemid.txt) and the format is as follows:

    ```
    userid itemid rating timestamp
    ```

   - For the Yelp dataset, we use the originally provided genre and city information of the locations. The format is as follows:

      ```
      id:11163|genre:Accountants,Professional Services,Tax Services,Financial Services|city:Peoria
      ```

## Modules of RKGE

  - For clarify, hereafter we use movielen dataset as a toy example to demonstrate the detailed modules of RKGE. For yelp dataset, you need to do some adaptation for the code and tune some parameters (e.g., input_dim=20, hidden_dim=32, learning_rate=0.01). 

  - ### Auxiliary Info Mapping (auxiliary-mapping.py)
    
    - Map the auxiliary infomation into ID
    
      - Input Data: auxiliary.txt
    
      - Output Data: auxilary-mapping.txt
    

  - ### Data Split (data-split.py)
  
    - Split the user-movie rating data into training and test data

      - Input Data: rating-delete-missing-itemid.txt

      - Output Data: training.txt, test.txt


  - ### Negative Sample (negative-sample.py)
  
    - Sample negative movies for each user to balance the model training 
  
      - Input Data: training.txt

      - Output Data: negative.txt


  - ### Path Extraction （path-extraction.py）
  
    - Extract paths for both positive and negative user-moive interaction
    
      - Input Data: training.txt, negative.txt, auxiliary-mapping.txt,
      
      - Output Data: positive-path.txt, negative-path.txt
      
  
  - ### Recurrent Neural Network (recurrent-neural-network.py)
  
    - Feed both postive and negative path into the recurrent neural network, train and evaluate the model
    
      - Input Data: positive-path.txt, negative-path.txt, training.txt, test.txt, pre-train-user-embedding.txt, pre-train-movie-embedding.txt (To speed up model training process, the user and movie embedding is pre-trained via [2]. You may also use matrix factorization [3] or bayesain personalized ranking [4] to pre-train the embeddings). 
      
      - Output Data: results.txt
      
      
  - ### References
    
    [1] Sun Zhu, Jie Yang et al. [Recurrent knowledge graph embedding for effective recommendation](http://sunzhuntu.wixsite.com/summer). ACM RecSys, 2018.               
        
       
    [2] Grbovic Mihajlo, et al. [E-commerce in your inbox: product recommendations at scale](https://arxiv.org/pdf/1606.07154.pdf). KDD, 2015
    
    
    [3] Koren, Yehuda, Robert Bell, et al. [Matrix factorization techniques for recommender systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf). Computer, 2009
    
    [4] Rendle, Steffen, et al. [BPR: bayesian personalized ranking from implicit feedback](https://arxiv.org/pdf/1205.2618.pdf). UAI, 2009.
