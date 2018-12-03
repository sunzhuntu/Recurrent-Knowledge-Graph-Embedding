# Recurrent-Knowledge-Graph-Embedding
This is the code of a knowledge graph embedding framework – RKGE – with a novel recurrent network architecture for high-quality recommendation. RKGE not only learns the semantic representation of different types of entities but also automatically captures entity relations encoded in KGs.

#### Pre-requisits

- We adopt two real-world datasets - MovieLens and Yelp. For the MoiveLens dataset, we crawl the corresponding IMDB dataset as movie auxiliary information, including genre, director, and actor. Note that we automatically remove the movies without auxilairy information. We then combined MovieLens and IMDB by movie-title and released year. The combined data is save in a txt file (ml-movie-auxiliary-info.txt) and the format is as follows:

```
id:1|genre:Animation,Adventure,Comedy|director:John Lasseter|actors:Tom Hanks,Tim Allen,Don Rickles,Jim Varney
```

- For the original user-movie rating data, we remove all items without auxiliary information. The data is save in a txt file (rating-delete-missing-itemid.txt) and the format is as follows:

```
userid  itemid  rating  timestamp
```


