This is the project of the chapter Clustering of the course "Machine Learning for graphs and sequential data" suggested by the technical university of Munich
The goal of this task is to find groups of users with similar preferences using Spectral clustering. You are given a fragment of the Yelp social network, represented by an undirected weighted graph. Nodes in the graph represent users. If two users are connected by an edge of weight 
, it means that they have both left positive reviews to the same 
 restaurants.

Additionally, you are given a matrix F that encodes user preferences to different categories of restaurants. If F[i, c] = 1, then user i likes restaurants in category c.
