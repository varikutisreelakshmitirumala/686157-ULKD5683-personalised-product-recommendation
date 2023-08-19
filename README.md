# 686157-ULKD5683-personalised-product-recommendation
Ecommerce-product-recommendation-system
Product Recommendation System is a machine learning-based project that provides personalized product recommendations to users based on their browsing and purchase history. The system utilizes collaborative filtering and content-based filtering algorithms to analyze user behavior and generate relevant recommendations. This project aims to improve the overall shopping experience for users, increase sales for e-commerce businesses
# Dataset
I have used an amazon dataset on user ratings for electronic products, this dataset doesn't have any headers. To avoid biases, each product and user is assigned a unique identifier instead of using their name or any other potentially biased information.
You can find the dataset here - https://www.kaggle.com/datasets/vibivij/amazon-electronics-rating-datasetrecommendation/download?datasetVersionNumber=1
  You can find many other similar datasets here - https://jmcauley.ucsd.edu/data/amazon/

# Approach
# Rank Based Product Recommendation
Objective -
   Recommend products with highest number of ratings.
   Target new customers with most popular products.
   Solve the Cold Start Problem
Outputs -
  Recommend top 5 products with 5/1 minimum ratings/interactions.
Approach -
      
      
      Calculate average rating for each product.
      Calculate total number of ratings for each product.
      Create a DataFrame using these values and sort it by average.
      Write a function to get 'n' top products with specified minimum number of interactions.
  
   




