# Amazon Recommendation Engine

Since its inception nearly 25 years ago, Amazon has been focused on delivering a world class customer experience. In a world where consumer choice is paramount, Amazon continues to need better ways to target new and existing customers by presenting them with products they are most likely to purchase. The key to solving this problem lies in one of Amazon’s greatest data assets: customer reviews.

Amazon has a subset of customers who write reviews about their purchased products (reviewers). Reviewers tend to be more engaged with Amazon’s platform, giving Amazon the ability to increase product exposure with this segment. By mining historical reviews for information on product preference and sentiment, Amazon can understand and utilize reviewer preference to develop a more personalized product recommendation experience for each reviewer. 

Dashboard link deployed here using a hobby instance of Heroku: https://cognoclick.herokuapp.com/

**Disclaimer:** This project was completed as part of the MSDS 498 Capstone Project course within the Northwestern University. All data, dashboards, and insights used throughout this project are completely simulated and not in any way connected to or a reﬂection of Amazon. Please do not duplicate or distribute outside of the context of this course. 

## Repo Structure

* **build** contains all code files to prepare data, run models, and visualize results.
* **dash** contains starter code for the dashboard portion of the project. The source of truth for the dashboard code base can be found here: https://github.com/dashpound/review_dashboard. 
* **data** contains input and output data used for the project.
* **documentation** contains final project reports and git getting started documentation.
* **logs** contains log files from the DNN model runs.
* **output** contains model objects for the project.

### Prerequisites

The following packages will need to be installed in order to run the project code.

```
python3.7.x
nltk
gensim.models.doc2vec
tensorflow
keras
```

## Contributing

* John Kiley
* Brian Merrill
* Hemant Patel
* Julia Rodd

## Acknowledgments

* Julian McAuley - provided packaged recommendation and product metadata that was used to support this project.
