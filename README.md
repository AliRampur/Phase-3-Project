# 1. Overview

For this project, I used pipelines, validation and grid searches to create the most effective model to predict a binary class (true / false) pertaining to customer churn.

   - Link to Technical Notebook: 
   - Link to final presentation: 
   - Link to original data sources: 

# 2. Business Problem

Legendary Preds is a consulting firm that works with telecom companies to help maximize revenue and reduce customer churn rates. SyriaTel has hired us to  prepare a model that identifies current active customers who may terminate their contract with SyriaTel based on historical data and churn rates for 3,333 customer accounts.

If we can create a model that can help the company identify specific customers at risk of churning, SyriaTel can then focus on those customers and provide certain incentives, rebates or one-time offers for extending the term of their current contracts.


# 3. Exploratory Data Analysis 

Our first step in analyzing the data was to verify the total number of records (3,333), review the target column, review other independent features, potentially modify data types (e.g. area code), assess the existence of any NA values or duplicate phone numbers, and then prepare a correlation heat map of all the numerical fields to assess any multi-collinearity. Based on the results of our EDA, I then engineered additional features as deemed necessary.


   ### Target Feature ('y')
   The target feature or column is the "churn" column. I analyzed the value counts of this column to determine any potential imbalance.
   
   True = 483 (~86%)
   False = 2850 (~14%)
   
   I was comfortable with the total make up of this target column and moved forward with our analyze without synthetic data creation (e.g. SMOTE).
   
   ### Feature Engineering:
   After analyzing the various features and data types, I created a feature called "price per min - total", to help identify the average price per minute paid by each customer.
   
       df['price per min - total'] = (df['total day charge'] + df['total eve charge'] + df['total night charge'] + df['total intl charge']) / (df['total day minutes'] + df['total eve minutes'] + df['total night minutes'] + df['total intl minutes'])
   
   
   After preparing and validating a few initial model types, and reviewing the accuracy scores, I decided to add another feature titled "total charges". I felt this would be another potential indicator of customer churn. The thought being customers that pay more are more likely to churn.
   
       df['total charge'] = (df['total day charge'] + df['total eve charge'] + df['total night charge'] + df['total intl charge'])

   Here is a graph of Average Total Charge by Area Code:
   
   ![image]()

   
   ### Correlation Heat Map:
   ![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/corr_heatmap.png)

    
   Although there is a .5 correlation between Total Charge and price per min - total, I kept both features as I felt both were relevant to identifying customers who may chur.
   

   ### Drop Features:
   I removed the following features from our modeling process, as they were either highly correlated with other features or I engineered a new feature as described above:
 - total day charge
 - total day minutes
 - total eve charge
 - total eve minutes
 - total night charge
 - total night minutes
 - total intl charge
 - total intl minutes
 - phone number
    

   
# 4. Modeling and Evaluation

As part of the modeling process, I first setup the train/test/split using the x features and y feature described above. I then created subpipelines using 'selector', one for numerical features and another for categorical features, and then used a ColumnTransformer to create the necessary array that would be used within each overall model pipeline. We also setup a class ModelWithCV() (taken from Flatiron Lecture #51) to help streamline the process of applying cross validation and extracting the results on each model pipeline. 

I then instantiated a pipeline for each of the following models, beginning with the Dummy Classifier. The other model types I considered in my analysis include:

- Logistic Regression
- Random Forest Classifier
- Gradient Boost Classifier
- Decision Tree
- K Nearest Neighbors
- Ada Boost Classifier

### _Pipeline and Cross Validation - 1st Attempt_

Using the pipeline and class ModelWithCV(), I applied a 10 kfold cross validation to each of the 7 model pipeline described above and arrived at the following mean scores +/- standard deviations.

![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/Scatter_Price%20vs%20Living%20Sqft.png)

Based on the results above, I decided to add in an additional feature described above - 'total charge' - and see if the models could be further improvied.



### _Pipeline and Cross Validation - 2nd Attempt_

After adding the 'total charge' feature and rerunning the Pipelines and Cross Validations, here is the updated mean scores and standard deviations for each model type:

![image](https://github.com/AliRampur/Phase-2-Group-Project/blob/main/pics/Scatter_Price%20vs%20Living%20Sqft.png)


Based on the results above, RFC and GBC had the two highest mean scores with a difference of only .006. Therefore, I selected both for consideration using GridSearch and further turning to arrive at the final model.

### _Grid Searching_

I then applied Grid Search to the two model types with the following parameters:






### _Further Iterations_

Based on the best parameters given for the GBC model, which had the higher mean score and the higher best score when compared with RFC,
I further adjusted certain parameter since they were at the end of the range I provided.






# 7. Recommendations / Conclusion


Based on our data analysis and the visualizations above, here are some key recommendations for King County Development to consider:

   1. The bigger the house, the higher the price. That said, we recommend houses at least in the 2,000 sqft range to garner higher interest from your average family and allow King County Development to control costs.
    
   2. The top home prices were generally in Medina, Clyde Hill, Mercer Island. 

   3. Build quality (i.e. grade) matters. This could be due to multiple factors, such as the impact of weather (rain and snow), the county being right off the shoreline, and the fact that people in this area command a higher salary and expect a higher overall build quality.

   4. Waterfronts and nicer views typically command a higher price. Even if shorelines are fully developed, King County Development should consider creating "man-made" lakes near areas that have a good view of the mountains or developing in areas that have access to natural bodies of water.

We were also able to create both an inferential and predictive model, which were analyzed to see if they meet the assumptions of linear regression and error.

   1. Our predictive model only meets one of the tenets of linear regression. Our inferential model, that uses log(price), is able to meet all assumptions of linear regression except having a normal distribution. We would continue to work on this model and it's variables given additional time.
   
   2. Our predictive model has an MAE of $270000, and an RMSE of $490000.
