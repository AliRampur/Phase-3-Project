# 1. Overview

For this project, I used pipelines, validation and grid searches to create the most effective model to predict a binary class (true / false) pertaining to customer churn.

   - Link to Technical Notebook: https://github.com/AliRampur/Phase-3-Project/blob/main/Phase%203%20Project.ipynb
   - Link to final presentation: https://github.com/AliRampur/Phase-3-Project/blob/main/Phase%203%20Presentation.pdf
   - Link to original data sources: https://github.com/AliRampur/Phase-3-Project/blob/main/data/SyriaTel%20Customer%20Data.csv

# 2. Business Problem

Legendary Preds is a consulting firm that works with telecom companies to help maximize revenue and reduce customer churn rates. SyriaTel has hired us to prepare and implement a predictive model that identifies current active customers who may terminate their contract (i.e. churn) with SyriaTel based on historical data and churn rates for 3,333 customer accounts.

If we can create a model that can help the company identify specific customers at risk of churning, SyriaTel can then focus on those customers and provide certain incentives, rebates or one-time offers for extending the term of their contracts.


# 3. Exploratory Data Analysis 

Our first step in analyzing the data was to verify the total number of records (3,333), review the target column (churn), review other independent features, potentially modify data types (e.g. area code), assess the existence of any NA values or duplicate phone numbers, and then prepare a correlation heat map of all the numerical fields to assess any multi-collinearity. Based on the results of our EDA, I then engineered additional features as deemed necessary.


   ### Target Feature ('y')
   The target feature or column is the "churn" column. I analyzed the value counts of this column to determine any potential imbalance.
   
    - True = 483 (~86%)
    - False = 2850 (~14%)
   
   I was comfortable with the total make up of this target column and moved forward with our analyze without synthetic data creation (e.g. SMOTE).
   
   ### Feature Engineering:
   After analyzing the various features and data types, I created a feature called "price per min - total", to help identify the average price per minute paid by each customer.
   
       df['price per min - total'] = (df['total day charge'] + df['total eve charge'] + df['total night charge'] + df['total intl charge']) / (df['total day minutes'] + df['total eve minutes'] + df['total night minutes'] + df['total intl minutes'])
   
   
   After preparing and validating a few initial model types, and reviewing the accuracy scores, I decided to add some other features titled "total charges", "total minutes" and "region". I felt these may be other potential indicators of customer churn rates, the thought being customers that pay more are more likely to churn, and customers may be churning more in certain regions than others.


   Here is a scatterplot of Total Charge vs. Total Minutes:
   
   ![image](https://github.com/AliRampur/Phase-3-Project/blob/main/pictures/Scatter%20Plot%20Charge%20vs%20Minutes.png)
   
   From this plot, we can see that your best customers (highest paying and highest usage) are churning.
   
   
   Here is a bar graph on $ Amount of Churn by region:
   
   ![image](https://github.com/AliRampur/Phase-3-Project/blob/main/pictures/Churn%20by%20Region.png)
   
   We can see that North appears to have the highest churn in terms of dollar amount, followed by South and West.
   
   
   Here is a boxplot of Average Total Charge by Area Code:
   
   ![image](https://github.com/AliRampur/Phase-3-Project/blob/main/pictures/Average%20Total%20Charges%20by%20Area%20Code%20-%20Box.png)

   
   ### Correlation Heat Map:
   ![image](https://github.com/AliRampur/Phase-3-Project/blob/main/pictures/Corr%20Heatmap.png)

    
   Although there is a .5 correlation between Total Charge and price per min - total, I kept both features as I felt both were relevant to identifying customers who may churn. However, I only left in Total Charge and removed Total Minutes from my final model, as the correlation was very high.
   

   ### Drop Features:
   In my first model iteration process, I removed the following (x) features from our modeling process, as they were either highly correlated with other features or I engineered a new feature as described above (e.g. 'price per min - total'):
 - total day charge
 - total day minutes
 - total eve charge
 - total eve minutes
 - total night charge
 - total night minutes
 - total intl charge
 - total intl minutes
 - total charge
 - total minutes
 - phone number
    
    In my 2nd model iteration process, I added back the 'total charge' feature. As summarized further below, this addition helped improve the model.

   
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

![image](https://github.com/AliRampur/Phase-3-Project/blob/main/pictures/CV%20Mean%20Scores%20and%20STD.png)

Based on the results above, I decided to add in the engineered feature described above - 'total charge' - and see if the models could be further improvied.



### _Pipeline and Cross Validation - 2nd Attempt_

After adding the 'total charge' feature and rerunning the Pipelines and Cross Validations, here is the updated mean scores and standard deviations for each model type:

![image](https://github.com/AliRampur/Phase-3-Project/blob/main/pictures/CV%20Mean%20Scores%20and%20STD%20%232.png)


Based on the results above, RFC and GBC had the two highest mean scores with a difference of only .006. Therefore, I selected both for consideration using GridSearch and further turning to arrive at the final model.

### _Grid Searching_

I then applied Grid Search to the two model types (RFC and GBC) with the following hyperparameters:


    #RFC HyperParamaters
    params = {}
    params['rfc__criterion'] = ['gini','entropy']
    params['rfc__min_samples_leaf'] = [1,5,10]
    params['rfc__max_depth'] = [5,7,9]


    #GBC HyperParamaters
    params2 = {"gbc__criterion": ["friedman_mse", "mae", "mse"],
              "gbc__loss":["deviance","exponential"],
              "gbc__max_depth": [3,5],
              "gbc__min_samples_leaf": [3,5],
              "gbc__subsample": [0.5,0.9],
              "gbc__n_estimators": [1,10,20]}

With setting a CV = 5, the RFC grid search ran 90 fits and the GBC grid search ran 720 fits.


### _Further Iterations_

Based on results of the grid searches, which resulted in GBC having the higher mean score and higher best score when compared with RFC for each iteration I attempted, I further adjusted and tweaked certain Hyperparameters.

The best Hyperparameters for GBC were:

 - criterion: 'friedman_mse'
 - loss: 'deviance'
 - max_depth: 3
 - min_samples_leaf: 3
 - n_estimators: 20
 - subsample: 0.9


However, the GBC best Hyperparameters above are still at the end of the given range. If given more time, I would further tune the Hyperparamaters (e.g. include max_depth of 1) to see if this would further improve the model accuracy score. 


# 5. Final Model

After applying GridSearch to both the RFC and GBC pipelines, GBC had the higher average CV score and the higher "best score". I then applied the final tuning to the GBC model based on the "best parameters" identified above resulting in the Final Model.


 - Final Score on Train Data: 0.9651860744297719
 - Final Score on Test Data: 0.9700239808153477

As summarized above, I applied the final model to the unseen test data (X_test, y_test), resulting in a score of .97. This is just .005 greater than the train score of .965 above, and therefore, not very indicative of overfitting or underfitting.

The results of applying the model to the test data (unseen) are summarized in the Confusion Matrix below:

![image](https://github.com/AliRampur/Phase-3-Project/blob/main/pictures/Confusion%20Matrix.png)


### The final model's accuracy on the test set is 0.97. 

### The final model's recall on the test set is 0.8 

### The final model's precision on the test set is 1.0 

### The final model's f1-score on the test is 0.89.


Although the accuracy of 97% implies a quite effective model, the recall and precision score are even more powerful in this case, as it shows that the model is very good in predicting customers that actually churn. The 100% precision means that there were no False positives predicted made by the model, and SyriaTel would not be wasting any resources or time by focusing its attention on the customers identified by the model.  


# 6. Recommendations / Next Steps


Based on the results of the final model, here are my recommendations:

 1. At the end of each quarter, feed the final model with the active customer account data
 
 2. Focus on customers that are identified by the model. Prioritize newly flagged customers
 
 3. Provide these customers with one-time incentives or consider rolling out “unlimited” plans by region


Next Steps:

If given additional time…

 - Further tune the model to increase the “recall” score 

 - Continue to add data each quarter and update model

 - Analyze new customer data to help identify potential trends
