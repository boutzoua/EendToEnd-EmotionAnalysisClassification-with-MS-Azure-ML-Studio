# EndtoEnd - Emotion Analysis Classification Project with MS Azure ML Studio
"Emotional Analysis of Social Media for Stock Market Prediction"

The main objective of our work was to develop a classification model for emotions in comments and social media posts. We used an emotional dataset comprising over 700,000 entries, with five emotional classes: happy, sad, love, surprise, and worry.

Data preprocessing was performed using various techniques, including the creation of an additional dataset and correction of typing errors to ensure text cleanliness. For model training, we used a bi-LSTM (Bidirectional Long Short Memory) architecture, with an initial training phase without word embeddings and a second phase that incorporated Glove word embeddings to enhance word representation.

We compared the performance of our model with other classical machine learning algorithms using the MLFlow framework. This allowed us to select the most effective model for deployment. Ultimately, we deployed our model on the Azure cloud platform, with a user-friendly interface developed using Streamlit.

The results obtained demonstrated high accuracy in emotion classification, with optimal performance achieved in certain emotional domains. This project has paved the way for future possibilities of improvement and research in the field of emotional analysis of social media.

# Metrics
For the LSTM Model: <br>
Classification Report and CM: <br>
![alt text](assets/bilstm-cr.jpg?raw=true) <br>
![alt text](assets/bilstm-cm.jpg?raw=true) <br>
For the LSTM Model + GLOVE EMBEDDING: <br>
Classification Report and CM:<br>
![alt text](assets/glove-cr.jpg?raw=true)<br>
![alt text](assets/glove-cm.jpg?raw=true)<br>

By using the Azure cloud platform, we benefit from a robust infrastructure and efficient management tools for deploying our model. This enables us to provide a reliable, scalable, and secure solution for emotional analysis in social media. <br>
AzureEndpoint<br>
![alt text](assets/azureEndpoint.jpg?raw=true)<br>
AzureEndpointConsume<br>
![alt text](assets/azureEndpointConsume.jpg?raw=true)<br>
AzureEndpointlogs<br>
![alt text](assets/azureEndpointlogs.jpg?raw=true)<br>

By using Streamlit, we have created a user-friendly interface that allows end-users to interact easily with our emotional analysis model. This enables them to quickly and efficiently obtain emotional predictions based on comments and social media posts. <br>
![alt text](assets/streamlitcode2.jpg?raw=true)<br>
![alt text](assets/happy.jpg?raw=true)<br>
![alt text](assets/love.jpg?raw=true)<br>
![alt text](assets/sad2.jpg?raw=true)<br>
![alt text](assets/worry.jpg?raw=true)<br>
![alt text](assets/mixed.jpg?raw=true)<br>



# Keywords: 
emotion analysis, social media, classification, bi-LSTM model, RandomForest, XGBoost, SVM, Glove word embeddings, MLOps, MLFlow, deployment on Azure, Streamlit interface.


