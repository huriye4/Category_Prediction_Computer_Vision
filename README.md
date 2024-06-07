Category Prediction with Computer Vision

Introduction

Accurate categorization of products is critical for e-commerce companies to enhance customer satisfaction and optimize sales. Proper categorization helps customers find what they're looking for quickly and easily, improving their shopping experience and boosting sales. Misclassified products can lead to customer dissatisfaction, reduced product visibility, and potential sales loss. Furthermore, incorrect categorization can cause errors in inventory management and marketing strategies, increasing operational costs.

To address these issues, a model that predicts product categories from images has been developed. This model aims to reduce misclassification, increase sales, and shorten the time spent on product categorization. By enabling accurate categorization, customers can find products more quickly, enhancing their shopping experience and satisfaction. Additionally, using accurate data in inventory management and marketing strategies improves operational efficiency and cost savings. This model contributes to the company's overall performance, providing a competitive advantage.

Project Phases and Methods
1. Data Collection and Review
   
Keepa Dataset Review: The Keepa dataset was examined, and product images were downloaded using the "Product Images" variable. A total of 8,281 image URLs were identified across 10 categories.
Walmart Dataset Review: The Walmart dataset was examined, and 1,044 image URLs were identified. These images were downloaded using the request library, named according to their "Product Category" variables, and organized into 8 different category folders.

2. Data Imbalance and Web Scraping
Due to the insufficiency of 9,325 images for model training and category imbalances, additional photos were downloaded using web scraping from e-commerce sites such as Amazon, Walmart, and eBay. Web scraping, which automates data collection from web pages, played a critical role in enriching the dataset for this project.

3. Creating Data Sets
Three different folders (train, test, validation) under 9 categories were created for model training. The total datasets are as follows:

Final Data Set 1: Total 15,119 photos (train: 11,330, val: 3,029, deployment_check: 760)
Final Data Set 2: Total 15,510 photos (train: 11,629, val: 3,099, deployment_check: 782)
Final Data Set 3: Total 18,175 photos (train: 13,992, val: 3,632, deployment_check: 551)

4. Model Training
Various deep learning techniques were used to develop a model for accurate categorization of product images. The steps and methods followed during the model training process include:

Data Preparation

ImageDataGenerator Usage: Data augmentation techniques were applied using Keras's ImageDataGenerator class to increase variation in the dataset. Transformations such as rotation, horizontal/vertical flipping, zooming, and brightness adjustments were performed on the photos. These augmentations improved the model's generalization ability and reduced overfitting.

Model Selection and Architecture Experiments
Pre-trained Models: Transfer learning was utilized with pre-trained models such as VGG19, VGG16, ResNet, MobileNetV2, EfficientNet, and InceptionV3, which are successful in identifying basic features due to their training on large datasets.

Different Optimization Algorithms: Various optimization algorithms, including Adam and RMSprop, were tested to enhance the model's accuracy and reduce loss.

Model Evaluation and Results

Performance Evaluation: During training, the model's performance was continuously monitored using the train, validation, and test sets. Metrics such as accuracy, loss, validation accuracy, and validation loss were used to evaluate the model's success.

Best Model Selection: The model that showed the best performance was selected after various models and training processes. MobileNetV2 and InceptionV3 achieved the best results, with MobileNetV2 showing the highest accuracy.
