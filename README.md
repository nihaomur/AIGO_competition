# AIGO Competition - Predict House Value
This is a competition hosted bt *Bank SinoPac* and *Trendmicro*, aiming to utilize **utilize machine learning methods for house price prediction**. </p>
I have prepared a presentation outlining my work, which I have uploaded on [YouTube](https://www.youtube.com/watch?v=IHKnk4oO_f0) along with the [presentation slides](https://drive.google.com/file/d/1TtM6ZHAStXn6w2Whlg3uxuen6iA1zAG0/view?usp=share_link
).Please feel free to comment and engage in discussions.

## Feature Engineering
A pivotal aspect of my work involved creating a novel feature by integrating **Actual Price Registration of Real Estate Transactions**.</p>
I developed a for loop to navigate through both the training data and Real Estate Transactions data to determine estimated prices based on geographical intersections (at the street level). For detailed information, refer to [this file](finding_average_price_1111.ipynb). </p>
I also created a couple of new features using the external dataset provided by the organizers. Utilizing numpy matrix multiplication to calculate the Euclidean distance, I determined the number of influencing factors surrounding each data point. My computational process is documented within [this file](np.ipynb).

## Model Training using ML Methods
Given the sparse matrix I generated in my data frame (utilizing one-hot encoding to handle categorical features), I opted for models based on *decision trees*. </p>
Random Forest, XGBoost, and CatBoost generated distinct predictions, culminating in a weighted average prediction. For detailed information, refer to [this file](AIGO_competition.py).

## Final Grade
The competition concluded on 2023/11/13, and I ranked in the top 10% as of 11/12, achieving a final grade among 107/972 teams.

## Extending DNN Model
In an effort to assess diverse outcomes, I constructed a [Multilayer Perceptron (MLP) within a Deep Neural Network (DNN)](Elmomentum_1103_Dense.ipynb). Despite the prediction not matching the efficacy of the ML methods, it still exhibited commendable performance.
