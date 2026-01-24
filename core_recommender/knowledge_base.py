
"""
knowledge_base.py

Centralized repository for "Storytelling" descriptions of models and techniques.
Used to populate the UI with educational and engaging content.
"""

MODEL_KNOWLEDGE = {
    "KNN": {
        "title": "K-Nearest Neighbors (KNN)",
        "story": "KNN is like asking your neighbors for advice. To make a prediction for a new data point, it looks at the 'k' most similar examples in history and takes a vote.",
        "how_it_works": "It calculates the distance between the new data point and all other points in the dataset. The majority class (for classification) or average value (for regression) of the nearest neighbors becomes the prediction.",
        "pros": ["Simple to understand", "No training phase (lazy learner)", "Adapts well to new data"],
        "cons": ["Slow with large datasets", "Sensitive to noise and outliers", "Requires feature scaling"],
        "best_for": "Small to medium datasets where relationships between data points are local."
    },
    "Random Forest": {
        "title": "Random Forest",
        "story": "Imagine a council of wise experts. A Random Forest creates many individual Decision Trees (the experts), each trained on a random subset of data. They all vote, and the majority wins.",
        "how_it_works": "It addresses the overfitting problem of single decision trees by averaging multiple deep decision trees, trained on different parts of the same training set.",
        "pros": ["Very accurate", "Resistant to overfitting", "Handles missing values well"],
        "cons": ["Complex to interpret", "Computationally expensive to train", "Large model size"],
        "best_for": "Complex datasets with many features, where high accuracy is paramount."
    },
    "Decision Tree": {
        "title": "Decision Tree",
        "story": "Think of a flowchart or a game of '20 Questions'. The model splits data step-by-step based on simple rules (e.g., 'Is value > 5?') until it reaches a conclusion.",
        "how_it_works": "It splits the data into subsets based on the most significant attribute at each node. This process repeats recursively until a stopping criterion is met.",
        "pros": ["Easy to visualize and explain", "Requires little data preparation", "Handles both numerical and categorical data"],
        "cons": ["Prone to overfitting (memorizing data)", "Unstable (small changes in data change the tree)", "Can be biased"],
        "best_for": "Problems where interpretability is key and you need to understand the 'why' behind a decision."
    },
    "SVM": {
        "title": "Support Vector Machine (SVM)",
        "story": "SVM is like finding the widest possible street that separates two groups of houses. It tries to draw a line (or boundary) that keeps the classes as far apart as possible.",
        "how_it_works": "It maps data to a high-dimensional space and finds the hyperplane that maximizes the margin between the classes.",
        "pros": ["Effective in high dimensional spaces", "Memory efficient", "Versatile kernels"],
        "cons": ["Hard to interpret", "Sensitive to noise", "Slow on large datasets"],
        "best_for": "Complex classification problems with clear margins of separation, like image or text classification."
    },
    "Naive Bayes": {
        "title": "Naive Bayes",
        "story": "The 'Optimist' of models. It assumes all features are independent (which is 'naive'), but uses probability theory (Bayes' Theorem) to make surprisingly fast and accurate predictions.",
        "how_it_works": "It calculates the probability of each class given the input features, assuming independence between features.",
        "pros": ["Extremely fast", "Works well with high dimensions", "Good for text classification"],
        "cons": ["'Naive' assumption often false", "Needs representative training data", "Zero frequency problem"],
        "best_for": "Real-time prediction, text classification, and spam filtering."
    },
    "Logistic Regression": {
        "title": "Logistic Regression",
        "story": "Despite the name, it's for classification! It estimates the probability of an event happening (like 'Click' vs 'No Click') by fitting data to an S-shaped curve.",
        "how_it_works": "It uses the logistic function to model a binary dependent variable.",
        "pros": ["Probabilistic interpretation", "Easy to update with new data", "Resistant to overfitting"],
        "cons": ["Assumes linearity between independent variables and log-odds", "Can't handle complex relationships well"],
        "best_for": "Binary classification where knowing the *probability* is important (e.g., credit scoring)."
    },
    "Linear Regression": {
        "title": "Linear Regression",
        "story": "The classic trend-finder. It tries to draw the straight line that best fits your data points, minimizing the distance between the line and the actual values.",
        "how_it_works": "It models the relationship between two variables by fitting a linear equation to observed data.",
        "pros": ["Simple and interpretable", "Fast to train", "Good baseline"],
        "cons": ["Assumes linear relationship", "Sensitive to outliers", "Prone to underfitting"],
        "best_for": "Predicting continuous values (like house prices) where the trend is relatively simple."
    }
}
