from sklearn.tree import DecisionTreeClassifier

def build_model():
    return DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=4,
        random_state=42
    )
