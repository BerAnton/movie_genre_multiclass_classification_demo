"""Make genres prediction on given dialogue from movie"""
from classifier import Classifier
from train import DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH, DEFAULT_MLB_PATH


def main():
    """Main function"""
    clf = Classifier(DEFAULT_MODEL_PATH,
                     DEFAULT_VECTORIZER_PATH, DEFAULT_MLB_PATH)
    dialogue = input()
    prediction = clf.predict(dialogue)
    print(" ".join(sorted(prediction)))


if __name__ == "__main__":
    main()
