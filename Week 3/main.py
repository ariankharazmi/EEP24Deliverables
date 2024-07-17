import json
import random

texts = [ #This sample data is generated by OpenAI playground
    "Once upon a time in a faraway land, there lived a king who had three sons.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world in unprecedented ways.",
    "She sells seashells by the seashore.",
    "To be or not to be, that is the question.",
    "I loved this movie! The acting was great and the plot was intriguing.",
    "The film was a complete waste of time. Terrible script and poor acting.",
    "A wonderful journey through the life of a remarkable individual. Highly recommended!",
    "I didn't enjoy this book at all. The storyline was dull and the characters were uninteresting.",
    "The new policy is expected to boost the economy.",
    "The weather today is sunny with a chance of rain in the afternoon.",
    "The latest smartphone model has received mixed reviews from users.",
    "I can't believe how bad the customer service was at the restaurant.",
    "The painting was a beautiful masterpiece, captivating everyone who saw it.",
    "The team won the championship after a thrilling match.",
    "The software update caused more problems than it fixed.",
    "The new project has potential to bring significant advancements in the field.",
    "He found the book to be incredibly inspiring and life-changing.",
    "The lecture was so boring that many students fell asleep.",
    "She was praised for her dedication and hard work."
]

labels = ["neutral", "positive", "negative"]
sample_data = []
for _ in range(2000):
    text = random.choice(texts)
    label = random.choice(labels)
    sample_data.append({"text": text, "label": label})

with open("train.json", "w") as f:
    json.dump(sample_data, f, indent=4)

print("Generated 1,000 lines of training data and saved to 'datasets.json'.")
