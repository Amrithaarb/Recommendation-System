"""
Simple Movie Recommendation System
Uses TF-IDF features from movie genres and descriptions, then cosine similarity.
Run: python recommendation_system.py
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIES = [
    {"title": "Interstellar", "genres": "science fiction space adventure drama", "description": "astronauts travel through a wormhole to find a new home for humanity"},
    {"title": "The Martian", "genres": "science fiction space adventure drama", "description": "an astronaut uses science and engineering to survive alone on Mars"},
    {"title": "Gravity", "genres": "science fiction space thriller", "description": "astronauts struggle to survive after an accident in orbit"},
    {"title": "Inception", "genres": "science fiction action thriller", "description": "a skilled team enters dreams to plant an idea in a target's mind"},
    {"title": "The Dark Knight", "genres": "action crime drama thriller", "description": "a hero protects a city from a chaotic criminal mastermind"},
    {"title": "Spider-Man: Into the Spider-Verse", "genres": "animation action adventure science fiction", "description": "a young hero meets alternate versions of himself across dimensions"},
    {"title": "Finding Nemo", "genres": "animation adventure family comedy", "description": "a clownfish travels across the ocean to find his missing son"},
    {"title": "Toy Story", "genres": "animation adventure family comedy", "description": "toys come to life and learn about friendship and teamwork"},
    {"title": "The Notebook", "genres": "romance drama", "description": "two people fall in love and face challenges over many years"},
    {"title": "Pride and Prejudice", "genres": "romance drama period", "description": "a young woman and a reserved man overcome misunderstandings"},
    {"title": "Knives Out", "genres": "mystery crime comedy", "description": "a detective investigates a suspicious death in a wealthy family"},
    {"title": "Sherlock Holmes", "genres": "mystery action crime adventure", "description": "a brilliant detective solves a complex criminal case"},
    {"title": "Coco", "genres": "animation family fantasy music", "description": "a boy explores the land of the dead and discovers his family history"},
    {"title": "The Lion King", "genres": "animation adventure family drama", "description": "a young lion learns responsibility and returns to reclaim his home"},
    {"title": "A Quiet Place", "genres": "horror thriller science fiction", "description": "a family tries to survive in a world where creatures hunt by sound"},
]

def build_features(movies):
    """Combine genre and description text, then convert it into TF-IDF vectors."""
    documents = [movie["genres"] + " " + movie["description"] for movie in movies]
    vectorizer = TfidfVectorizer(stop_words="english")
    return vectorizer.fit_transform(documents)

def recommend(title, movies=MOVIES, count=5):
    """Return titles most similar to the selected movie."""
    titles = [movie["title"] for movie in movies]
    matches = [i for i, name in enumerate(titles) if name.lower() == title.lower()]
    if not matches:
        return None
    features = build_features(movies)
    similarities = cosine_similarity(features[matches[0]], features).flatten()
    ranked = sorted(enumerate(similarities), key=lambda item: item[1], reverse=True)
    return [(movies[i]["title"], score) for i, score in ranked if i != matches[0]][:count]

def main():
    print("=== Simple Movie Recommendation System ===")
    print("Available movies:")
    for movie in MOVIES:
        print(f"- {movie['title']}")
    print("\nType a movie title to get similar recommendations, or 'quit' to exit.")
    while True:
        choice = input("\nMovie title: ").strip()
        if choice.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        results = recommend(choice)
        if results is None:
            print("Movie not found. Please choose a title from the list.")
            continue
        print(f"\nRecommendations similar to '{choice}':")
        for rank, (title, score) in enumerate(results, start=1):
            print(f"{rank}. {title} (similarity: {score:.2f})")

if __name__ == "__main__":
    main()