import requests
from bs4 import BeautifulSoup
import pandas as pd
from sentence_transformers import SentenceTransformer
import time

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# URL to fetch free courses
BASE_URL = "https://courses.analyticsvidhya.com"

def fetch_courses():
    url = f"{BASE_URL}/collections?category=free"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    courses = []

    # Look for course links on the "free" page
    for anchor in soup.select("a.bundle_card"):
        title = anchor.select_one("div.course_name")
        description = anchor.select_one("div.course_details")
        link = BASE_URL + anchor.get("href", "")

        if title and description:
            course_info = {
                "title": title.get_text(strip=True),
                "description": description.get_text(strip=True),
                "link": link
            }
            courses.append(course_info)

    return courses

def add_embeddings(courses):
    for course in courses:
        embedding = model.encode(course['description'])
        course['embedding'] = embedding.tolist()
    return courses

def save_to_csv(courses, filename="courses_data.csv"):
    df = pd.DataFrame(courses)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} courses to {filename}")

def main():
    print("Fetching courses from Analytics Vidhya...")
    courses = fetch_courses()
    if not courses:
        print("No courses found.")
        return

    print("Generating embeddings...")
    courses = add_embeddings(courses)

    print("Saving to CSV...")
    save_to_csv(courses)

if __name__ == "__main__":
    main()
