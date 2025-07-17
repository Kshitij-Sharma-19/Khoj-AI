import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load course data
df = pd.read_csv("courses_data.csv")

# Load embeddings as lists (or arrays if using .npy files for larger datasets)
df['embedding'] = df['embedding'].apply(eval)

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to search courses
def search_courses(query, top_k=5):
    # Generate embedding for the user query
    query_embedding = model.encode(query)

    # Compute cosine similarity between query and course embeddings
    df['similarity'] = df['embedding'].apply(lambda x: util.cos_sim(query_embedding, x).item())

    # Sort by similarity and return top_k results
    results = df.sort_values(by='similarity', ascending=False).head(top_k)
    return results[['title', 'description', 'link']]

# # Streamlit UI
# st.title("Khoj AI")
# st.write("Find the most relevant free courses on Analytics Vidhya by entering a keyword or phrase.")

# # User input
# query = st.text_input("Enter your search query")

# # Display search results
# if query:
#     results = search_courses(query)
#     st.write("### Search Results:")
#     for _, row in results.iterrows():
#         st.write(f"**Title**: {row['title']}")
#         st.write(f"**Description**: {row['description']}")
#         st.write(f"[Course Link]({row['link']})")
#         st.write("---")


st.markdown("""
    <style>
    /* Set background color */
    body {
        background-color: #04192f;
    }
    /* Title styling */
    .title {
        font-size: 2.5em;
        color: white;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .description {
        font-size: 1.1em;
        color: #e2e2e2;
        text-align: center;
        margin-top: -0.5em;
        margin-bottom: 1em;
    }
    /* Input box styling */
    .input-container {
        text-align: center;
        margin: 1em auto;
    }
    .input-box {
        width: 100%;
        max-width: 500px;
        padding: 0.6em;
        font-size: 1.1em;
        border-radius: 8px;
        border: 1px solid #e2e2e2;
        background-color: #f7f7f7;
    }
    /* Container for search results */
    .search-results-container {
        background-color: #F4F6F7;
        padding: 1.5em;
        border-radius: 8px;
        margin-top: 1em;
    }
    /* Individual course item styling */
    .course-item {
        background-color: white;
        padding: 1em;
        border-radius: 8px;
        margin-bottom: 1em;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Course title styling */
    .course-title {
        font-size: 1.4em;
        color: #e2b814;
        font-weight: bold;
        margin: 0.5em 0 0.2em;
    }
    /* Course description styling */
    .course-description {
        display: list-item; /* Adds bullet points */
    list-style-position: inside; /* Aligns bullets with text */
    list-style-type: disc; /* Sets the bullet type to disc */
    padding-left: 1em; /* Adds indentation for text alignment */
    margin-bottom: 0.5em;
    font-size: 1em;
    color: #566573;
    max-width: 600px; /* Controls width to limit words per line */
    line-height: 1.5; /* Line spacing for readability */
    overflow-wrap: break-word; /* Ensures words wrap within the container */
    }   
    /* Course link styling */
    .course-link {
        font-size: 1em;
        color: #3498DB;
        text-decoration: none;
        font-weight: bold;
    }
    .course-link:hover {
        color: #1ABC9C;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class="title">Khoj AI</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Find the most relevant free courses on Analytics Vidhya by entering a keyword or phrase.</div>', unsafe_allow_html=True)

# User input for search with added margin and styling
st.markdown('<div class="input-container">', unsafe_allow_html=True)
query = st.text_input("Enter your search query:", placeholder="e.g., data science, machine learning, AI", key="input_box")
st.markdown('</div>', unsafe_allow_html=True)

# Display search results
if query:
    results = search_courses(query)
    if not results.empty:
        # Outer container for search results, only if there are results
        for _, row in results.iterrows():
            course_html = f"""
                <div class="search-results-container">
                    <div class="course-item">
                        <div class="course-title">{row['title']}</div>
                        <div class="course-description">{row['description']}</div>
                        <a href="{row['link']}" target="_blank" class="course-link">👉 Course Link</a>
                    </div>
                </div>
            """
            st.markdown(course_html, unsafe_allow_html=True)
    else:
        st.warning("No results found. Try a different keyword or phrase.")
