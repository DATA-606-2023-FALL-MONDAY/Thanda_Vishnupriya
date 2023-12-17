import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from PIL import Image
#from streamlit_extras.switch_page_button import switch_page
#import streamlit_shadcn_ui as ui
import numpy as np
import csv

# Use CSS to style the dashboard components
style = """
<style>
    body {
        background-color: #f2f2f2;
        font-family: sans-serif;
        padding: 20px;
    }

    h1 {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }

    h2 {
        font-size: 20px;
        font-weight: bold;
    }

    p {
        font-size: 16px;
        line-height: 1.5;
    }

    a {
        color: blue;
        text-decoration: none;
    }

    button {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }

    button:hover {
        background-color: #0056b3;
    }
</style>
"""
# Load the data
data = pd.read_csv("papers_data.csv")
data = data.reset_index()
data = data.reset_index()
data.columns = ['id', 'id2', 'Title', 'Summary', 'Authors', 'Published', 'Link']
data = data[['id', 'Title', 'Summary', 'Authors', 'Published', 'Link']]
embeddings_specter = np.load('sentence_transformer_embeddings.npy')


# Define the search function using cosine similarity on titles
def search_papers(keyword):
    # Preprocess the keyword
    keyword = keyword.lower()
    return data[data["Title"].str.lower().str.contains(keyword)][0:5]

# Create the Streamlit app

# Set the page config for fullscreen mode and a wide layout
st.set_page_config(layout="wide")
st.write(style, unsafe_allow_html=True)

# Add a navigation menu to the sidebar
st.sidebar.title("Paper Craft AI")
# Display an image in the dashboard
image = Image.open("cover_image1.webp")
st.sidebar.image(image, caption="PaperCraft AI")

def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    value = dot_product / (norm1 * norm2)
    return value

def access_elements(nums, list_index):
    result = [nums[i] for i in list_index]
    result = result[::-1]
    return result[1:]


# Define the show_details function to show the details of paper
def show_details(paper_id):
    paper = data[data['id'] == paper_id].iloc[0]
    # Display the paper details
    st.title(paper['Title'])
    st.write(paper['Summary'])
    st.write(f"**Authors:** {paper['Authors']}")
    st.write(f"[Link]({paper['Link']})")

# Define the show_recommendations function using cosine similarity on summary
def show_recommendations(paper_id):

    query_embedding = embeddings_specter[paper_id]
    similarities = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings_specter])
    top_k_indexes = np.argsort(similarities)[-10:]
    top_k_indexes = top_k_indexes[::-1]
    top_k_indexes = top_k_indexes[1:]

    filtered_data_by_summary = data.iloc[top_k_indexes][:5]

    # Display related paper recommendations
    st.subheader("Related Papers")

    for i, row in filtered_data_by_summary.iterrows():
        title = row['Title']
        link = row['Link']

        # Display the related paper title and published date
        st.write(f"**{title}** {link}")

# Search for papers based on the user's input
def search_page(keyword):
    filtered_data = search_papers(keyword)

    # Display the search results if there are any
    if len(filtered_data) > 0:
        st.header('Search Results')

        for i, row in filtered_data.iterrows():
            title = row['Title']
            paper_id = row['id']
            summary = row['Summary']

            # Display the paper title and published date
            st.write(f"**{title}**")
            col1, col2 = st.columns([1,1,])

            with col1:
                # Add a button to view the paper summary
                if st.button(f"View Summary", key=f"details{i}"):
                    show_details(paper_id)
            with col2:
                # Add a button to view the paper recommendations
                if st.button(f"View Recommendations", key=f"recommendations{i}"):
                    #switch_page(show_recommendations(paper_id))
                    show_recommendations(paper_id)
    else:
        st.warning(f"No papers found for the keyword '{keyword}'.")

# Use a Streamlit input field for user input
keyword = st.text_input('Enter a keyword or title to search:', placeholder="Search for papers...")
st.write(search_page(keyword))
