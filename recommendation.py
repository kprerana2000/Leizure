import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile


def get_data():
        zf = zipfile.ZipFile('dataset/booksdata.zip') 
        movie_data = pd.read_csv(zf.open('booksdata.csv'))
        movie_data['original_title'] = movie_data['original_title'].str.lower()
        return movie_data

def combine_data(books):
    books['corpus'] = (pd.Series(books[['authors', 'tag_name_x']].fillna('').values.tolist()).str.join(' '))
    return books
        
def transform_data(books, data_plot):
    tf_corpus = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix_corpus = tf_corpus.fit_transform(books['corpus'])
    cosine_sim_corpus = linear_kernel(tfidf_matrix_corpus, tfidf_matrix_corpus)
    return cosine_sim_corpus


def recommend_movies(title, books,cosine_sim_corpus):
        titles = books['original_title']
        indices = pd.Series(books.index, index=books['original_title'])
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim_corpus[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        book_indices = [i[0] for i in sim_scores]
        tit=titles.iloc[book_indices]
        auth=books['authors'].iloc[book_indices]
        recommendation_data = pd.DataFrame(columns=['Name','Authors'])
        recommendation_data['Name'] = tit
        recommendation_data['Authors'] = auth
        return recommendation_data



def results(movie_name):
        movie_name = movie_name.lower()

        find_movie = get_data()
        combine_result = combine_data(find_movie)
        transform_result = transform_data(combine_result,find_movie)

        if movie_name not in find_movie['original_title'].unique():
                return 'Book not in Database'

        else:
                recommendations = recommend_movies(movie_name, find_movie,transform_result)
                return recommendations.to_dict('records')