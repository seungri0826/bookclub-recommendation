from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

book_info_total = pd.read_pickle('./book_info_total.pkl')

# 사용자가 검색어로 넣는 책 이름의 띄어쓰기가 정확하지 않아도 검색 가능하도록 함
def get_book_idx_by_title(book_title):
    # 책 제목이 검색어와 완전히 일치하지 않아도 포함만 되면 검색 되도록 함
    book_title_concat = book_title.replace(' ', '')
    book_idx = book_info_total.index[(book_info_total['title_concat'] == book_title_concat)].to_list()
    if book_idx == []:
        book_idx = book_info_total.index[(book_info_total['title_concat'].str.contains(book_title_concat))].to_list()
    if book_idx:
        i = random.randrange(len(book_idx))
        return book_idx[i]
    else:
        return None

def get_book_title(book_idx):
    if book_idx in book_info_total.index.values:
        return book_info_total.loc[book_idx]['title']
    else:
        return None
    
# 사용자가 검색어로 넣은 단어가 책의 feature keywords 중 하나와 매치되는지
def get_book_idx_by_keyword(keyword):
    book_idx = book_info_total.index[(book_info_total['feature'].str.contains(keyword))].to_list()
    if book_idx:
        i = random.randrange(len(book_idx))
        return book_idx[i]
    else:
        return None

# min_df를 1로 설정해줌으로써 한번이라도 노출된 정보도 다 고려함
# ngram_range: n_gram 범위 지정 연속으로 나오는 단어들의 순서도 고려함
tf = TfidfVectorizer(min_df=1, ngram_range=(1,5))
tfidf_matrix = tf.fit_transform(book_info_total['feature'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 제목으로 검색
def similar_books_by_title(title, n=10):
    title = title.replace(' ', '')
    # 해당 제목의 책의 index를 구한다.
    book_idx = get_book_idx_by_title(title)
    if book_idx is None:
        #print("해당 도서는 데이터셋에 존재하지 않습니다. 다른 도서나 키워드로 다시 검색해주세요.")
        return None
    else:
        # 모든 책에 대해서 해당 책과의 cosine similarity를 구한다.
        book_similarities = list(enumerate(cosine_sim[book_idx]))
        # cosine similarity에 따라 책들을 정렬한다.
        book_similarities = sorted(book_similarities, key=lambda x:x[1], reverse=True)
        most_similar_books = list(map(lambda x: (get_book_title(x[0]), x[0]), book_similarities[:n+1]))       
        most_similar_books = list(filter(lambda x: title != x[0].replace(' ',''), most_similar_books))
        book_indices = []
        # 검색어가 책 제목에 포함되는 경우, 첫번째 검색 결과부터 n개
        if len(most_similar_books) > n:
            for i, j in book_similarities[:n]:
                book_indices.append(i)
        # 검색어가 책 제목과 완전히 일치하는 경우, 두번째 검색 결과부터 n개 (첫번째 검색 결과는 자기자신이므로)
        else:
            for i, j in book_similarities[1:n+1]:
                book_indices.append(i) 
        return(book_indices)
        
# 카테고리, 저자, 키워드로 검색
def similar_books_by_keyword(keyword, n=10):
    # 해당 제목의 책의 index를 구한다.
    book_idx = get_book_idx_by_keyword(keyword)
    if book_idx is None:
        #print("해당 키워드는 검색이 불가능합니다. 다른 도서나 키워드, 카테고리, 저자명으로 다시 검색해주세요.\n혹시 여러 개의 키워드를 입력하였다면 개수를 줄여주세요.")
        return None
    else:
        # 모든 책에 대해서 해당 책과의 cosine similarity를 구한다.
        book_similarities = list(enumerate(cosine_sim[book_idx]))
        # cosine similarity에 따라 책들을 정렬한다.
        book_similarities = sorted(book_similarities, key=lambda x:x[1], reverse=True)
        most_similar_books = list(map(lambda x: (get_book_title(x[0]), x[0]), book_similarities[:n]))       
        most_similar_books = list(filter(lambda x: keyword != x[0], most_similar_books))
        book_indices = []
        for i, j in book_similarities[:n]:
            book_indices.append(i)
        return(book_indices)

# 검색 결과 dataframe을 json으로 변환
def dataframe_to_json(dataframe):
    return dataframe.to_json(orient = 'records', force_ascii=False)


# 책 제목으로 검색
# request에 "input" 넣기
@app.route("/title", methods=['GET','POST'])
def title():
    content = request.json
    idx = similar_books_by_title(content['input'])

    # 검색 결과가 없는 경우 
    if idx == None:
      result = {"message": "해당 도서는 데이터셋에 존재하지 않습니다. 다른 도서나 키워드로 다시 검색해주세요."}
      return jsonify(result)

    result_df = book_info_total.loc[idx, ['title', 'author', 'rating', 'category', 'tag', 'keyword', 'image']]
    result_json = dataframe_to_json(result_df)
    return result_json

# 저자, 카테고리, 키워드로 검색
# request에 "input" 넣기
@app.route("/keyword", methods=['GET','POST'])
def keyword():
    content = request.json
    idx = similar_books_by_keyword(content['input'])

    # 검색 결과가 없는 경우 
    if idx == None:
      result = {"message": "해당 키워드는 검색이 불가능합니다. 다른 도서나 키워드, 카테고리, 저자명으로 다시 검색해주세요.\n여러 개의 키워드를 입력하였다면 개수를 줄여주세요."}
      return jsonify(result)
      
    result_df = book_info_total.loc[idx, ['title', 'author', 'rating', 'category', 'tag', 'keyword', 'image']]
    result_json = dataframe_to_json(result_df)
    return result_json


if __name__ == '__main__':
  app.run(host="127.0.0.1", port=5000, debug=True)