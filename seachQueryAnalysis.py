import sqlite3
import os
import datetime
import argparse
import json
import re
import string
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim

"""
Firefox browsing history tool to quickly access daily data and build an interest timeline.
"""
german_stopwords = stopwords.words("german")
english_stopwords = stopwords.words("english")
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", re.UNICODE)


def parse_date(date_str: str) -> datetime:
    # date conversion from input
    format_string = "%Y-%m-%d"
    try:
        parsed_date = datetime.datetime.strptime(date_str, format_string)
        return parsed_date
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid date. Please use the format: YYYY-MM-DD")


def format_time(date: datetime):
    # time conversion to firefox history format
    date = int(datetime.datetime.timestamp(date)*1_000_000)
    return date


def firefox_scavenge(times):
    # gather search queries regarding given date
    query = f"SELECT title, last_visit_date FROM moz_places WHERE last_visit_date >= {times[0]} AND last_visit_date <= {times[1]};"
    firefox_path = os.path.expanduser(
        "~\\AppData\\Roaming\\Mozilla\\Firefox\\Profiles")
    dirs = os.listdir(firefox_path)
    for direct in dirs:
        db_path = firefox_path + "\\" + direct
        for filename in os.scandir(db_path):
            if filename.name == "places.sqlite":
                firefox_path = os.path.join(
                    os.path.join(firefox_path, direct), filename.name)
                data = get_data(firefox_path, query)
                return data
    return 


def json_file_scavenge(jason_path: str) -> list:
    # extracting json training data, separating it in 25 keyphrases
    data_list = []
    current_bundle = []
    num_lines = sum(1 for _ in open(jason_path))
    with open(jason_path, 'r') as file:
        for linenumber, line in enumerate(file):
            if linenumber <= num_lines:
                try:
                    # cleaning queries
                    data = json.loads(line)
                    keyphrase = data.get("keyphrase")
                    translator = str.maketrans('', '', string.punctuation)
                    cleaned_query = keyphrase.translate(translator)
                    cleaned_query = emoji_pattern.sub(
                        r'', cleaned_query).lower()
                    current_bundle.append(cleaned_query)
                    if len(current_bundle) == 25:
                        data_list.append(current_bundle)
                        current_bundle = []

                except json.JSONDecodeError as exception:
                    print(f"Json decoding error occurred {exception}")
    return data_list


def model_creation(data: list):
    # training basic gensim lda model, change stats if needed
    dictionary_data = corpora.Dictionary(data)
    texts = data
    corpus = [dictionary_data.doc2bow(text) for text in texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary_data,
                                                num_topics=20,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=50,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    print(lda_model.print_topics())
    lda_model.save("trainedModel")

    # model score
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))

    # coherence score
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=data, dictionary=dictionary_data, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    return (lda_model, dictionary_data)


def topic_information(modic: tuple, search_queries: list) -> None:
    # uses existing dictionary and trained model to analyse firefox queries
    dictionary_data = modic[1]
    lda_model = modic[0]
    texts = search_queries
    new_corpus = [dictionary_data.doc2bow(text) for text in texts]
    new_topics = []
    # iterate through dates(documents)
    for i in range(0, len(new_corpus)-1):
        topics = lda_model.get_document_topics(
            new_corpus[i], minimum_probability=0.0)
        most_likely = max(topics, key=lambda item: item[1])
        topic_name = lda_model.show_topic(most_likely[0])
        topic_name = [word for word, prob in topic_name]
        new_topics.append(topic_name)
    for doc_idx, topic_words in enumerate(new_topics):
        print(f"Dokument {doc_idx}: {', '.join(topic_words)}")

    vis = pyLDAvis.gensim.prepare(lda_model, new_corpus, dictionary_data)
    # nice data template, for data browsing
    pyLDAvis.save_html(vis, 'LDA_Visualization.html')


def get_data(connection_path, query: str) -> list:
    # extract firefox data and cleaning it
    bad_word = set(["duckduckgo", "youtube"])
    connection = sqlite3.connect(connection_path)
    cursor = connection.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    connection.close()
    seach_queries = []
    for row in rows:
        # cleaning queries, removing stopwords
        if str(row[0]) != "None":
            translator = str.maketrans('', '', string.punctuation)
            cleaned_query = row[0].translate(translator)
            cleaned_query = emoji_pattern.sub(r'', cleaned_query).lower()
            words = re.findall(r'\b\w+\b', cleaned_query)
            filtered_words = [word for word in words if word not in bad_word]
            filtered_words = [
                word for word in filtered_words if word not in german_stopwords]
            filtered_words = [
                word for word in filtered_words if word not in english_stopwords]
            porter = WordNetLemmatizer()
            filtered_string = [porter.lemmatize(
                word) for word in filtered_words]
            for i in filtered_string:
                if len(i) > 1:
                    seach_queries.append(i)
        else:
            continue
    return seach_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date", type=parse_date,
                        help="Date in the format YYYY-MM-DD")
    parser.add_argument("end_date", type=parse_date,
                        help="Date in the format YYYY-MM-DD")
    parser.add_argument("training_data", type=str,
                        help="Supply the path to the training data")
    args = parser.parse_args()

    delta = args.end_date - args.start_date
    day_data = []
    liste = json_file_scavenge(args.training_data)
    model = model_creation(liste)

    for i in range(delta.days + 1):
        current_date = args.start_date + datetime.timedelta(days=i)
        end_date = current_date.replace(hour=23, minute=59, second=59)
        times = [format_time(current_date), format_time(end_date)]
        day_list = firefox_scavenge(times)
        if len(day_list) > 0:
            day_data.append(day_list)
    topic_information(model, day_data)


if __name__ == "__main__":
    main()
