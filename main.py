import spacy

# Load spaCy's medium English model with word vectors
nlp = spacy.load("en_core_web_md")

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

def semantic_search(data, query, threshold=0.65):
    query_doc = nlp(query)
    results = []

    for line in data:
        line_doc = nlp(line)
        similarity = query_doc.similarity(line_doc)
        if similarity >= threshold:
            results.append((line, similarity))

    # Sort by similarity score, descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results

def main():
    file_path = 'data.txt'
    data = load_data(file_path)

    while True:
        query = input("Search (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break

        results = semantic_search(data, query)
        if results:
            print("\nResults:")
            for result, score in results:
                print(f"- {result}  (Score: {score:.2f})")
        else:
            print("No semantically similar matches found.\n")

if __name__ == "__main__":
    main()
