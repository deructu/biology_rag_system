import ollama
from sklearn.metrics.pairwise import cosine_similarity

LIST_OF_TEXTS = [
    "Транскрипція — це процес синтезу молекули РНК на матриці ДНК.",
    "Трансляція — це процес синтезу молекули білка на матриці іРНК",
    "Клітини прокаріотів не мають оформленого ядра, а їхня ДНК розташована безпосередньо в цитоплазмі.",
    "Клітини еукаріотів мають оформлене ядро, всередині якого міститься генетичний матеріал.",
    "Мітохондрії є енергетичною станцією клітини."
]

embeddings = ollama.embed(
    model="nomic-embed-text-v2-moe",
    input=LIST_OF_TEXTS
)["embeddings"]

print(cosine_similarity(embeddings))
