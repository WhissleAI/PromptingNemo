import spacy

# Load the German language model
nlp = spacy.load('de_core_news_lg')

# Sample German textuu
text = """noch wirksamer als amphion's leier sollte die pfeife der locomotive sie bald aus dem amerikanischen
 boden hervorwachsen lassen um acht uhr vormittags hatte man das fort mac pherson hinter sich welches dreihundertsiebenundf
ünfzig meilen von omaha entfernt liegt"""

# Process the text
doc = nlp(text)

# Extract and print named entities
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
