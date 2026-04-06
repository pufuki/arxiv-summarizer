import arxiv
import pandas as pd

client = arxiv.Client()

query = '"Game theory"'

search = arxiv.Search(query=query, max_results=7, sort_by=arxiv.SortCriterion.SubmittedDate)

papers = []

#Fetch the papers
for result in client.results(search):
  papers.append({
      'published': result.published,
      'title': result.title,
      'abstract': result.summary,
      'categories': result.categories
  })

#Convert to dataframe
df = pd.DataFrame(papers)

pd.set_option('display.max_colwidth',None)
num = 4
df.head(num)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Example abstract from API
abstract = df['abstract'][0]

model_name = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Prepare the input for the model
inputs = tokenizer([abstract], max_length=1024, return_tensors='pt', truncation=True)

# Generate the summary
summary_ids = model.generate(
    inputs['input_ids'], num_beams=4, min_length=30, max_length=150, early_stopping=True
)

# Decode the summary
summarization_result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

summarization_result[]
