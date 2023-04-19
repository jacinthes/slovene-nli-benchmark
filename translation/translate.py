from deep_translator import GoogleTranslator
from datasets import load_dataset
from tqdm import tqdm
from datasets import Dataset


# Translate a batch of samples
def translate_batch(batch, translator):
    # Translate the premises and the hypotheses
    batch['premise'] = translator.translate_batch(batch['premise'])
    batch['hypothesis'] = translator.translate_batch(batch['hypothesis'])
    return batch

dataset = load_dataset("glue", "mnli")
#dataset = load_dataset("snli")  # If you want to work with the snli dataset

# Select the number of training samples to translate
num_samples = 2
train_subset = dataset['train'].select(range(num_samples))

# Define the batch size
batch_size = 32

# Create a new dataset object with translated examples
dataset_translated = {
    'premise': [],
    'hypothesis': [],
    'label': [],
    'org_premise': [],
    'org_hypothesis': [],

}
# Using Google's translator English to Slovene
translator = GoogleTranslator(source='en', target='sl')
for i in tqdm(range(0, len(train_subset), batch_size)):
    batch = train_subset[i:i + batch_size]
    dataset_translated['org_premise'].extend(batch['premise'])
    dataset_translated['org_hypothesis'].extend(batch['hypothesis'])
    batch_translated = translate_batch(batch, translator)
    dataset_translated['premise'].extend(batch_translated['premise'])
    dataset_translated['hypothesis'].extend(batch_translated['hypothesis'])
    dataset_translated['label'].extend(batch_translated['label'])

# Add source
dataset_translated['source'] = ['mnli'] * num_samples

# Create a Dataset object which can be pushed to HuggingFace
dataset_translated = Dataset.from_dict(dataset_translated)
df = dataset_translated.to_pandas()

#Save locally
df.to_excel('mnli_translated.xlsx')