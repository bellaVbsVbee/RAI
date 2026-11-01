from src.data import study_table, findings_texts, usable_findings, binary_label
from src.embeddings import embedding_path, load_embeddings

studies = study_table()
# X = load_embeddings(embedding_path("findings"), studies)   # shape (10000, 768)

# print(X.shape)
# print(studies.head())
# print(studies.columns)
print(studies["age_years"].unique())

# print(studies["findings_words"].head())

# print(studies["manufacturer"].head())

# print(studies["chexbert_support_devices"].head())
# columns = ['chexbert_edema', 'chexbert_pleural_effusion', 'health_system', 'manufacturer', 'age_years', 'study_year']

# for c in columns:
#     print(studies[c].unique())

# print(studies["chexbert_fracture"].head())