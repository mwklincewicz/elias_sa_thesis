# Lemotif cleaning + extra EDA

This package adds:
- one data cleaning script
- three additional EDA scripts for thesis-ready visuals

Files:
- src/clean_data.py
- src/eda_extra/01_emotion_cooccurrence_network.py
- src/eda_extra/02_length_vs_emotion_probability.py
- src/eda_extra/03_topic_to_emotion_probability.py

Put your CSV here:
data/Lemotif thesis data.csv

Install:
python -m pip install -r requirements.txt

Run cleaning:
python src/clean_data.py

Run extra EDA:
python src/eda_extra/01_emotion_cooccurrence_network.py
python src/eda_extra/02_length_vs_emotion_probability.py
python src/eda_extra/03_topic_to_emotion_probability.py
