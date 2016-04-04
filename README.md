# Recogntion-Error-Classification

This is the tool and data for Interspeech 2016 paper submission "A Language Independent Feature for Identifying Recognition Errors in Conversational Speech"

This is a python based tool, for this to work, you also need:

###python-crfsuite:
`https://python-crfsuite.readthedocs.org/en/latest/`

###scikit-learn:
`http://scikit-learn.org/stable/`

In order to generate the data in raw format, you need:

`https://github.com/belambert/asr_evaluation`
(The text file in the rawdata folder is created by this script)

After cloning the folder, use the following 4 lines can reproduce our experiment results

```
python Core.py prep youtube/rawdata/WER20.train youtube/CRFdata/WER20.train
python Core.py prep youtube/rawdata/WER20.test youtube/CRFdata/WER20.test
python Core.py train youtube/CRFdata/WER20.train youtube/model/WER20.crfsuite
python Core.py test youtube/model/WER20.crfsuite youtube/CRFdata/WER20.test
```

 | precision | recall | f1-score | support
--- | --- | --- | --- | ---
CORRECT | 0.85 | 1.00 | 0.92 | 92686
ERROR | 0.59 | 0.02 | 0.04 | 16837
avg / total | 0.81 | 0.85 | 0.78 | 109523

Notes:
This is Word Burst based model, in order to create the baseline model, three lines need to be commented out in the script (Please read Core.py)
