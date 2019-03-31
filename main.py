from Utils import loadData
from PreProcessing import preProcessing
import pandas as pd

filename = "data/trainingObamaRomneytweets.xlsx"

for sheet in ['Obama','Romney']:
    df = loadData(filename,sheet)
    df = preProcessing(df).df