import matplotlib.pyplot as plt
import pandas as pd
import json

with open("evaluation_results/resultsInstanceLevelSemnaticLabeling.json") as json_file:
            # print(json_file)
            results = json.load(json_file)

df = pd.DataFrame(results['averages']['classes'])
