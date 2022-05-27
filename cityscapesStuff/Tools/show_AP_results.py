import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns

path = "/Store/travail/kajoda/CenterPoly/CenterPoly/exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/results/evaluationResults/"

with open(path + "resultInstanceLevelSemanticLabeling.json") as json_file:
            # print(json_file)
            results = json.load(json_file)

classes = results['averages']['classes']
data = []

data.append({'classe' : 'all', 'type' : 'ap', 'metric': results['averages']['allAp']})
data.append({'classe' : 'all', 'type' : 'ap50%', 'metric': results['averages']['allAp50%']})

for c in classes.keys() :
  data.append({'classe' : c, 'type' : 'ap', 'metric': classes[c]['ap']})
  data.append({'classe' : c, 'type' : 'ap50%', 'metric': classes[c]['ap50%']})


df = pd.DataFrame(data)
print(df.head)

ax = sns.catplot(data = df, kind = 'bar', x = 'classe', y = 'metric', hue = 'type')
plt.xticks(rotation=45)
plt.savefig(path + "results_classes.png")
plt.show()

