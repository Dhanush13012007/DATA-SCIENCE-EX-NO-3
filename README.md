## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df.head()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)  
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

df2=pd.concat([df2,enc],axis=1)
df2
pd.get_dummies(df2,columns=["nom_0"])

pip install --upgrade category_encoders

from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df

df.skew()
np.log(df["Highly Positive Skew"])
np.reciprocal(df["Moderate Positive Skew"])
np.sqrt(df["Highly Positive Skew"])
np.square(df["Highly Positive Skew"])
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()


```
<img width="369" height="176" alt="Screenshot 2026-05-25 211011" src="https://github.com/user-attachments/assets/4ff5aadf-8b75-426b-8788-a833ce542662" />
<img width="317" height="187" alt="Screenshot 2026-05-25 211037" src="https://github.com/user-attachments/assets/3fdc1914-80d7-469b-9ce8-2f72c7e91f70" />
<img width="379" height="295" alt="Screenshot 2026-05-25 211226" src="https://github.com/user-attachments/assets/7a4df64f-9f97-4b01-8b57-515abb2bdfaf" />
<img width="425" height="301" alt="Screenshot 2026-05-25 211245" src="https://github.com/user-attachments/assets/e4a11ed2-37ca-4e08-a0ed-687945f3dfdd" />
<img width="471" height="309" alt="Screenshot 2026-05-25 211300" src="https://github.com/user-attachments/assets/5c91ccdc-1a29-422d-9337-7230e1a1934c" />
<img width="639" height="320" alt="Screenshot 2026-05-25 211314" src="https://github.com/user-attachments/assets/0733fca0-d8f8-407e-ae46-8d5c05e2839f" />
<img width="1018" height="417" alt="Screenshot 2026-05-25 211335" src="https://github.com/user-attachments/assets/c4b860fd-e08f-472a-9bc1-da600076b3c0" />
<img width="1018" height="417" alt="Screenshot 2026-05-25 211335" src="https://github.com/user-attachments/assets/76ffee75-e902-41f0-8ab5-1b6bf51e33f2" />
<img width="666" height="319" alt="Screenshot 2026-05-25 211402" src="https://github.com/user-attachments/assets/1ac5fd2b-284f-4132-a7cd-a73a6fbbb1c2" />
<img width="601" height="304" alt="Screenshot 2026-05-25 211415" src="https://github.com/user-attachments/assets/c93268b6-ce31-4323-8c11-be9a9234070c" />
<img width="721" height="363" alt="Screenshot 2026-05-25 211426" src="https://github.com/user-attachments/assets/e4b83802-00f3-4514-a388-917ed6a332d4" />
<img width="363" height="99" alt="Screenshot 2026-05-25 211448" src="https://github.com/user-attachments/assets/de9aa654-db71-4f2a-9e25-912c70d649d4" />
<img width="558" height="215" alt="Screenshot 2026-05-25 211501" src="https://github.com/user-attachments/assets/4580050b-e8e8-4ba1-bd35-9f2641aea608" />
<img width="562" height="217" alt="Screenshot 2026-05-25 211516" src="https://github.com/user-attachments/assets/1bf794d1-2864-4ddc-8de9-1e438436b5f2" />
<img width="549" height="212" alt="Screenshot 2026-05-25 211530" src="https://github.com/user-attachments/assets/348db5b9-88b0-4245-8458-005deb7214da" />
<img width="624" height="226" alt="Screenshot 2026-05-25 211551" src="https://github.com/user-attachments/assets/404aa3e2-a8ac-44b7-9911-ae62b87ab346" />
<img width="923" height="350" alt="Screenshot 2026-05-25 211611" src="https://github.com/user-attachments/assets/3e1eeca7-a900-4973-bdb6-051ebf09e180" />
<img width="707" height="433" alt="Screenshot 2026-05-25 211619" src="https://github.com/user-attachments/assets/a5725257-d2aa-4b0e-893a-96afb2839b05" />
<img width="1061" height="372" alt="Screenshot 2026-05-25 211641" src="https://github.com/user-attachments/assets/f1943e34-f2e1-47f6-8d61-4a44d3a49ef5" />
<img width="606" height="430" alt="Screenshot 2026-05-25 211709" src="https://github.com/user-attachments/assets/91738c65-7805-4dfb-a044-bc8ad9410498" />
<img width="650" height="431" alt="Screenshot 2026-05-25 211719" src="https://github.com/user-attachments/assets/177cc82c-ced5-4f91-96ca-fbfa4481164f" />
<img width="709" height="429" alt="Screenshot 2026-05-25 211729" src="https://github.com/user-attachments/assets/934a429e-8206-4823-84d5-c549cea8ba9c" />

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
