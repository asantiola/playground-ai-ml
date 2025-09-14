## Machine Learning Tutorial - Magic

### Youtube
- https://youtu.be/i_LwzRVP7bg?si=UJSEnIjzGOotcbAu

### Dataset
https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

### for the imports...
- pip install matplotlib
- pip install scikit-learn
- pip install imbalanced-learn

### FieldNames
- TLDR: ['fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class']
-Variable Name;Role;Type;Description;Units;Missing Values
-fLength;Feature;Continuous;major axis of ellipse;mm;no
-fWidth;Feature;Continuous;minor axis of ellipse;mm;no
-fSize;Feature;Continuous;10-log of sum of content of all pixels;#phot;no
-fConc;Feature;Continuous;ratio of sum of two highest pixels over fSize;;no
-fConc1;Feature;Continuous;ratio of highest pixel over fSize;;no
-fAsym;Feature;Continuous;distance from highest pixel to center, projected onto major axis;;no
-fM3Long;Feature;Continuous;3rd root of third moment along major axis;mm;no
-fM3Trans;Feature;Continuous;3rd root of third moment along minor axis;mm;no
-fAlpha;Feature;Continuous;angle of major axis with vector to origin;deg;no
-fDist;Feature;Continuous;distance from origin to center of ellipse;mm;no
-class;Target;Binary;gamma (signal), hadron (background);;no


### What is Machine Learning
- subdomain of computer science that focuses on algorithms ehich help computer learn from data without explicit programming
- AI - artificial intelligence - enable computer to perform human like tasks and to simulate human behavior.
- ML is subset of AI that tries to solve a specific problem and make predictions using data
- Data Science is a field that attempts to find patterns and draw insights from data

### Types of Learning
- Supervised Learning: uses labeled inputs to train models and learn output(s)
- Unsupervised Learning: uses unlabed data to learn about patterns in the data
- Reinforcement Learning: agent learning in interactive environment based on rewards and penalties.

- Focus of this class - Supervised Learning


### types of feature
- qualitative (finite number of categories or groups, e.g. Gender, Nationality, Location). No inherent order. Nominal data.
    - One-encoding
        - E.g. Nominal data:
        - USA = [1,0,0,0]
        - India = [0,1,0,0]
        - Canada = [0,0,1,0]
        - France = [0,0,0,1]
    - Ordinal data, e.g. level = worst - 1,2,3,4,5 - best
- quantitative (numerical, could be discrete or continuous)
    - e.g. Temperature, Length

### types of prediction
- Classification- predict discrete classes
    - binary classification
    - multiclass classification
- regression - prefict continuous values
    - e.g.: price of ethereum tomorrow, what would be the temperature
    - predict to a close as possible

### datasets
- training datasets: 80%
    - loss is fed back to the model
- validation datasets: 10%
    - loss is not fed back
- testing datasets: 10%

### Loss - difference between prediction and actual
- L1 Loss: loss = sum(abs(Yreal - Ypredicted))
    - abs of difference value of real vs predicted output
- L2 Loss: loss = sum((Yreal - Ypredicted) ** 2)
    - quadratic
- Binary Cross-Entropy Loss:
    - loss decreases as the performance gets better

### Models
- KNN: K-nearest neighbors ~ 46:00
    - look at whats around you and take label of whats the majority
    - say, K = 3, select a point, look for 3 closest points
    - closest point means Eucleadean Distance
