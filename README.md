## Medical Appointment No Shows

#### Link to dataset: https://www.kaggle.com/joniarroba/noshowappointments

### Introduction

- Many times people do not show up for a medical appointment. Previous studies have shown that about 25% of the people did not show up.
- No-show is a loss for doctors since they lose their payments. On the other hand, patients who wanted an appointment as soon as possible were unable to get one.
- Thus, there are two losses: the financial loss for the doctor and the loss of an appointment for the person in need.
- The notebook could help clinics and hospitals in understanding what attributes are associated with the individuals who did not show up.

## Project Description

- Performed predictive analysis on a large data-set of “Medical Appointment No Shows” to determine whether patient will show
up for appointment.
- Used <strong> Decision Tree, Random Forest Classifier, Light GBM, XGBoost models </strong> to determine best approach using Confusion Matrix.
- Analyzed data and plotted graphs using Interactive Visualizations (IPyWidgets, Plotly, Seaborn, Matplotlib).
- Utilized <strong>Python widgets</strong> to consolidate all the graphs in a <strong>drop-down menu style</strong>.
- Applied feature engineering which led to an increase in accuracy of the model from <strong>68%</strong> to <strong>76%</strong>.
### Data description
- <strong>PatientId</strong>: Identification of a patient
- <strong>AppointmentID</strong>: Identification of each appointment
- <strong>Gender</strong>: Male or Female . Female is the greater proportion, woman takes way more care of they health in comparison to man.
- <strong>ScheduledDay</strong>: The day the patient booked the appointment, this is before or at the same day of AppointmentDay of course.
- <strong>AppointmentDay</strong>: The day day the the apponitment is booked for
- <strong>Age</strong>: How old is the patient.
- <strong>Neighbourhood</strong>: Where the appointment takes place.
- <strong>Scholarship</strong>: True of False (A government financial aid to poor Brazilian families) https://en.wikipedia.org/wiki/Bolsa_Fam%C3%ADlia
- <strong>Hypertension</strong>: True or False (High blood pressure)
- <strong>Diabetes</strong>: True or False
- <strong>Alcoholism</strong>: True or False
- <strong>Handcap</strong>: 0-4 (the handcap refers to the number of disabilites a person has. For example, if the person is blind and can't walk the total is 2)
- <strong>SMS_received</strong>: 1 or more messages sent to the patient.
- <strong>No-show</strong>: True or False.



#### Link to dataset: https://www.kaggle.com/joniarroba/noshowappointments


### Loading Python libraries


```python
#Python Numpy and Pandas
import numpy as np
import pandas as pd

#Python Seaborn
import seaborn as sns

#Python Matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

#Plotly
import plotly.express as px

#Graphviz
from sklearn import tree
from sklearn.tree import export_graphviz
from graphviz import Source

#Python Widgets
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display

#Maximum rows to display
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
```

###  Loading Dataset


```python
df = pd.read_csv("../input/medical-appointment-no-shows.csv", converters={"PatientId":str})
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentID</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hipertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handcap</th>
      <th>SMS_received</th>
      <th>No-show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29T18:38:08Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29T16:08:27Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29T16:19:04Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29T17:29:31Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29T16:07:23Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (110527, 14)



##  <span style="color:blue">**Data Cleaning** </span>

#### Renaming Columns


```python
df.columns=['PatientId', 'AppointmentId', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hypertension',
       'Diabetes', 'Alcoholism', 'Handicap', 'SmsReceived', 'NoShow']
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29T18:38:08Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29T16:08:27Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29T16:19:04Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29T17:29:31Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29T16:07:23Z</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



####  Check for Null values in the dataset


```python
df.isna().sum()
```




    PatientId         0
    AppointmentId     0
    Gender            0
    ScheduledDay      0
    AppointmentDay    0
    Age               0
    Neighbourhood     0
    Scholarship       0
    Hypertension      0
    Diabetes          0
    Alcoholism        0
    Handicap          0
    SmsReceived       0
    NoShow            0
    dtype: int64



####  <span style="color:green">**As we see, there are no Null values in our dataset** </span>

#### Extracting date from Scheduled Day


```python
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"]) # COnverting Date from String type to Datetime format
df['ScheduledDate'] = df['ScheduledDay'].dt.date 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29T00:00:00Z</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
    </tr>
  </tbody>
</table>
</div>



#### Extracting date from Appointment Day


```python
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]) ##COnverting Date from String type to Datetime format
df['AppointmentDate'] = df['AppointmentDay'].dt.date
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
      <th>AppointmentDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 00:00:00+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
  </tbody>
</table>
</div>



#### Checking whether any Appointment days are before Scheduled days


```python
len(df[df["AppointmentDay"] < df["ScheduledDay"]])
```




    38568



####  <span style="color:green">**'AppointmentDay' has no value for time of the day and it has many values(38568) smaller than 'ScheduledDay' which is not possible !** </span>

####  <span style="color:green">**The reason for this problem is that probably these appointments happened at the same day that they're booked but because we don't have the exact hour for 'AppointmentDay' their difference is negative.** </span>

####  <span style="color:green">**To solve this problem, we added 23 hrs and 59 min and 59 secs to the 'AppointmentDay'. Now all 'AppointmentDay' are still at the same the same day but we only have 5 negative values for:** </span>
####  <span>**df["AppointmentDay"] - df["ScheduledDay"]** </span>


```python
df['AppointmentDay'] = df['AppointmentDay'] + pd.Timedelta('1d') - pd.Timedelta('1s')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
      <th>AppointmentDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df[ df.AppointmentDay < df.ScheduledDay ])
```




    5



####  <span style="color:green">**Now we have only 5 rows where ScheduledDay is later than AppointmentDay which can be dropped** </span>


```python
df.drop( df[df.AppointmentDay <= df.ScheduledDay].index, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
      <th>AppointmentDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
  </tbody>
</table>
</div>



#### Extracting day name from Scheduled day


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
      <th>AppointmentDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['ScheduledDay']=pd.to_datetime(df['ScheduledDay'])
df['ScheduledDayOfWeek']=df['ScheduledDay'].dt.date
df['ScheduledDayOfWeek']=pd.to_datetime(df['ScheduledDayOfWeek'])
df['ScheduledDayOfWeek'] = df['ScheduledDayOfWeek'].dt.day_name()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
      <th>AppointmentDate</th>
      <th>ScheduledDayOfWeek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
    </tr>
  </tbody>
</table>
</div>



#### Extracting day name from Appointment day


```python
df['AppointmentDay']=pd.to_datetime(df['AppointmentDay'])
df['AppointmentDayOfWeek']=df['AppointmentDay'].dt.date
df['AppointmentDayOfWeek']=pd.to_datetime(df['AppointmentDayOfWeek'])
df['AppointmentDayOfWeek'] = df['AppointmentDayOfWeek'].dt.day_name()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
      <th>AppointmentDate</th>
      <th>ScheduledDayOfWeek</th>
      <th>AppointmentDayOfWeek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
    </tr>
  </tbody>
</table>
</div>



#### Converting Categorical values to Numeric values


```python
d_replace = {"Yes": 1, "No": 0}
df = df.replace({"NoShow": d_replace})
```

#### Adding "Lead days" column


```python
df['LeadDays']=(df["AppointmentDay"] - df["ScheduledDay"]).astype('timedelta64[D]').astype(int)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
      <th>AppointmentDate</th>
      <th>ScheduledDayOfWeek</th>
      <th>AppointmentDayOfWeek</th>
      <th>LeadDays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



##  <span style="color:blue">**What are the major factors contributing to No Shows of patients ?** </span>

#### Interactive plots to compare number of No Shows in relation with various factors


```python
df_for_plots=df.copy()
```

#### Now, we classify age of the Patients into 5 categories


```python
def age_category(age):
    if(age<=15):
        return 1
    elif(age>15 and age<=30):
        return 2
    elif(age>30 and age<=45):
        return 3
    elif(age>45 and age<=60):
        return 4
    elif(age>60):
        return 5

df_for_plots['AgeCategory'] = df_for_plots.apply(lambda x: age_category(x['Age']), axis=1)
```

#### We classify Lead Days into 6 categories


```python
def lead_day_category(LeadDay):
    if(LeadDay<=2):
        return 'Within 2 Days'
    elif(LeadDay>2 and LeadDay<=7):
        return 'Within 1 Week'
    elif(LeadDay>7 and LeadDay<=14):
        return 'Within 2 Weeks'
    elif(LeadDay>14 and LeadDay<=21):
        return 'Within 3 Weeks'
    elif(LeadDay>21 and LeadDay<=28):
        return 'Within 4 Weeks'
    elif(LeadDay>28):
        return 'After 1 Month'

```

#### Creating a dataframe to calculate the percentage of people who miss their appointments, grouped by lead day category


```python
df_for_plots['LeadDayCategory'] = df_for_plots.apply(lambda x: lead_day_category(x['LeadDays']), axis=1)

df_lead_days = df_for_plots.groupby(by=['LeadDayCategory', 'NoShow'])['PatientId'].agg(['count']).rename(columns={'count':'LeadDayCount'})
df_lead_days.reset_index(inplace=True)
df_lead_days = df_lead_days[df_lead_days['NoShow']==1]

df_total = df_for_plots[['LeadDayCategory', 'PatientId']]
df_total = df_total.groupby(by='LeadDayCategory')['PatientId'].agg(['count']).rename(columns={'count':'Total'})
df_total.reset_index(inplace=True)

df_lead_days = df_lead_days.merge(df_total, how='left', left_on=['LeadDayCategory'], right_on=['LeadDayCategory'])

df_lead_days['Percent'] = round((df_lead_days['LeadDayCount']*100) / df_lead_days['Total'], 2)
df_lead_days
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LeadDayCategory</th>
      <th>NoShow</th>
      <th>LeadDayCount</th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>After 1 Month</td>
      <td>1</td>
      <td>3968</td>
      <td>12171</td>
      <td>32.60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Within 1 Week</td>
      <td>1</td>
      <td>5057</td>
      <td>20247</td>
      <td>24.98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Within 2 Days</td>
      <td>1</td>
      <td>4507</td>
      <td>50501</td>
      <td>8.92</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Within 2 Weeks</td>
      <td>1</td>
      <td>3664</td>
      <td>12025</td>
      <td>30.47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Within 3 Weeks</td>
      <td>1</td>
      <td>2861</td>
      <td>8874</td>
      <td>32.24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Within 4 Weeks</td>
      <td>1</td>
      <td>2257</td>
      <td>6704</td>
      <td>33.67</td>
    </tr>
  </tbody>
</table>
</div>



#### We subsitute the numerical values in column 'SmsReceived' to categorical values


```python
df_for_plots['SmsReceived'] = df_for_plots['SmsReceived'].map({0:'SMS not received', 1:'SMS received'})
```

#### We consider 'Hypertension' and 'Diabetes' as Chronic illness and we classify Patients into 2 categories:
- Patients belonging to Age Category 5 and suffering from Chronic illness
- Patients belonging to Age Category 1-4 and suffering from Chronic illness


```python
df_for_plots.loc[((df_for_plots['Hypertension'] == 1) | (df_for_plots['Diabetes'] == 1)) & (df_for_plots['AgeCategory']==5), 'Old_Chronic'] = 'OldChronic'
df_for_plots.loc[((df_for_plots['Hypertension'] == 1) | (df_for_plots['Diabetes'] == 1)) & (df_for_plots['AgeCategory']!=5), 'Old_Chronic'] = 'YoungChronic'

df_for_plots['Old_Chronic'] = df_for_plots['Old_Chronic'].fillna('Healthier')

df_for_plots.head()

Pct_Old_Chronic = round((df_for_plots[(df_for_plots['Old_Chronic']=='OldChronic') & (df_for_plots['NoShow']==1)]['PatientId'].count() \
/ df_for_plots[df_for_plots['NoShow']==1]['PatientId'].count()) * 100, 2)

Pct_Young_Chronic = round((df_for_plots[(df_for_plots['Old_Chronic']=='YoungChronic') & (df_for_plots['NoShow']==1)]['PatientId'].count() \
/ df_for_plots[df_for_plots['NoShow']==1]['PatientId'].count()) * 100, 2)

print("Patients belonging to Age Category 5 and suffering from Chronic illness: "+str(Pct_Old_Chronic))
print("Patients belonging to Age Category 1-4 and suffering from Chronic illness: "+str(Pct_Young_Chronic))
```

    Patients belonging to Age Category 5 and suffering from Chronic illness: 8.2
    Patients belonging to Age Category 1-4 and suffering from Chronic illness: 10.0
    

### We make use of Python Widgets to consolidate all our Plots in one cell and give freedom to the user to select whichever Plot he wants to see.


```python
size=15
params = {'legend.fontsize': 'large',
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)


@widgets.interact_manual(Plot_Name=['-- Select a Plot --','No Shows as per Age Category of Patients', 
                                'No Shows as per Neighbourhood',
                                'No Shows as per Appointment Day',
                                'No Shows as per number of Disabilities',
                                'Effect of an SMS notification on No Shows',
                                'Distribution of No Shows as per Number of Disabilities and Age Category',
                                'Distribution of No Shows as per Number of Chronic Illness',
                                'Effect of Waiting time on No Show'])

def plot(Plot_Name):
    if Plot_Name=='No Shows as per Neighbourhood':
        df_for_plots["All"]="Brazil"
        fig = px.treemap(data_frame=df_for_plots,path=['All','Neighbourhood','Gender'],
                         values='NoShow',title='No Shows as per Neighbourhood')
        fig.show()
        
    elif Plot_Name=='Distribution of No Shows as per Number of Disabilities and Age Category':
        sns.catplot(x='AgeCategory', y='NoShow', hue='Handicap', data=df_for_plots, kind='bar',aspect=3)
        plt.title("Distribution of No Shows as per Number of Disabilities and Age Category")
        
    elif Plot_Name=='No Shows as per Age Category of Patients':
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        sns.barplot(x=df_for_plots['AgeCategory'],y=df_for_plots['NoShow'],hue=df_for_plots['Gender'],palette='Blues')
        plt.title("No Shows as per Age Category of Patients")
        
    elif Plot_Name=='No Shows as per Appointment Day':
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        sns.barplot(x=df_for_plots['AppointmentDayOfWeek'],y=df_for_plots['NoShow']*100,palette='ocean')
        plt.title("No Shows as per Appointment Day")
        
    elif Plot_Name=='No Shows as per number of Disabilities':
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        sns.barplot(x=df_for_plots['Handicap'],y=df_for_plots['NoShow']*100,palette='ocean')   
        plt.xlabel("No. of Disabilities")
        plt.ylabel("% of NoShow")
        plt.title("No Shows as per number of Disabilities")
        
    elif Plot_Name=='Distribution of No Shows as per Number of Chronic Illness':
        x = [8.19,10.0]
        years = ['Senior Citizen with Chronic Illness', 'Young with Chronic Illness']
        plt.figure(figsize=(16,9))
        plt.bar(years, x,color='Purple')
        plt.ylabel("% of NoShow")
        plt.title("Distribution of No Shows as per Number of Chronic Illness")
        plt.show()
        
    elif Plot_Name=='Effect of an SMS notification on No Shows':
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        sns.barplot(x=df_for_plots['NoShow']*100,y=df_for_plots['SmsReceived'],palette='pastel',orient='h') 
        plt.xlabel("% of NoShow")
        plt.title("Effect of an SMS notification on No Shows")
        plt.ylabel('SMS Notification')
        
    elif Plot_Name=='Effect of Waiting time on No Show':
        sns.lineplot(x=df_lead_days['LeadDayCategory'],y=df_lead_days['Percent'])
        plt.title("Effect of Waiting time on No Show")
        plt.xlabel('Lead Day Category')
        plt.ylabel('% No Show')
    else:
        print()
```


    interactive(children=(Dropdown(description='Plot_Name', options=('-- Select a Plot --', 'No Shows as per Age C…


## <span style="color:green"> Conclusion: </span>
#### - Patients with Age category 2 are more likely to miss their appointments
#### - In all of the top 10 Neighbourhoods with highest No-Shows, the proportion of females not showing up is more than males.
#### - Patients are most likely to show up if the day of their appointment is Saturday.
#### - Patients having Number of disabilities as 4 are are the one's with highest no-shows.
#### - Patients receiving an SMS Notification are more likely to not show-up for an appointment as compared to the one's not receiving an SMS notification. Hospitals needs to further assess their strategy about notifying patients for their appointments.
#### - Patients belonging to Age category 1 and having Number of disabilities as 3 and 4 are more likely to not show up.
#### - Younger patients (with age category between 1-4) suffering from Chronic illness(Diabetes and Hypertension) constitute a higher proportion of no-shows as compared to senior citizens (with age category 5) suffering from chronic illness.

# <span style="color:blue">**Machine Leaning**</span>


##  <span style="color:blue">**Decision Tree** </span>

#### Let's make a Decision Tree based on only the original columns


```python
df_for_decision_tree=df.copy()
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
      <th>AppointmentDate</th>
      <th>ScheduledDayOfWeek</th>
      <th>AppointmentDayOfWeek</th>
      <th>LeadDays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29872499824296</td>
      <td>5642903</td>
      <td>F</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>558997776694438</td>
      <td>5642503</td>
      <td>M</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4262962299951</td>
      <td>5642549</td>
      <td>F</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>867951213174</td>
      <td>5642828</td>
      <td>F</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8841186448183</td>
      <td>5642494</td>
      <td>F</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-29</td>
      <td>2016-04-29</td>
      <td>Friday</td>
      <td>Friday</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### We extract the Original columns from modified dataset to check the accuracy of Decision Tree algorithm


```python
df_for_decision_tree = df[['NoShow', 'Gender', 'Age', 'Neighbourhood', 'Scholarship', 'Hypertension',
       'Diabetes', 'Alcoholism', 'Handicap', 'SmsReceived', 'ScheduledDayOfWeek',\
        'ScheduledDay','AppointmentDay','AppointmentDayOfWeek']]
```


```python
df_for_decision_tree['ScheduledDay']=df_for_decision_tree['ScheduledDay'].dt.day
df_for_decision_tree['AppointmentDay']=df_for_decision_tree['AppointmentDay'].dt.day
```


```python
df_for_decision_tree.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NoShow</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>ScheduledDayOfWeek</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>AppointmentDayOfWeek</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>F</td>
      <td>62</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>29</td>
      <td>29</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>M</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>29</td>
      <td>29</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>F</td>
      <td>62</td>
      <td>MATA DA PRAIA</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>29</td>
      <td>29</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>F</td>
      <td>8</td>
      <td>PONTAL DE CAMBURI</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>29</td>
      <td>29</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>F</td>
      <td>56</td>
      <td>JARDIM DA PENHA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>29</td>
      <td>29</td>
      <td>Friday</td>
    </tr>
  </tbody>
</table>
</div>



#### We convert categorical values to dummies


```python
df_for_decision_tree = pd.get_dummies(df_for_decision_tree, columns=['Neighbourhood','AppointmentDayOfWeek','ScheduledDayOfWeek','Gender'])

```


```python
df_for_decision_tree.shape
```




    (110522, 105)




```python
pip install pydotplus
```

    Collecting pydotplus
      Downloading pydotplus-2.0.2.tar.gz (278 kB)
    [K     |████████████████████████████████| 278 kB 2.8 MB/s eta 0:00:01
    [?25hRequirement already satisfied: pyparsing>=2.0.1 in /opt/conda/lib/python3.7/site-packages (from pydotplus) (2.4.7)
    Building wheels for collected packages: pydotplus
      Building wheel for pydotplus (setup.py) ... [?25ldone
    [?25h  Created wheel for pydotplus: filename=pydotplus-2.0.2-py3-none-any.whl size=24566 sha256=7981f54b48290de6507f056decfbb4ac4f32e397065a33a8c3aef900506e9d52
      Stored in directory: /root/.cache/pip/wheels/1e/7b/04/7387cf6cc9e48b4a96e361b0be812f0708b394b821bf8c9c50
    Successfully built pydotplus
    Installing collected packages: pydotplus
    Successfully installed pydotplus-2.0.2
    Note: you may need to restart the kernel to use updated packages.
    


```python
import sklearn as sk
import sklearn.tree as tree
from IPython.display import Image  
import pydotplus
from IPython.display import SVG
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,f1_score
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
X = df_for_decision_tree.drop('NoShow',axis=1)
Y = df_for_decision_tree['NoShow']
dt = tree.DecisionTreeClassifier(max_depth=2).fit(X,Y)
```


```python
dt_feature_names = list(X.columns)
dt_target_names = [str(s) for s in Y.unique()]
```


```python
graph = Source(tree.export_graphviz(dt, out_file=None
  , feature_names=dt_feature_names, class_names=dt_target_names
  , filled = True))
display(SVG(graph.pipe(format='svg')))
```


    
![svg](output_70_0.svg)
    


#### The decision tree classifier classifies the data points into the same category, although our dataset isn't skewed. Hence moving onto the next model

### Random Forest Classifier on the original columns


```python
df_for_random_forest = df[['NoShow', 'Gender', 'Age', 'Neighbourhood', 'Scholarship', 'Hypertension','Diabetes', \
                           'Alcoholism', 'Handicap', 'SmsReceived', 'ScheduledDayOfWeek','ScheduledDay',\
                           'AppointmentDay','AppointmentDayOfWeek']]
```


```python
df_for_random_forest = pd.get_dummies(df_for_random_forest,columns=['Neighbourhood', 'AppointmentDayOfWeek'])
df_for_random_forest.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NoShow</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>ScheduledDayOfWeek</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Neighbourhood_AEROPORTO</th>
      <th>Neighbourhood_ANDORINHAS</th>
      <th>Neighbourhood_ANTÔNIO HONÓRIO</th>
      <th>Neighbourhood_ARIOVALDO FAVALESSA</th>
      <th>Neighbourhood_BARRO VERMELHO</th>
      <th>Neighbourhood_BELA VISTA</th>
      <th>Neighbourhood_BENTO FERREIRA</th>
      <th>Neighbourhood_BOA VISTA</th>
      <th>Neighbourhood_BONFIM</th>
      <th>Neighbourhood_CARATOÍRA</th>
      <th>Neighbourhood_CENTRO</th>
      <th>Neighbourhood_COMDUSA</th>
      <th>Neighbourhood_CONQUISTA</th>
      <th>Neighbourhood_CONSOLAÇÃO</th>
      <th>Neighbourhood_CRUZAMENTO</th>
      <th>Neighbourhood_DA PENHA</th>
      <th>Neighbourhood_DE LOURDES</th>
      <th>Neighbourhood_DO CABRAL</th>
      <th>Neighbourhood_DO MOSCOSO</th>
      <th>Neighbourhood_DO QUADRO</th>
      <th>Neighbourhood_ENSEADA DO SUÁ</th>
      <th>Neighbourhood_ESTRELINHA</th>
      <th>Neighbourhood_FONTE GRANDE</th>
      <th>Neighbourhood_FORTE SÃO JOÃO</th>
      <th>Neighbourhood_FRADINHOS</th>
      <th>Neighbourhood_GOIABEIRAS</th>
      <th>Neighbourhood_GRANDE VITÓRIA</th>
      <th>Neighbourhood_GURIGICA</th>
      <th>Neighbourhood_HORTO</th>
      <th>Neighbourhood_ILHA DAS CAIEIRAS</th>
      <th>Neighbourhood_ILHA DE SANTA MARIA</th>
      <th>Neighbourhood_ILHA DO BOI</th>
      <th>Neighbourhood_ILHA DO FRADE</th>
      <th>Neighbourhood_ILHA DO PRÍNCIPE</th>
      <th>Neighbourhood_ILHAS OCEÂNICAS DE TRINDADE</th>
      <th>Neighbourhood_INHANGUETÁ</th>
      <th>Neighbourhood_ITARARÉ</th>
      <th>Neighbourhood_JABOUR</th>
      <th>Neighbourhood_JARDIM CAMBURI</th>
      <th>Neighbourhood_JARDIM DA PENHA</th>
      <th>Neighbourhood_JESUS DE NAZARETH</th>
      <th>Neighbourhood_JOANA D´ARC</th>
      <th>Neighbourhood_JUCUTUQUARA</th>
      <th>Neighbourhood_MARIA ORTIZ</th>
      <th>Neighbourhood_MARUÍPE</th>
      <th>Neighbourhood_MATA DA PRAIA</th>
      <th>Neighbourhood_MONTE BELO</th>
      <th>Neighbourhood_MORADA DE CAMBURI</th>
      <th>Neighbourhood_MÁRIO CYPRESTE</th>
      <th>Neighbourhood_NAZARETH</th>
      <th>Neighbourhood_NOVA PALESTINA</th>
      <th>Neighbourhood_PARQUE INDUSTRIAL</th>
      <th>Neighbourhood_PARQUE MOSCOSO</th>
      <th>Neighbourhood_PIEDADE</th>
      <th>Neighbourhood_PONTAL DE CAMBURI</th>
      <th>Neighbourhood_PRAIA DO CANTO</th>
      <th>Neighbourhood_PRAIA DO SUÁ</th>
      <th>Neighbourhood_REDENÇÃO</th>
      <th>Neighbourhood_REPÚBLICA</th>
      <th>Neighbourhood_RESISTÊNCIA</th>
      <th>Neighbourhood_ROMÃO</th>
      <th>Neighbourhood_SANTA CECÍLIA</th>
      <th>Neighbourhood_SANTA CLARA</th>
      <th>Neighbourhood_SANTA HELENA</th>
      <th>Neighbourhood_SANTA LUÍZA</th>
      <th>Neighbourhood_SANTA LÚCIA</th>
      <th>Neighbourhood_SANTA MARTHA</th>
      <th>Neighbourhood_SANTA TEREZA</th>
      <th>Neighbourhood_SANTO ANDRÉ</th>
      <th>Neighbourhood_SANTO ANTÔNIO</th>
      <th>Neighbourhood_SANTOS DUMONT</th>
      <th>Neighbourhood_SANTOS REIS</th>
      <th>Neighbourhood_SEGURANÇA DO LAR</th>
      <th>Neighbourhood_SOLON BORGES</th>
      <th>Neighbourhood_SÃO BENEDITO</th>
      <th>Neighbourhood_SÃO CRISTÓVÃO</th>
      <th>Neighbourhood_SÃO JOSÉ</th>
      <th>Neighbourhood_SÃO PEDRO</th>
      <th>Neighbourhood_TABUAZEIRO</th>
      <th>Neighbourhood_UNIVERSITÁRIO</th>
      <th>Neighbourhood_VILA RUBIM</th>
      <th>AppointmentDayOfWeek_Friday</th>
      <th>AppointmentDayOfWeek_Monday</th>
      <th>AppointmentDayOfWeek_Saturday</th>
      <th>AppointmentDayOfWeek_Thursday</th>
      <th>AppointmentDayOfWeek_Tuesday</th>
      <th>AppointmentDayOfWeek_Wednesday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>F</td>
      <td>62</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>2016-04-29 18:38:08+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>M</td>
      <td>56</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>2016-04-29 16:08:27+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>F</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>2016-04-29 16:19:04+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>F</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>2016-04-29 17:29:31+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>F</td>
      <td>56</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Friday</td>
      <td>2016-04-29 16:07:23+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Converting categorical values of the Gender and ScheduledDayOfWeek into numeric values


```python
df_for_random_forest['Gender'] = df_for_random_forest['Gender'].map({'M':1, 'F':0})
df_for_random_forest['ScheduledDayOfWeek'] = df_for_random_forest['ScheduledDayOfWeek'].map({'Monday':0, 'Tuesday':1,\
                                                                                             'Wednesday':2, 'Thursday':3,\
                                                                                             'Friday':4, 'Saturday':5})
```

As we have already extracted the date values from the ScheduledDay and AppointmentDay columns, dropping them as we do not need the timestamp. 


```python
df_for_random_forest.drop(['ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)
```


```python
target = 'NoShow'
predictors = df_for_random_forest.columns[1:]
```


```python
from sklearn.model_selection import train_test_split

X = df_for_random_forest[predictors]
Y = df_for_random_forest[target]

X_train, X_test, Y_train, Y_test= \
train_test_split(X,Y,test_size=0.3,random_state = 5)
```


```python
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as met

clf_rf = RandomForestClassifier(random_state = 5, n_estimators=100)
clf_rf.fit(X_train,Y_train)


y_pred_rf = clf_rf.predict(X_test)
y_pred_proba_rf = clf_rf.predict_proba(X_test)[:,1]
```

Accuracy-


```python
round(met.accuracy_score(Y_test, y_pred_rf), 4)
```




    0.7618



AUC Score-


```python
round(met.roc_auc_score(Y_test, y_pred_proba_rf), 4)
```




    0.6615



# **AutoML Result**

Now we want to try auto ml H2o:

First, we need to add the dataset into the h2o driverless app,

then split the data into train and test,

then set the accuracey, time and interpretability as you can see in the screenshot below

and finaly, we run the model.

After the calculation is done, you can see that the best model is LightGBM and AUC is 0.7403


```python
Image(url= "https://i.ibb.co/6P1Wj26/H2-O-Appointment.png")
```




<img src="https://i.ibb.co/6P1Wj26/H2-O-Appointment.png"/>



## H2o gives the AUC of 0.7403
We want to make a more accurate model

We're going to create a model to produce better AUC:

First we try creating new columns(feature engineering)<br>
Then we try different models to get the best AUC possibel(hyper parameter tuning)

##  <span style="color:blue">**Feature Engineering** </span>

#### Added following new columns to extract more information from our dataset

- <strong>Prior No-Show Rate:</strong> No-shows as a percentage of total appointments (hypothesis: some patients persistently miss their appointments)
- <strong>Status of the last appointment:</strong> (hypothesis: if you miss your last appointment, you are more likely to attend the next one)
- <strong>Number of previous appointments</strong> (hypothesis: patients with persistent conditions are more likely to attend)
- <strong>Days since last appointment: </strong> Difference in days between the last 2 appointments

Note-<br>
The following cell takes a while to run


```python
%%time

df = df.sort_values(["PatientId","AppointmentDate"])
l_no_show_rates = []
l_last_shows = []
l_appts_counts = []
l_appts_lasts = []

for pat in df["PatientId"].unique():
    dfx = df[df["PatientId"] == pat]
    l_no_show = list(dfx["NoShow"])
    
    l_no_show_rate = [0]
    l_appts_last = [0]
    for i in range(1,len(dfx)):
        # no show rates
        rate_so_far = round((np.sum(l_no_show[0:i]) / i)*100,1)
        l_no_show_rate.append(rate_so_far)
        # appointments since last appointment
        dates_list = list(dfx["AppointmentDate"])
        l_appt_diff = (dates_list[i] - dates_list[i-1]).days
        l_appts_last.append(l_appt_diff)
        
    # appointment count
    l_appts_count = []
    for i in range(0,len(dfx)):
        l_appts = i+1
        l_appts_count.append(l_appts)
        
    l_no_show_rates.extend(l_no_show_rate)
    l_appts_counts.extend(l_appts_count)
    l_appts_lasts.extend(l_appts_last)
    
    # last show
    l_last_show = []
    l_last_show = [0] + l_no_show[:-1]
    l_last_shows.extend(l_last_show)   

df["NoShowRate"] = l_no_show_rates
df["LastShowStatus"] = l_last_shows
df["AppointmentCount"] = l_appts_counts
df["LastAppointmentDays"] = l_appts_lasts

df.head()
```

    CPU times: user 22min 7s, sys: 246 ms, total: 22min 7s
    Wall time: 22min 8s
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PatientId</th>
      <th>AppointmentId</th>
      <th>Gender</th>
      <th>ScheduledDay</th>
      <th>AppointmentDay</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>NoShow</th>
      <th>ScheduledDate</th>
      <th>AppointmentDate</th>
      <th>ScheduledDayOfWeek</th>
      <th>AppointmentDayOfWeek</th>
      <th>LeadDays</th>
      <th>NoShowRate</th>
      <th>LastShowStatus</th>
      <th>AppointmentCount</th>
      <th>LastAppointmentDays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84473</th>
      <td>11111462625267</td>
      <td>5715720</td>
      <td>F</td>
      <td>2016-05-18 14:58:29+00:00</td>
      <td>2016-06-08 23:59:59+00:00</td>
      <td>65</td>
      <td>REPÚBLICA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2016-05-18</td>
      <td>2016-06-08</td>
      <td>Wednesday</td>
      <td>Wednesday</td>
      <td>21</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51059</th>
      <td>111124532532143</td>
      <td>5531224</td>
      <td>M</td>
      <td>2016-03-31 09:17:26+00:00</td>
      <td>2016-05-03 23:59:59+00:00</td>
      <td>9</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2016-03-31</td>
      <td>2016-05-03</td>
      <td>Thursday</td>
      <td>Tuesday</td>
      <td>33</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95359</th>
      <td>111124532532143</td>
      <td>5624030</td>
      <td>M</td>
      <td>2016-04-26 15:05:58+00:00</td>
      <td>2016-06-01 23:59:59+00:00</td>
      <td>9</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2016-04-26</td>
      <td>2016-06-01</td>
      <td>Tuesday</td>
      <td>Wednesday</td>
      <td>36</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
      <td>29</td>
    </tr>
    <tr>
      <th>60744</th>
      <td>11114485119737</td>
      <td>5621757</td>
      <td>F</td>
      <td>2016-04-26 10:46:38+00:00</td>
      <td>2016-05-25 23:59:59+00:00</td>
      <td>12</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2016-04-26</td>
      <td>2016-05-25</td>
      <td>Tuesday</td>
      <td>Wednesday</td>
      <td>29</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2321</th>
      <td>11116239871275</td>
      <td>5625265</td>
      <td>F</td>
      <td>2016-04-27 07:05:38+00:00</td>
      <td>2016-04-29 23:59:59+00:00</td>
      <td>13</td>
      <td>SÃO PEDRO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-04-27</td>
      <td>2016-04-29</td>
      <td>Wednesday</td>
      <td>Friday</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv("after_preprocessing.csv")
```


```python
df['ScheduledDayDay'] = df['ScheduledDay'].dt.day
df['AppointmentDayDay'] = df['AppointmentDay'].dt.day
```

Now we create a new dataframe with these new columns


```python
df_ml = df[['NoShow', 'Gender', 'Age', 'Neighbourhood', 'Scholarship', 'Hypertension','Diabetes', 'Alcoholism', \
            'Handicap', 'SmsReceived','LeadDays', 'ScheduledDayOfWeek','ScheduledDayDay','AppointmentDayDay',\
            'AppointmentDayOfWeek','NoShowRate','LastShowStatus', 'AppointmentCount', 'LastAppointmentDays']]
```


```python
df_ml.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NoShow</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Neighbourhood</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>LeadDays</th>
      <th>ScheduledDayOfWeek</th>
      <th>ScheduledDayDay</th>
      <th>AppointmentDayDay</th>
      <th>AppointmentDayOfWeek</th>
      <th>NoShowRate</th>
      <th>LastShowStatus</th>
      <th>AppointmentCount</th>
      <th>LastAppointmentDays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84473</th>
      <td>0</td>
      <td>F</td>
      <td>65</td>
      <td>REPÚBLICA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>21</td>
      <td>Wednesday</td>
      <td>18</td>
      <td>8</td>
      <td>Wednesday</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51059</th>
      <td>0</td>
      <td>M</td>
      <td>9</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>33</td>
      <td>Thursday</td>
      <td>31</td>
      <td>3</td>
      <td>Tuesday</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95359</th>
      <td>1</td>
      <td>M</td>
      <td>9</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>36</td>
      <td>Tuesday</td>
      <td>26</td>
      <td>1</td>
      <td>Wednesday</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
      <td>29</td>
    </tr>
    <tr>
      <th>60744</th>
      <td>1</td>
      <td>F</td>
      <td>12</td>
      <td>MARIA ORTIZ</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>29</td>
      <td>Tuesday</td>
      <td>26</td>
      <td>25</td>
      <td>Wednesday</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2321</th>
      <td>0</td>
      <td>F</td>
      <td>13</td>
      <td>SÃO PEDRO</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>Wednesday</td>
      <td>27</td>
      <td>29</td>
      <td>Friday</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Converting categorical values of the Gender, AppointmentDayOfWeek and ScheduledDayOfWeek into numeric values


```python
df_ml['Gender'] = df_ml['Gender'].map({'M':1, 'F':0})
df_ml['ScheduledDayOfWeek'] = df_ml['ScheduledDayOfWeek'].map({'Monday':0, 'Tuesday':1, 'Wednesday':2, \
                                                               'Thursday':3, 'Friday':4, 'Saturday':5})
df_ml['AppointmentDayOfWeek'] = df_ml['AppointmentDayOfWeek'].map({'Monday':0, 'Tuesday':1, 'Wednesday':2, \
                                                               'Thursday':3, 'Friday':4, 'Saturday':5})
```


```python
df_ml = pd.get_dummies(df_ml,columns=['Neighbourhood'])
df_ml
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NoShow</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Scholarship</th>
      <th>Hypertension</th>
      <th>Diabetes</th>
      <th>Alcoholism</th>
      <th>Handicap</th>
      <th>SmsReceived</th>
      <th>LeadDays</th>
      <th>ScheduledDayOfWeek</th>
      <th>ScheduledDayDay</th>
      <th>AppointmentDayDay</th>
      <th>AppointmentDayOfWeek</th>
      <th>NoShowRate</th>
      <th>LastShowStatus</th>
      <th>AppointmentCount</th>
      <th>LastAppointmentDays</th>
      <th>Neighbourhood_AEROPORTO</th>
      <th>Neighbourhood_ANDORINHAS</th>
      <th>Neighbourhood_ANTÔNIO HONÓRIO</th>
      <th>Neighbourhood_ARIOVALDO FAVALESSA</th>
      <th>Neighbourhood_BARRO VERMELHO</th>
      <th>Neighbourhood_BELA VISTA</th>
      <th>Neighbourhood_BENTO FERREIRA</th>
      <th>Neighbourhood_BOA VISTA</th>
      <th>Neighbourhood_BONFIM</th>
      <th>Neighbourhood_CARATOÍRA</th>
      <th>Neighbourhood_CENTRO</th>
      <th>Neighbourhood_COMDUSA</th>
      <th>Neighbourhood_CONQUISTA</th>
      <th>Neighbourhood_CONSOLAÇÃO</th>
      <th>Neighbourhood_CRUZAMENTO</th>
      <th>Neighbourhood_DA PENHA</th>
      <th>Neighbourhood_DE LOURDES</th>
      <th>Neighbourhood_DO CABRAL</th>
      <th>Neighbourhood_DO MOSCOSO</th>
      <th>Neighbourhood_DO QUADRO</th>
      <th>Neighbourhood_ENSEADA DO SUÁ</th>
      <th>Neighbourhood_ESTRELINHA</th>
      <th>Neighbourhood_FONTE GRANDE</th>
      <th>Neighbourhood_FORTE SÃO JOÃO</th>
      <th>Neighbourhood_FRADINHOS</th>
      <th>Neighbourhood_GOIABEIRAS</th>
      <th>Neighbourhood_GRANDE VITÓRIA</th>
      <th>Neighbourhood_GURIGICA</th>
      <th>Neighbourhood_HORTO</th>
      <th>Neighbourhood_ILHA DAS CAIEIRAS</th>
      <th>Neighbourhood_ILHA DE SANTA MARIA</th>
      <th>Neighbourhood_ILHA DO BOI</th>
      <th>Neighbourhood_ILHA DO FRADE</th>
      <th>Neighbourhood_ILHA DO PRÍNCIPE</th>
      <th>Neighbourhood_ILHAS OCEÂNICAS DE TRINDADE</th>
      <th>Neighbourhood_INHANGUETÁ</th>
      <th>Neighbourhood_ITARARÉ</th>
      <th>Neighbourhood_JABOUR</th>
      <th>Neighbourhood_JARDIM CAMBURI</th>
      <th>Neighbourhood_JARDIM DA PENHA</th>
      <th>Neighbourhood_JESUS DE NAZARETH</th>
      <th>Neighbourhood_JOANA D´ARC</th>
      <th>Neighbourhood_JUCUTUQUARA</th>
      <th>Neighbourhood_MARIA ORTIZ</th>
      <th>Neighbourhood_MARUÍPE</th>
      <th>Neighbourhood_MATA DA PRAIA</th>
      <th>Neighbourhood_MONTE BELO</th>
      <th>Neighbourhood_MORADA DE CAMBURI</th>
      <th>Neighbourhood_MÁRIO CYPRESTE</th>
      <th>Neighbourhood_NAZARETH</th>
      <th>Neighbourhood_NOVA PALESTINA</th>
      <th>Neighbourhood_PARQUE INDUSTRIAL</th>
      <th>Neighbourhood_PARQUE MOSCOSO</th>
      <th>Neighbourhood_PIEDADE</th>
      <th>Neighbourhood_PONTAL DE CAMBURI</th>
      <th>Neighbourhood_PRAIA DO CANTO</th>
      <th>Neighbourhood_PRAIA DO SUÁ</th>
      <th>Neighbourhood_REDENÇÃO</th>
      <th>Neighbourhood_REPÚBLICA</th>
      <th>Neighbourhood_RESISTÊNCIA</th>
      <th>Neighbourhood_ROMÃO</th>
      <th>Neighbourhood_SANTA CECÍLIA</th>
      <th>Neighbourhood_SANTA CLARA</th>
      <th>Neighbourhood_SANTA HELENA</th>
      <th>Neighbourhood_SANTA LUÍZA</th>
      <th>Neighbourhood_SANTA LÚCIA</th>
      <th>Neighbourhood_SANTA MARTHA</th>
      <th>Neighbourhood_SANTA TEREZA</th>
      <th>Neighbourhood_SANTO ANDRÉ</th>
      <th>Neighbourhood_SANTO ANTÔNIO</th>
      <th>Neighbourhood_SANTOS DUMONT</th>
      <th>Neighbourhood_SANTOS REIS</th>
      <th>Neighbourhood_SEGURANÇA DO LAR</th>
      <th>Neighbourhood_SOLON BORGES</th>
      <th>Neighbourhood_SÃO BENEDITO</th>
      <th>Neighbourhood_SÃO CRISTÓVÃO</th>
      <th>Neighbourhood_SÃO JOSÉ</th>
      <th>Neighbourhood_SÃO PEDRO</th>
      <th>Neighbourhood_TABUAZEIRO</th>
      <th>Neighbourhood_UNIVERSITÁRIO</th>
      <th>Neighbourhood_VILA RUBIM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84473</th>
      <td>0</td>
      <td>0</td>
      <td>65</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>21</td>
      <td>2</td>
      <td>18</td>
      <td>8</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51059</th>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>33</td>
      <td>3</td>
      <td>31</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95359</th>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>36</td>
      <td>1</td>
      <td>26</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60744</th>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>29</td>
      <td>1</td>
      <td>26</td>
      <td>25</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2321</th>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>27</td>
      <td>29</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>46976</th>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>35</td>
      <td>2</td>
      <td>20</td>
      <td>25</td>
      <td>2</td>
      <td>100.0</td>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>76224</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1641</th>
      <td>1</td>
      <td>0</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26</td>
      <td>29</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31366</th>
      <td>1</td>
      <td>0</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>14</td>
      <td>2</td>
      <td>20</td>
      <td>4</td>
      <td>2</td>
      <td>100.0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87707</th>
      <td>0</td>
      <td>1</td>
      <td>59</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>25</td>
      <td>3</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>110522 rows × 99 columns</p>
</div>




```python
df_ml.shape
```




    (110522, 99)




```python
target = 'NoShow'
```


```python
predictors = df_ml.columns[1:]
```

### RandomForest Classifier

We run Random Forest Classifier on the new Data Set


```python
from sklearn.model_selection import train_test_split

X = df_ml[predictors]
Y = df_ml[target]


X_train, X_test, Y_train, Y_test= \
train_test_split(X,Y,test_size=0.3,random_state = 0)
```


```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state = 5, n_estimators=100)
clf.fit(X_train,Y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=5, verbose=0,
                           warm_start=False)




```python
y_pred = clf.predict(X_test)
```


```python
y_pred_proba = clf.predict_proba(X_test)[:,1]
```


```python
from sklearn.metrics import confusion_matrix
import sklearn.metrics as met

confusion_matrix(Y_test,y_pred)
```




    array([[25523,   856],
           [ 5697,  1081]])



Accuracy-


```python
round(met.accuracy_score(Y_test, y_pred), 4)
```




    0.8024



AUC Score-


```python
round(met.roc_auc_score(Y_test, y_pred_proba), 4)
```




    0.7441




```python
l=[]
for feature in zip(predictors, clf.feature_importances_):
    l.append(feature)
    
df_importance = pd.DataFrame(l,columns=["Variable","Importance"])\
.sort_values("Importance", ascending=False)

df_importance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>1.691654e-01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LeadDays</td>
      <td>1.403033e-01</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ScheduledDayDay</td>
      <td>7.595050e-02</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AppointmentDayDay</td>
      <td>6.689942e-02</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LastAppointmentDays</td>
      <td>5.382999e-02</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ScheduledDayOfWeek</td>
      <td>4.527465e-02</td>
    </tr>
    <tr>
      <th>12</th>
      <td>AppointmentDayOfWeek</td>
      <td>4.056236e-02</td>
    </tr>
    <tr>
      <th>15</th>
      <td>AppointmentCount</td>
      <td>3.916821e-02</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Gender</td>
      <td>2.841735e-02</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NoShowRate</td>
      <td>1.892474e-02</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SmsReceived</td>
      <td>1.657221e-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Scholarship</td>
      <td>1.104533e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hypertension</td>
      <td>1.059220e-02</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LastShowStatus</td>
      <td>1.054305e-02</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Neighbourhood_JARDIM CAMBURI</td>
      <td>8.952422e-03</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Neighbourhood_MARIA ORTIZ</td>
      <td>8.604134e-03</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Neighbourhood_RESISTÊNCIA</td>
      <td>7.887819e-03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Diabetes</td>
      <td>7.644763e-03</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Neighbourhood_CENTRO</td>
      <td>7.156669e-03</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Neighbourhood_JARDIM DA PENHA</td>
      <td>6.248983e-03</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Neighbourhood_BONFIM</td>
      <td>6.204262e-03</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Neighbourhood_TABUAZEIRO</td>
      <td>6.033757e-03</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Neighbourhood_ITARARÉ</td>
      <td>5.917283e-03</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Neighbourhood_SÃO PEDRO</td>
      <td>5.841644e-03</td>
    </tr>
    <tr>
      <th>86</th>
      <td>Neighbourhood_SANTO ANTÔNIO</td>
      <td>5.591699e-03</td>
    </tr>
    <tr>
      <th>85</th>
      <td>Neighbourhood_SANTO ANDRÉ</td>
      <td>5.472483e-03</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Alcoholism</td>
      <td>5.349752e-03</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Neighbourhood_ANDORINHAS</td>
      <td>5.306941e-03</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Neighbourhood_CARATOÍRA</td>
      <td>5.289505e-03</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Neighbourhood_DA PENHA</td>
      <td>5.260573e-03</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Neighbourhood_ILHA DO PRÍNCIPE</td>
      <td>5.201232e-03</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Neighbourhood_SANTA MARTHA</td>
      <td>5.088866e-03</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Neighbourhood_JABOUR</td>
      <td>5.032229e-03</td>
    </tr>
    <tr>
      <th>77</th>
      <td>Neighbourhood_ROMÃO</td>
      <td>5.022522e-03</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Neighbourhood_NOVA PALESTINA</td>
      <td>4.912082e-03</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Neighbourhood_JESUS DE NAZARETH</td>
      <td>4.907973e-03</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Neighbourhood_MARUÍPE</td>
      <td>4.841603e-03</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Neighbourhood_BELA VISTA</td>
      <td>4.762411e-03</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Neighbourhood_SÃO JOSÉ</td>
      <td>4.707066e-03</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Handicap</td>
      <td>4.700585e-03</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Neighbourhood_SÃO CRISTÓVÃO</td>
      <td>4.677261e-03</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Neighbourhood_ILHA DE SANTA MARIA</td>
      <td>4.673624e-03</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Neighbourhood_FORTE SÃO JOÃO</td>
      <td>4.502472e-03</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Neighbourhood_GURIGICA</td>
      <td>4.337711e-03</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Neighbourhood_SANTOS DUMONT</td>
      <td>4.156303e-03</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Neighbourhood_CRUZAMENTO</td>
      <td>4.040916e-03</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Neighbourhood_PRAIA DO SUÁ</td>
      <td>3.847197e-03</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Neighbourhood_JOANA D´ARC</td>
      <td>3.846516e-03</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Neighbourhood_REDENÇÃO</td>
      <td>3.715672e-03</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Neighbourhood_SANTA TEREZA</td>
      <td>3.608642e-03</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Neighbourhood_SÃO BENEDITO</td>
      <td>3.529670e-03</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Neighbourhood_CONSOLAÇÃO</td>
      <td>3.470228e-03</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Neighbourhood_GRANDE VITÓRIA</td>
      <td>3.314318e-03</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Neighbourhood_ILHA DAS CAIEIRAS</td>
      <td>3.305864e-03</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Neighbourhood_INHANGUETÁ</td>
      <td>3.089079e-03</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Neighbourhood_BENTO FERREIRA</td>
      <td>2.901508e-03</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Neighbourhood_PRAIA DO CANTO</td>
      <td>2.896194e-03</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Neighbourhood_MONTE BELO</td>
      <td>2.792688e-03</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Neighbourhood_CONQUISTA</td>
      <td>2.737942e-03</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Neighbourhood_PARQUE MOSCOSO</td>
      <td>2.719188e-03</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Neighbourhood_REPÚBLICA</td>
      <td>2.678750e-03</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Neighbourhood_GOIABEIRAS</td>
      <td>2.479552e-03</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Neighbourhood_VILA RUBIM</td>
      <td>2.340987e-03</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Neighbourhood_DO QUADRO</td>
      <td>2.336906e-03</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Neighbourhood_FONTE GRANDE</td>
      <td>2.227330e-03</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Neighbourhood_SANTA CLARA</td>
      <td>2.221243e-03</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Neighbourhood_JUCUTUQUARA</td>
      <td>2.125099e-03</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Neighbourhood_MATA DA PRAIA</td>
      <td>1.993604e-03</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Neighbourhood_SANTA CECÍLIA</td>
      <td>1.930582e-03</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Neighbourhood_SANTOS REIS</td>
      <td>1.904374e-03</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Neighbourhood_ESTRELINHA</td>
      <td>1.841227e-03</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Neighbourhood_BARRO VERMELHO</td>
      <td>1.737657e-03</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Neighbourhood_DO CABRAL</td>
      <td>1.634919e-03</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Neighbourhood_SANTA LÚCIA</td>
      <td>1.553069e-03</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Neighbourhood_PIEDADE</td>
      <td>1.534364e-03</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Neighbourhood_SOLON BORGES</td>
      <td>1.473676e-03</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Neighbourhood_DO MOSCOSO</td>
      <td>1.449518e-03</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Neighbourhood_SANTA LUÍZA</td>
      <td>1.438040e-03</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Neighbourhood_ARIOVALDO FAVALESSA</td>
      <td>1.281974e-03</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Neighbourhood_BOA VISTA</td>
      <td>1.239128e-03</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Neighbourhood_ENSEADA DO SUÁ</td>
      <td>1.235900e-03</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Neighbourhood_MÁRIO CYPRESTE</td>
      <td>1.209967e-03</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Neighbourhood_DE LOURDES</td>
      <td>1.017371e-03</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Neighbourhood_FRADINHOS</td>
      <td>9.981958e-04</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Neighbourhood_COMDUSA</td>
      <td>9.853778e-04</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Neighbourhood_ANTÔNIO HONÓRIO</td>
      <td>9.436609e-04</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Neighbourhood_HORTO</td>
      <td>8.640116e-04</td>
    </tr>
    <tr>
      <th>80</th>
      <td>Neighbourhood_SANTA HELENA</td>
      <td>8.594130e-04</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Neighbourhood_UNIVERSITÁRIO</td>
      <td>6.938000e-04</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Neighbourhood_NAZARETH</td>
      <td>6.218302e-04</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Neighbourhood_SEGURANÇA DO LAR</td>
      <td>6.161107e-04</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Neighbourhood_MORADA DE CAMBURI</td>
      <td>5.226169e-04</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Neighbourhood_PONTAL DE CAMBURI</td>
      <td>3.003572e-04</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Neighbourhood_ILHA DO BOI</td>
      <td>1.290846e-04</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Neighbourhood_ILHAS OCEÂNICAS DE TRINDADE</td>
      <td>1.011864e-04</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Neighbourhood_ILHA DO FRADE</td>
      <td>8.280124e-05</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Neighbourhood_AEROPORTO</td>
      <td>4.535425e-05</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Neighbourhood_PARQUE INDUSTRIAL</td>
      <td>4.254570e-09</td>
    </tr>
  </tbody>
</table>
</div>



### Printing the importance of each column


```python
import matplotlib.pyplot as plt
```


```python
plt.style.use('ggplot')
```


```python
feature_importances = pd.DataFrame(clf.feature_importances_,
 index = predictors,
 columns=['importance']).sort_values('importance',
 ascending=False)
num = min([50,len(predictors)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = feature_importances.iloc[:num].values.ravel()[::-1]
feature_labels = list(feature_importances.iloc[:num].index)[::-1]
plt.figure(num=None, figsize=(6, 18), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align = 'center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Feature Importance Score — Random Forest')
plt.yticks(ylocs, feature_labels)
plt.show()
```


    
![png](output_122_0.png)
    


### The plot above shows the importance of each column for our model
## <span style="color:blue">**Which classifier obtains the highest performance?**</span>
### H2o declared Light GBM as the best model, let's try it.
#  <span style="color:blue">**Light GBM Classifier** </span>
If you're working for the first time with Light GBM, you should first install it.

#### Light GBM doesn't work with non ASCII characters, so we need to drop these letters from column names. 


```python
# Need to rename columns here, because Light GBM doesnt like non_ASCII column names
cols_l = df_ml.columns
cols_l_new = []
for col_name in cols_l:
    col_name_new = str(col_name).encode("ascii", "ignore").decode()
    cols_l_new.append(col_name_new)
    
df_ml.columns = cols_l_new
predictors = df_ml.columns[1:]
```


```python
X = df_ml[predictors]
Y = df_ml[target]


X_train, X_test, Y_train, Y_test= \
train_test_split(X,Y,test_size=0.3,random_state = 0)
```

## <span style="color:blue">**Hyper Parameter Tuning**</span>
### We have tried many different combinations of parameters manualy to get the best result.
These parameters are the best!


```python
import lightgbm

clf_lgbm = lightgbm.LGBMClassifier(
    n_estimators=500,
    n_jobs=-1,
    num_leaves = 51,
    objective = 'binary',
    learning_rate = 0.065,
    feature_fraction = 0.7,
    metric = 'auc'
)
```

Note-<br>
The next cell takes a while to run


```python
%%time
clf_lgbm.fit(X_train, Y_train)

```

    CPU times: user 11.4 s, sys: 138 ms, total: 11.6 s
    Wall time: 3.04 s
    




    LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                   feature_fraction=0.7, importance_type='split',
                   learning_rate=0.065, max_depth=-1, metric='auc',
                   min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                   n_estimators=500, n_jobs=-1, num_leaves=51, objective='binary',
                   random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                   subsample=1.0, subsample_for_bin=200000, subsample_freq=0)




```python
y_pred_lgbm = clf_lgbm.predict(X_test)
```


```python
y_pred_proba_lgbm = clf_lgbm.predict_proba(X_test)[:,1]
```

Accuracy-


```python
round(met.accuracy_score(Y_test, y_pred_lgbm), 4)
```




    0.8094



AUC Score-


```python
round(met.roc_auc_score(Y_test, y_pred_proba_lgbm), 4)
```




    0.7609



Light GBM is a super fast and accurate model, Let's compare it with a highly tuned XGBoost model!

#  <span style="color:blue">**XGBoost Classifier** </span>

### XGBoost is one the most powerful models

If you're working for the first time with XGBoost, you should first instal it.

pip install xgboost


```python
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
```


```python
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import scipy as sp 
from sklearn.model_selection import RandomizedSearchCV
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
```

## <span style="color:blue">**Hyper Parameter Tuning**</span>

### We're going to find the best parameter for XGBoost 

Note-<br>
The next cell takes a while to run


```python
%%time
from sklearn.model_selection import StratifiedKFold

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    clf = XGBClassifier(
        n_estimators=100,
        n_jobs=-1,
        **params
    )

    score = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=StratifiedKFold()).mean()
    print("Accuracy {:.8f} params {}".format(-score, params))
    return -score

space = {
    'max_depth': hp.quniform('max_depth', 2, 8, 1),
    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
    'reg_lambda': hp.uniform('reg_lambda', 0.7, 1.0),
    'learning_rate': hp.uniform('learning_rate', 0.05, 0.2),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50)

```

    Accuracy -0.80741938 params {'max_depth': 4, 'gamma': '0.062', 'reg_alpha': '0.215', 'learning_rate': '0.081', 'colsample_bytree': '0.490'}
    Accuracy -0.80597169 params {'max_depth': 2, 'gamma': '0.076', 'reg_alpha': '0.313', 'learning_rate': '0.084', 'colsample_bytree': '0.528'}
    Accuracy -0.80810444 params {'max_depth': 5, 'gamma': '0.340', 'reg_alpha': '0.313', 'learning_rate': '0.116', 'colsample_bytree': '0.349'}
    Accuracy -0.80560977 params {'max_depth': 2, 'gamma': '0.467', 'reg_alpha': '0.187', 'learning_rate': '0.147', 'colsample_bytree': '0.681'}
    Accuracy -0.80829833 params {'max_depth': 8, 'gamma': '0.112', 'reg_alpha': '0.022', 'learning_rate': '0.120', 'colsample_bytree': '1.000'}
    Accuracy -0.80772959 params {'max_depth': 3, 'gamma': '0.120', 'reg_alpha': '0.130', 'learning_rate': '0.192', 'colsample_bytree': '0.821'}
    Accuracy -0.80853099 params {'max_depth': 6, 'gamma': '0.008', 'reg_alpha': '0.258', 'learning_rate': '0.177', 'colsample_bytree': '0.644'}
    Accuracy -0.80590706 params {'max_depth': 2, 'gamma': '0.286', 'reg_alpha': '0.219', 'learning_rate': '0.082', 'colsample_bytree': '0.475'}
    Accuracy -0.80845344 params {'max_depth': 7, 'gamma': '0.283', 'reg_alpha': '0.087', 'learning_rate': '0.146', 'colsample_bytree': '0.591'}
    Accuracy -0.80787178 params {'max_depth': 6, 'gamma': '0.313', 'reg_alpha': '0.307', 'learning_rate': '0.195', 'colsample_bytree': '0.432'}
    Accuracy -0.80629484 params {'max_depth': 3, 'gamma': '0.226', 'reg_alpha': '0.132', 'learning_rate': '0.062', 'colsample_bytree': '0.492'}
    Accuracy -0.80855684 params {'max_depth': 6, 'gamma': '0.287', 'reg_alpha': '0.305', 'learning_rate': '0.142', 'colsample_bytree': '0.412'}
    Accuracy -0.80850514 params {'max_depth': 7, 'gamma': '0.218', 'reg_alpha': '0.181', 'learning_rate': '0.051', 'colsample_bytree': '0.971'}
    Accuracy -0.80866025 params {'max_depth': 7, 'gamma': '0.018', 'reg_alpha': '0.234', 'learning_rate': '0.053', 'colsample_bytree': '0.709'}
    Accuracy -0.80856977 params {'max_depth': 5, 'gamma': '0.070', 'reg_alpha': '0.019', 'learning_rate': '0.176', 'colsample_bytree': '0.992'}
    Accuracy -0.80582951 params {'max_depth': 2, 'gamma': '0.000', 'reg_alpha': '0.174', 'learning_rate': '0.053', 'colsample_bytree': '0.825'}
    Accuracy -0.80850514 params {'max_depth': 7, 'gamma': '0.427', 'reg_alpha': '0.261', 'learning_rate': '0.102', 'colsample_bytree': '0.987'}
    Accuracy -0.80858269 params {'max_depth': 7, 'gamma': '0.412', 'reg_alpha': '0.302', 'learning_rate': '0.080', 'colsample_bytree': '0.362'}
    Accuracy -0.80867317 params {'max_depth': 7, 'gamma': '0.015', 'reg_alpha': '0.392', 'learning_rate': '0.150', 'colsample_bytree': '0.574'}
    Accuracy -0.80849221 params {'max_depth': 7, 'gamma': '0.282', 'reg_alpha': '0.094', 'learning_rate': '0.140', 'colsample_bytree': '0.368'}
    Accuracy -0.80854392 params {'max_depth': 8, 'gamma': '0.168', 'reg_alpha': '0.400', 'learning_rate': '0.164', 'colsample_bytree': '0.741'}
    Accuracy -0.80838881 params {'max_depth': 8, 'gamma': '0.014', 'reg_alpha': '0.388', 'learning_rate': '0.162', 'colsample_bytree': '0.764'}
    Accuracy -0.80847929 params {'max_depth': 6, 'gamma': '0.164', 'reg_alpha': '0.360', 'learning_rate': '0.105', 'colsample_bytree': '0.580'}
    Accuracy -0.80850514 params {'max_depth': 8, 'gamma': '0.045', 'reg_alpha': '0.362', 'learning_rate': '0.133', 'colsample_bytree': '0.879'}
    Accuracy -0.80838881 params {'max_depth': 5, 'gamma': '0.124', 'reg_alpha': '0.261', 'learning_rate': '0.158', 'colsample_bytree': '0.678'}
    Accuracy -0.80873780 params {'max_depth': 6, 'gamma': '0.040', 'reg_alpha': '0.350', 'learning_rate': '0.128', 'colsample_bytree': '0.601'}
    Accuracy -0.80789763 params {'max_depth': 4, 'gamma': '0.152', 'reg_alpha': '0.351', 'learning_rate': '0.129', 'colsample_bytree': '0.575'}
    Accuracy -0.80819492 params {'max_depth': 6, 'gamma': '0.091', 'reg_alpha': '0.396', 'learning_rate': '0.106', 'colsample_bytree': '0.301'}
    Accuracy -0.80798811 params {'max_depth': 4, 'gamma': '0.209', 'reg_alpha': '0.341', 'learning_rate': '0.183', 'colsample_bytree': '0.634'}
    Accuracy -0.80858269 params {'max_depth': 6, 'gamma': '0.364', 'reg_alpha': '0.381', 'learning_rate': '0.093', 'colsample_bytree': '0.527'}
    Accuracy -0.80841466 params {'max_depth': 5, 'gamma': '0.042', 'reg_alpha': '0.336', 'learning_rate': '0.151', 'colsample_bytree': '0.615'}
    Accuracy -0.80867317 params {'max_depth': 8, 'gamma': '0.045', 'reg_alpha': '0.270', 'learning_rate': '0.122', 'colsample_bytree': '0.535'}
    Accuracy -0.80854392 params {'max_depth': 6, 'gamma': '0.492', 'reg_alpha': '0.370', 'learning_rate': '0.133', 'colsample_bytree': '0.461'}
    Accuracy -0.80798811 params {'max_depth': 5, 'gamma': '0.086', 'reg_alpha': '0.329', 'learning_rate': '0.173', 'colsample_bytree': '0.522'}
    Accuracy -0.80652750 params {'max_depth': 3, 'gamma': '0.045', 'reg_alpha': '0.290', 'learning_rate': '0.121', 'colsample_bytree': '0.301'}
    Accuracy -0.80890584 params {'max_depth': 7, 'gamma': '0.185', 'reg_alpha': '0.281', 'learning_rate': '0.113', 'colsample_bytree': '0.778'}
    Accuracy -0.80844051 params {'max_depth': 8, 'gamma': '0.184', 'reg_alpha': '0.230', 'learning_rate': '0.066', 'colsample_bytree': '0.939'}
    Accuracy -0.80797518 params {'max_depth': 4, 'gamma': '0.138', 'reg_alpha': '0.204', 'learning_rate': '0.113', 'colsample_bytree': '0.791'}
    Accuracy -0.80827247 params {'max_depth': 6, 'gamma': '0.253', 'reg_alpha': '0.286', 'learning_rate': '0.092', 'colsample_bytree': '0.690'}
    Accuracy -0.80871195 params {'max_depth': 7, 'gamma': '0.103', 'reg_alpha': '0.242', 'learning_rate': '0.074', 'colsample_bytree': '0.879'}
    Accuracy -0.80867317 params {'max_depth': 6, 'gamma': '0.359', 'reg_alpha': '0.324', 'learning_rate': '0.113', 'colsample_bytree': '0.856'}
    Accuracy -0.80828540 params {'max_depth': 5, 'gamma': '0.250', 'reg_alpha': '0.203', 'learning_rate': '0.095', 'colsample_bytree': '0.927'}
    Accuracy -0.80950042 params {'max_depth': 8, 'gamma': '0.328', 'reg_alpha': '0.155', 'learning_rate': '0.088', 'colsample_bytree': '0.726'}
    Accuracy -0.80900924 params {'max_depth': 8, 'gamma': '0.329', 'reg_alpha': '0.058', 'learning_rate': '0.087', 'colsample_bytree': '0.729'}
    Accuracy -0.80913850 params {'max_depth': 8, 'gamma': '0.412', 'reg_alpha': '0.052', 'learning_rate': '0.065', 'colsample_bytree': '0.721'}
    Accuracy -0.80924191 params {'max_depth': 8, 'gamma': '0.412', 'reg_alpha': '0.153', 'learning_rate': '0.061', 'colsample_bytree': '0.672'}
    Accuracy -0.80687649 params {'max_depth': 3, 'gamma': '0.451', 'reg_alpha': '0.150', 'learning_rate': '0.060', 'colsample_bytree': '0.675'}
    Accuracy -0.80887998 params {'max_depth': 8, 'gamma': '0.383', 'reg_alpha': '0.111', 'learning_rate': '0.075', 'colsample_bytree': '0.828'}
    Accuracy -0.80867317 params {'max_depth': 7, 'gamma': '0.489', 'reg_alpha': '0.165', 'learning_rate': '0.059', 'colsample_bytree': '0.638'}
    Accuracy -0.80858269 params {'max_depth': 8, 'gamma': '0.308', 'reg_alpha': '0.129', 'learning_rate': '0.071', 'colsample_bytree': '0.792'}
    100%|██████████| 50/50 [1:43:17<00:00, 123.95s/trial, best loss: -0.8095004200866025]
    CPU times: user 1h 42min 55s, sys: 21.9 s, total: 1h 43min 17s
    Wall time: 1h 43min 17s
    


```python
best['max_depth'] = int(best['max_depth'])

print("BEST PARAMS: ", best)
```

    BEST PARAMS:  {'colsample_bytree': 0.7261873262077718, 'gamma': 0.3281358263269441, 'learning_rate': 0.08807305921408945, 'max_depth': 8, 'reg_alpha': 0.15468018968513403, 'reg_lambda': 0.9432992526356866}
    


```python
clf_xgb = XGBClassifier(
        n_estimators=600,
        n_jobs=-1,
        **best
    )
```

Note-<br>
The next cell takes a while to run


```python
%%time

clf_xgb.fit(X_train, Y_train)
```

    CPU times: user 4min 5s, sys: 142 ms, total: 4min 5s
    Wall time: 4min 5s
    




    XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.7261873262077718,
                  gamma=0.3281358263269441, gpu_id=-1, importance_type='gain',
                  interaction_constraints=None, learning_rate=0.08807305921408945,
                  max_delta_step=0, max_depth=8, min_child_weight=1, missing=nan,
                  monotone_constraints=None, n_estimators=600, n_jobs=-1,
                  num_parallel_tree=1, objective='binary:logistic', random_state=0,
                  reg_alpha=0.15468018968513403, reg_lambda=0.9432992526356866,
                  scale_pos_weight=1, subsample=1, tree_method=None,
                  validate_parameters=False, verbosity=None)




```python
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

y_pred_xgb = clf_xgb.predict(X_test)
y_pred_proba_xgb = clf_xgb.predict_proba(X_test)[:,1]
```

Accuracy-


```python
round(met.accuracy_score(Y_test, y_pred_xgb), 4)
```




    0.8061



AUC Score-


```python
round(met.roc_auc_score(Y_test, y_pred_proba_xgb), 4)
```




    0.7582



XGBoost is a very powerful model

# <span>**Conclusion:** </span>
## <span>*Light GBM Classifier** </span> <span>**is the most accurate model.** </span>


### <span>*Accuracy: 0.809* </span>
### <span>*AUC: 0.761* </span>

### Plot ROC curve


```python
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import sklearn as sk
import seaborn as sns  
from sklearn.datasets import make_classification  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import roc_curve
```


```python
plt.style.use('dark_background')
```


```python
fper, tper, thresholds = roc_curve(Y_test, y_pred_proba_lgbm) 
plt.plot(fper, tper, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
```


    
![png](output_161_0.png)
    


## Summary of machine learning:

The first model we created gave us the AUC = 0.7441
But then we did some feature engineering and try different models (Light GBM and XGBoost) and tune their parameters and reached the AUC of 0.7609


## Prediction
<br>
After creating the prediction model with AUC of 0.76, we realised that showing up for an appointment is not easy to predict.

All analysis to solve this problem results in AUC around 0.75 which makes a lot of sense because at the end of the day we, humans are unpredictable. A lot of factors may cause a person not showing up for an appointment (for example a sick child or an emergency issue)

### Working with H2O:

We used H2O to find out the best ML Predictive model for our dataset. We found that Light GBM and XGBoost were the most accurate models. This helped us to devote more time on making the existing ML model more efficient. Once we were sure about the model to work on, we coupled feature engineering with hyper-parameter tuning to obtain our desired outcome.
The interface was user friendly and the in-built tutorials helped us to locate various functionalities easily which we wished to perform.

### Thank You
