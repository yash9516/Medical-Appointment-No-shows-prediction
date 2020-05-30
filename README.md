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
- Analyzed data and plotted graphs using Interactive Visualizations (IPyWidgets, Plotly).
-Utilized <strong>Python widgets</strong> to consolidate all the graphs in a <strong>drop-down menu style</strong>.
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
