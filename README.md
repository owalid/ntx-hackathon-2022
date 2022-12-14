<div align="center">
  <img width="700" src="https://user-images.githubusercontent.com/28403617/199127256-6ee22579-c24d-4f1c-be1c-fca68cf25af5.svg"/>
</div>

- [Introduction](#introduction)
- [Issue](#issue)
- [How the experiment works](#how-the-experiment-works)
- [Analysis](#analysis)
- [Modelization](#modelization)
- [Conclusion](#conclusion)

## Introduction

We started with a simple question: It's possible to measured the impact of music on the brain ?

So we conducted an experiment

## Issue

Does music have an impact on our signals ?

Does it allow us to relax ?


## How the experiment works

We have acquired data from a bitalino kit and using the Timeflux tool. We focused on measuring the brain with an EEG sensor.
The whole experience is spent in calmness, with eyes closed. (To remove the maximum amount of external interference)

![20221030_141859](https://user-images.githubusercontent.com/28403617/199126599-168464dd-cccb-4068-a5f3-7f95b2a1794d.jpg)

3 different stimuli randomly triggered by an operator. 2 times per subject. for 3 subjects in total.
A data collection takes an average of 10 minutes

We had 3 different stimulation:

- Baseline status: Eyes closed without music.
- Active listening to relaxing music.
- Active listening to stressful music.

Each music was chosen by the subject, because the notion of relaxing and stressful music is subjective to each person.

## Analysis

With our data we tried to find curves or breaks between the states but we didn't really find anything.

<img width="1423" alt="Capture_decran_2022-10-30_a_14 33 02" src="https://user-images.githubusercontent.com/28403617/199126736-2a4e1a92-012c-41e4-b632-f04858636023.png">


<img width="1677" alt="Capture_decran_2022-10-30_a_14 31 20" src="https://user-images.githubusercontent.com/28403617/199127469-f130354b-080f-4f5e-a1d5-a2d1930019ec.png">


## Modelization

Many models are tested:

<strong>Machine Learning:</strong>
- non supervised : KNeighborsClassifier, KMeans
- Supervised : XGBoost

<strong>Deep Learning:</strong>
- CNN-LSTM

Results : mitigated results with a score of <strong>56%</strong> for the xgboost.


## Conclusion

Currently we have not had good results and this can be explained by several reasons:
- A lack of data
- Lack of different panels
- An identical protocol for each person
- Lack of electrode etc.


We could have had better results, if we had had a lot more electrode, more data and a different panel of people.


### Tools and links

#### Tools
- [Bitalino](https://www.pluxbiosignals.com/collections/bitalino)
- [EEG](https://www.pluxbiosignals.com/products/electroencephalography-eeg)
- [Timeflux](https://timeflux.io/)


#### Links

- [Keras timeseries example](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)
- [Recognizing Emotions using EEG signals](https://www.researchgate.net/publication/343250071_Recognizing_Emotions_Evoked_by_Music_using_CNN-LSTM_Networks_on_EEG_signals)
- [Notion notes and links during hackathon](https://beneficial-thunbergia-ab4.notion.site/NTX-2022-0b305223dfeb412d980c049caf880097)
